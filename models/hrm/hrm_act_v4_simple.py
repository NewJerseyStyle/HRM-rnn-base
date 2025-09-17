"""HRM v4 Simple - Iterative refinement without complex diffusion."""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.local_sru import SRU
from models.common import trunc_normal_init_
from models.layers import (
    CastedLinear, CastedEmbedding, RotaryEmbedding, 
    Attention, SwiGLU, rms_norm
)
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV4SimpleInnerCarry:
    z_H: torch.Tensor  # High-level state
    z_L: torch.Tensor  # Low-level SRU state
    refinement_step: torch.Tensor  # Current refinement step


@dataclass
class HierarchicalReasoningModel_ACTV4SimpleCarry:
    inner_carry: HierarchicalReasoningModel_ACTV4SimpleInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV4SimpleConfig(BaseModel):
    # Standard HRM config
    vocab_size: int
    seq_len: int = 81  # Default to standard Sudoku length
    hidden_size: int
    num_heads: int
    expansion: int
    L_layers: int
    H_layers: int = 4  # Number of refinement blocks
    
    # Refinement parameters
    refinement_steps: int = 5  # Number of iterative refinement steps
    
    # Puzzle embeddings
    puzzle_emb_ndim: int
    num_puzzle_identifiers: int = 4000
    batch_size: int
    
    # Position encodings
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0
    
    # Halting config
    halt_max_steps: int
    halt_exploration_prob: float
    
    forward_dtype: str = "float16"


class SimpleRefinementBlock(nn.Module):
    """Simple refinement block using self-attention and feedforward."""
    
    def __init__(self, hidden_size: int, num_heads: int, expansion: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, head_dim, num_heads, num_heads, causal=False)
        
        # Feedforward
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = SwiGLU(hidden_size, hidden_size * expansion)
        
        # Gating for iterative refinement
        self.gate = CastedLinear(hidden_size * 2, hidden_size, bias=True)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual (no rotary embeddings for simplicity)
        attn_out = self.attn(None, self.norm1(x))
        x = x + attn_out
        
        # Feedforward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        # Gate with context (from L_level)
        gate_input = torch.cat([x, context], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        
        # Gated combination
        x = gate * x + (1 - gate) * context
        
        return x


class IterativeRefinementHLevel(nn.Module):
    """H_level using simple iterative refinement."""
    
    def __init__(self, config):
        super().__init__()
        # Handle both dict and BaseModel config
        if isinstance(config, dict):
            self.config = HierarchicalReasoningModel_ACTV4SimpleConfig(**config) if not isinstance(config, HierarchicalReasoningModel_ACTV4SimpleConfig) else config
        else:
            self.config = config
        
        # Refinement blocks
        self.blocks = nn.ModuleList([
            SimpleRefinementBlock(
                self.config.hidden_size,
                self.config.num_heads,
                self.config.expansion
            )
            for _ in range(self.config.H_layers)
        ])
        
        # Step embedding
        self.step_embed = nn.Embedding(self.config.refinement_steps, self.config.hidden_size)
        
    def forward(
        self, 
        z_L: torch.Tensor,
        z_H: Optional[torch.Tensor] = None,
        step: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_L: Low-level features (batch, seq_len, hidden_size)
            z_H: Previous high-level state or None for initialization
            step: Current refinement step
        Returns:
            z_H: Refined high-level state
            step: Next step
        """
        batch_size, seq_len, hidden_size = z_L.shape
        device = z_L.device
        
        # Initialize if needed
        if z_H is None:
            z_H = z_L.clone()  # Start from L_level output
            step = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Add step embedding
        step_emb = self.step_embed(step).unsqueeze(1)  # (batch, 1, hidden)
        z_H = z_H + step_emb
        
        # Apply refinement blocks
        for block in self.blocks:
            z_H = block(z_H, z_L)
        
        # Increment step (with wraparound)
        step = (step + 1) % self.config.refinement_steps
        
        return z_H, step


class HierarchicalReasoningModel_ACTV4Simple_Inner(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Handle both dict and BaseModel config
        if isinstance(config, dict):
            self.config = HierarchicalReasoningModel_ACTV4SimpleConfig(**config)
        else:
            self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        
        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, 
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype
        )
        
        # Puzzle embeddings
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype
            )
        
        # Position encodings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta
            )
        
        # L_level: SRU for fast sequential processing
        self.L_level = SRU(
            self.config.hidden_size,
            self.config.hidden_size,
            num_layers=self.config.L_layers
        )
        
        # H_level: Simple iterative refinement
        self.H_level = IterativeRefinementHLevel(self.config)
        
        # Initial states
        device = torch.device('cpu')
        self.L_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(
                    self.config.L_layers, 1, self.config.hidden_size,
                    dtype=self.forward_dtype, device=device
                ), 
                std=1
            ),
            persistent=True
        )
        
        # Output heads
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        
        # Q head initialization
        with torch.no_grad():
            self.q_head.weight.data *= 0.01
            self.q_head.bias.data[0] = 5.0  # Bias towards halting
            self.q_head.bias.data[1] = 0.0
    
    def process_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process input embeddings with puzzle embeddings."""
        B = batch["inputs"].shape[0]
        input_embeds = self.embed_tokens(batch["inputs"])
        
        if self.config.puzzle_emb_ndim > 0:
            if "puzzle_id" in batch:
                p_embeds = self.puzzle_emb(batch["puzzle_id"])
                p_embeds = p_embeds.view(B, self.puzzle_emb_len, self.config.hidden_size)
                input_embeds = torch.cat([p_embeds, input_embeds], dim=1)
            else:
                p_embeds = input_embeds.new_zeros(B, self.puzzle_emb_len, self.config.hidden_size)
                input_embeds = torch.cat([p_embeds, input_embeds], dim=1)
        
        return self.embed_scale * input_embeds
    
    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV4SimpleInnerCarry(
            z_H=None,  # Will be initialized in H_level
            z_L=self.L_init.expand(-1, batch_size, -1),
            refinement_step=None  # Will be initialized in H_level
        )
    
    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV4SimpleInnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV4SimpleInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Get embeddings
        input_embeddings = self.process_embeddings(batch)
        
        # L_level processing (SRU)
        # Transpose for SRU (expects seq_len, batch, hidden)
        input_embeddings_t = input_embeddings.transpose(0, 1)
        z_L, z_L_hidden = self.L_level(input_embeddings_t, carry.z_L)
        z_L = z_L.transpose(0, 1)  # Back to (batch, seq_len, hidden)
        
        # H_level refinement
        z_H, next_step = self.H_level(z_L, carry.z_H, carry.refinement_step)
        
        # Combine H and L levels
        z_combined = z_L + z_H  # Simple addition
        
        # Apply RMS norm with epsilon for numerical stability
        z_combined = rms_norm(z_combined, variance_epsilon=1e-5)
        
        # Generate outputs
        logits = self.lm_head(z_combined)
        
        # Q-values for halting
        pooled = z_combined.mean(dim=1)
        q_logits = self.q_head(pooled)
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]
        
        # Extract only sequence part (remove puzzle embeddings)
        if self.config.puzzle_emb_ndim > 0:
            logits = logits[:, self.puzzle_emb_len:, :]
        
        # Update carry
        new_carry = HierarchicalReasoningModel_ACTV4SimpleInnerCarry(
            z_H=z_H,
            z_L=z_L_hidden,
            refinement_step=next_step
        )
        
        return new_carry, logits, (q_halt, q_continue)


class HierarchicalReasoningModel_ACTV4Simple(nn.Module):
    """Simple HRM v4 with iterative refinement."""
    
    def __init__(self, config):
        super().__init__()
        # Handle both dict and BaseModel config
        if isinstance(config, dict):
            self.config = HierarchicalReasoningModel_ACTV4SimpleConfig(**config)
        else:
            self.config = config
        self.inner = HierarchicalReasoningModel_ACTV4Simple_Inner(self.config)
    
    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb if self.config.puzzle_emb_ndim > 0 else None
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        
        return HierarchicalReasoningModel_ACTV4SimpleCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV4SimpleCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV4SimpleCarry, Dict[str, torch.Tensor]]:
        # Reset carry for halted sequences
        reset_mask = carry.halted
        if reset_mask.any():
            # reset_mask shape: (batch,)
            # z_L shape: (L_layers, batch, hidden)
            carry.inner_carry.z_L = torch.where(
                reset_mask.view(1, -1, 1),
                self.inner.L_init.expand(-1, batch["inputs"].shape[0], -1),
                carry.inner_carry.z_L
            )
            if carry.inner_carry.z_H is not None:
                # z_H shape: (batch, seq_len, hidden)
                carry.inner_carry.z_H = torch.where(
                    reset_mask.view(-1, 1, 1),
                    torch.zeros_like(carry.inner_carry.z_H),
                    carry.inner_carry.z_H
                )
            carry.inner_carry.refinement_step = None
            carry.current_data = {
                k: torch.where(reset_mask.view(-1, *[1]*(v.ndim-1)), v, carry.current_data[k])
                for k, v in batch.items()
            }
        
        # Forward pass
        new_inner_carry, logits, (q_halt, q_continue) = self.inner(
            carry.inner_carry,
            carry.current_data
        )
        
        # Halting decision
        should_halt = q_halt > q_continue
        explore = torch.rand_like(q_halt) < self.config.halt_exploration_prob
        should_halt = should_halt | explore
        should_halt = should_halt | (carry.steps >= self.config.halt_max_steps - 1)
        
        # Update carry
        new_carry = HierarchicalReasoningModel_ACTV4SimpleCarry(
            inner_carry=new_inner_carry,
            steps=carry.steps + 1,
            halted=should_halt,
            current_data=carry.current_data
        )
        
        outputs = {
            "logits": logits,
            "q_halt": q_halt,
            "q_continue": q_continue,
            "halted": should_halt,
            "steps": new_carry.steps
        }
        
        return new_carry, outputs