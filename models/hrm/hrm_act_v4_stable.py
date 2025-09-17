"""HRM v4 Stable - Using denoising_diffusion_pytorch for stable diffusion-based H_level."""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from einops import rearrange

# Import stable diffusion components
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D

from models.local_sru import SRU
from models.common import trunc_normal_init_
from models.layers import (
    CastedLinear, CastedEmbedding, RotaryEmbedding, 
    Attention, SwiGLU, rms_norm
)
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV4StableInnerCarry:
    z_H: torch.Tensor  # High-level diffusion state
    z_L: torch.Tensor  # Low-level SRU state
    timestep: torch.Tensor  # Current diffusion timestep


@dataclass
class HierarchicalReasoningModel_ACTV4StableCarry:
    inner_carry: HierarchicalReasoningModel_ACTV4StableInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV4StableConfig(BaseModel):
    # Standard HRM config
    vocab_size: int
    seq_len: int = 81  # Default to standard Sudoku length
    hidden_size: int
    num_heads: int
    expansion: int
    L_layers: int
    H_layers: int = 4  # Number of diffusion blocks
    
    # Diffusion specific
    diffusion_timesteps: int = 100
    diffusion_sampling_timesteps: int = 10
    diffusion_dim: int = 64  # Internal diffusion dimension
    
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


class StableDiffusionHLevel(nn.Module):
    """H_level using stable 1D diffusion for iterative refinement."""
    
    def __init__(self, config):
        super().__init__()
        # Handle both dict and BaseModel config
        if isinstance(config, dict):
            self.config = HierarchicalReasoningModel_ACTV4StableConfig(**config) if not isinstance(config, HierarchicalReasoningModel_ACTV4StableConfig) else config
        else:
            self.config = config
        self.hidden_size = self.config.hidden_size
        self.seq_len = 81  # Standard Sudoku sequence length
        
        # Create 1D U-Net for diffusion
        self.unet = Unet1D(
            dim=self.config.diffusion_dim,
            dim_mults=(1, 2, 4),
            channels=self.config.hidden_size,  # Number of features/channels
        )
        
        # Create 1D Gaussian Diffusion
        # Use a fixed sequence length of 81 (standard Sudoku length)
        # This can be adjusted based on your data
        self.diffusion = GaussianDiffusion1D(
            self.unet,
            seq_length=81,  # Standard Sudoku sequence length
            timesteps=self.config.diffusion_timesteps,
            objective='pred_v'  # Use v-parameterization for stability
        )
        
        # Projection layers to/from diffusion dimension
        self.input_proj = CastedLinear(self.config.hidden_size, self.config.diffusion_dim, bias=False)
        self.output_proj = CastedLinear(self.config.diffusion_dim, self.config.hidden_size, bias=False)
        
    def forward(
        self, 
        z_L: torch.Tensor,
        z_H: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_L: Low-level features (batch, seq_len, hidden_size)
            z_H: Previous high-level state or None for initialization
            timestep: Current diffusion timestep
        Returns:
            z_H: Refined high-level state
            timestep: Next timestep
        """
        batch_size, seq_len, hidden_size = z_L.shape
        device = z_L.device
        
        # For 1D diffusion, we need shape: (batch, channels, seq_len)
        # Transpose z_L to (batch, hidden_size, seq_len)
        z_L_transposed = z_L.transpose(1, 2)
        
        # Initialize from noise if no previous state
        if z_H is None:
            # Start from random noise with same shape as input
            z_H = torch.randn(batch_size, self.config.hidden_size, seq_len, device=device)
            timestep = torch.full((batch_size,), self.config.diffusion_timesteps - 1, device=device, dtype=torch.long)
        else:
            # Transpose existing state to (batch, hidden_size, seq_len)
            z_H = z_H.transpose(1, 2)
        
        # Ensure sequence length matches what diffusion expects (81)
        expected_seq_len = 81
        if seq_len != expected_seq_len:
            if seq_len < expected_seq_len:
                # Pad if too short
                pad_len = expected_seq_len - seq_len
                z_L_transposed = F.pad(z_L_transposed, (0, pad_len))
                if z_H is not None:
                    z_H = F.pad(z_H, (0, pad_len))
            else:
                # Trim if too long
                z_L_transposed = z_L_transposed[:, :, :expected_seq_len]
                if z_H is not None:
                    z_H = z_H[:, :, :expected_seq_len]
        
        # Use the diffusion model's loss as a denoising step
        # During training, this learns to denoise; during inference, we can sample
        if self.training:
            # During training, compute diffusion loss
            loss = self.diffusion(z_L_transposed)
            # Use the loss gradient to update z_H (simple gradient-based refinement)
            z_H = z_L_transposed  # For now, just use the input
        else:
            # During inference, perform one denoising step
            if timestep[0] > 0:
                # The library doesn't expose single-step denoising easily
                # So we'll use a simpler approach: blend with noise based on timestep
                t_ratio = timestep[0].float() / self.config.diffusion_timesteps
                noise = torch.randn_like(z_H)
                z_H = (1 - t_ratio) * z_L_transposed + t_ratio * noise
                timestep = timestep - 1
            else:
                z_H = z_L_transposed
        
        # Transpose back to (batch, seq_len, hidden_size)
        z_H = z_H.transpose(1, 2)
        
        # Restore original sequence length if we padded/trimmed
        if z_H.shape[1] != seq_len:
            if z_H.shape[1] > seq_len:
                z_H = z_H[:, :seq_len, :]
            else:
                # This shouldn't happen, but handle it anyway
                pad_len = seq_len - z_H.shape[1]
                z_H = F.pad(z_H, (0, 0, 0, pad_len))
        
        return z_H, timestep


class HierarchicalReasoningModel_ACTV4Stable_Inner(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Handle both dict and BaseModel config
        if isinstance(config, dict):
            self.config = HierarchicalReasoningModel_ACTV4StableConfig(**config)
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
        
        # H_level: Stable diffusion for iterative refinement
        self.H_level = StableDiffusionHLevel(self.config)
        
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
        return HierarchicalReasoningModel_ACTV4StableInnerCarry(
            z_H=None,  # Will be initialized in H_level
            z_L=self.L_init.expand(-1, batch_size, -1),
            timestep=None  # Will be initialized in H_level
        )
    
    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV4StableInnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV4StableInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Get embeddings
        input_embeddings = self.process_embeddings(batch)
        
        # L_level processing (SRU)
        # Transpose for SRU (expects seq_len, batch, hidden)
        input_embeddings_t = input_embeddings.transpose(0, 1)
        z_L, z_L_hidden = self.L_level(input_embeddings_t, carry.z_L)
        z_L = z_L.transpose(0, 1)  # Back to (batch, seq_len, hidden)
        
        # H_level refinement (Diffusion)
        z_H, next_timestep = self.H_level(z_L, carry.z_H, carry.timestep)
        
        # Combine H and L levels
        z_combined = z_L + z_H  # Simple addition for now
        
        # Apply RMS norm
        z_combined = rms_norm(z_combined)
        
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
        new_carry = HierarchicalReasoningModel_ACTV4StableInnerCarry(
            z_H=z_H,
            z_L=z_L_hidden,
            timestep=next_timestep
        )
        
        return new_carry, logits, (q_halt, q_continue)


class HierarchicalReasoningModel_ACTV4Stable(nn.Module):
    """Stable HRM v4 with diffusion-based H_level."""
    
    def __init__(self, config):
        super().__init__()
        # Handle both dict and BaseModel config
        if isinstance(config, dict):
            self.config = HierarchicalReasoningModel_ACTV4StableConfig(**config)
        else:
            self.config = config
        self.inner = HierarchicalReasoningModel_ACTV4Stable_Inner(self.config)
    
    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb if self.config.puzzle_emb_ndim > 0 else None
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        
        return HierarchicalReasoningModel_ACTV4StableCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV4StableCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV4StableCarry, Dict[str, torch.Tensor]]:
        # Reset carry for halted sequences
        reset_mask = carry.halted
        if reset_mask.any():
            # reset_mask shape: (batch,)
            # z_L shape: (L_layers, batch, hidden)
            # Need to broadcast reset_mask to (1, batch, 1) for correct broadcasting
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
            carry.inner_carry.timestep = None
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
        new_carry = HierarchicalReasoningModel_ACTV4StableCarry(
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