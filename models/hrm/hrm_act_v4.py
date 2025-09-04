"""
HRM v4: Hierarchical Reasoning Model with Diffusion-based High-level Reasoning

This version uses a diffusion model (DDPM/DDIM) for the H_level to enable
powerful iterative refinement of high-level reasoning through denoising steps.
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.local_sru import SRU


class DiffusionBlock(nn.Module):
    """A single diffusion block for denoising."""
    
    def __init__(self, hidden_size: int, num_heads: int, expansion: float):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Self-attention for global reasoning
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False  # Non-causal for diffusion
        )
        
        # Feedforward
        self.mlp = SwiGLU(hidden_size, expansion)
        
        # Time embedding projection
        self.time_embed = nn.Sequential(
            CastedLinear(hidden_size, hidden_size * 2, bias=True),
            nn.SiLU(),
            CastedLinear(hidden_size * 2, hidden_size, bias=True)
        )
        
        # Context projection (to combine L_level info)
        self.context_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        
        # Layer norms
        self.ln1 = lambda x: rms_norm(x, 1e-5)
        self.ln2 = lambda x: rms_norm(x, 1e-5)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, 
                context: torch.Tensor, cos_sin: Optional[CosSin] = None) -> torch.Tensor:
        """
        Args:
            x: Current H_level state [batch, seq_len, hidden_size]
            t_emb: Time embedding [batch, hidden_size]
            context: L_level output for conditioning [batch, seq_len, hidden_size]
            cos_sin: Rotary position embeddings
        """
        # Add time embedding and context
        t_emb_expanded = self.time_embed(t_emb).unsqueeze(1)
        
        # Condition on context by adding it (simple but effective)
        # Make sure context has same shape as x
        if context.shape[1] != x.shape[1]:
            # Pad or truncate context to match x's sequence length
            if context.shape[1] < x.shape[1]:
                context = F.pad(context, (0, 0, 0, x.shape[1] - context.shape[1]))
            else:
                context = context[:, :x.shape[1]]
        
        context_proj = self.context_proj(context)
        x = x + t_emb_expanded + context_proj
        
        # Self-attention
        residual = x
        x = self.ln1(x)
        x = self.self_attn(cos_sin, x) + residual
        
        # Feedforward
        residual = x
        x = self.ln2(x)
        x = self.mlp(x) + residual
        
        return x


class DDIMSampler(nn.Module):
    """DDIM sampler for deterministic, fast inference."""
    
    def __init__(self, num_timesteps: int = 1000, num_inference_steps: int = 50):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps
        
        # Use CPU device to avoid CUDA/TPU issues during initialization
        device = torch.device('cpu')
        
        # Create beta schedule (linear for simplicity, can use cosine)
        betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1.0 - betas, dim=0))
        
        # DDIM sampling timesteps
        step_ratio = num_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_timesteps, step_ratio, device=device)
        self.register_buffer('timesteps', timesteps.flip(0))  # Reverse for denoising
        
    def get_time_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal time embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    @torch.no_grad()
    def sample(self, model: nn.Module, z_L: torch.Tensor, 
               shape: Tuple[int, ...], cos_sin: Optional[CosSin] = None) -> torch.Tensor:
        """
        DDIM sampling for deterministic generation.
        
        Args:
            model: The denoising model
            z_L: L_level output for conditioning
            shape: Shape of the output tensor
            cos_sin: Position embeddings
        """
        device = z_L.device
        batch_size = shape[0]
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device, dtype=z_L.dtype)
        
        for i, t in enumerate(self.timesteps):
            # Time embedding
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_emb = self.get_time_embedding(t_batch.float(), shape[-1])
            
            # Predict noise
            noise_pred = model(x_t, t_emb, z_L, cos_sin)
            
            # DDIM update rule
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[self.timesteps[i+1]] if i < len(self.timesteps)-1 else torch.tensor(1.0)
            
            # Predicted x_0
            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Direction pointing to x_t
            dir_x_t = torch.sqrt(1 - alpha_t_prev) * noise_pred
            
            # DDIM update
            x_t = torch.sqrt(alpha_t_prev) * x_0_pred + dir_x_t
            
        return x_t


class DiffusionHLevel(nn.Module):
    """Diffusion-based H_level for iterative reasoning refinement."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Stack of diffusion blocks
        self.blocks = nn.ModuleList([
            DiffusionBlock(
                config.hidden_size,
                config.num_heads,
                config.expansion
            )
            for _ in range(config.H_layers)
        ])
        
        # DDIM sampler
        self.sampler = DDIMSampler(
            num_timesteps=config.diffusion_timesteps,
            num_inference_steps=config.diffusion_inference_steps
        )
        
        # Rotary embeddings
        if config.pos_encodings == "rope":
            self.rotary = RotaryEmbedding(
                config.hidden_size // config.num_heads,
                config.seq_len + 10,  # Add buffer for puzzle embeddings
                config.rope_theta
            )
        else:
            self.rotary = None
    
    def denoise(self, x: torch.Tensor, t_emb: torch.Tensor, 
                context: torch.Tensor, cos_sin: Optional[CosSin] = None) -> torch.Tensor:
        """Single denoising step through all blocks."""
        for block in self.blocks:
            x = block(x, t_emb, context, cos_sin)
        return x
    
    def forward(self, z_L: torch.Tensor, z_H_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass using DDIM sampling for iterative refinement.
        
        Args:
            z_L: L_level output [batch, seq_len, hidden_size]
            z_H_prev: Previous H_level state (optional, for warm start)
        """
        batch_size, seq_len, hidden_size = z_L.shape
        shape = (batch_size, seq_len, hidden_size)
        
        # Get position embeddings
        cos_sin = self.rotary() if self.rotary else None
        
        # Use DDIM sampling for inference
        z_H = self.sampler.sample(
            lambda x, t, c, cs: self.denoise(x, t, c, cs),
            z_L,
            shape,
            cos_sin
        )
        
        return z_H


@dataclass
class HierarchicalReasoningModel_ACTV4InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV4Carry:
    inner_carry: HierarchicalReasoningModel_ACTV4InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV4Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_layers: int
    L_layers: int
    
    # Diffusion settings
    diffusion_timesteps: int = 1000
    diffusion_inference_steps: int = 50

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_exploration_prob: float = 0.1
    halt_epsilon: float = 0.01
    halt_max_steps: int = 8

    forward_dtype: str = "float16"


class HierarchicalReasoningModel_ACTV4_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV4Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Embeddings
        self.embed_scale = config.hidden_size**0.5
        embed_init_std = 1 / config.hidden_size**0.5
        
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        
        # Puzzle embeddings
        self.puzzle_emb_len = 0
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb_len = 10
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                                                    batch_size=config.batch_size, init_std=0, cast_to=self.forward_dtype)
        else:
            self.puzzle_emb = None
            
        # Position embeddings
        if config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        elif config.pos_encodings == "rope":
            self.embed_pos = None
        else:
            raise NotImplementedError()

        # Reasoning Layers
        # L_level uses SRU for fast sequential processing
        self.L_level = SRU(self.config.hidden_size, self.config.hidden_size, num_layers=self.config.L_layers)
        
        # H_level uses diffusion for iterative refinement
        self.H_level = DiffusionHLevel(self.config)
        
        # Initial states
        # Use CPU device to avoid CUDA/TPU issues during initialization
        device = torch.device('cpu')
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.L_layers, 1, self.config.hidden_size, dtype=self.forward_dtype, device=device), std=1), persistent=True)
        
        # LM Head
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        
        # Q head special init
        with torch.no_grad():
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        # For diffusion H_level, we don't need a fixed initial state
        # It will start from noise
        return HierarchicalReasoningModel_ACTV4InnerCarry(
            z_H=torch.zeros(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, 
                           dtype=self.forward_dtype, device=self.L_init.device),
            z_L=self.L_init.expand(-1, batch_size, -1),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV4InnerCarry):
        return HierarchicalReasoningModel_ACTV4InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), 
                           torch.zeros_like(carry.z_H), carry.z_H),
            z_L=torch.where(reset_flag.view(1, -1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV4InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV4InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        # L_level: SRU expects (seq_len, batch_size, hidden_size) format
        input_embeddings_transposed = input_embeddings.transpose(0, 1)
        z_L, z_L_hidden = self.L_level(input_embeddings_transposed, carry.z_L)
        z_L = z_L.transpose(0, 1)  # Back to (batch_size, seq_len, hidden_size)
        
        # H_level: Diffusion-based iterative refinement
        z_H = self.H_level(z_L, carry.z_H)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV4InnerCarry(z_H=z_H.detach(), z_L=z_L_hidden.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV4(nn.Module):
    """ACT wrapper with diffusion-based H_level."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV4Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV4_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return HierarchicalReasoningModel_ACTV4Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV4Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV4Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            # Decide whether to halt
            halt_prob = torch.sigmoid(q_halt_logits - q_continue_logits).squeeze(-1)
            halt_bernoulli = torch.bernoulli(halt_prob).bool()
            
            # Exploration
            explore_random = torch.rand_like(halt_prob) < self.config.halt_exploration_prob
            explore_bernoulli = torch.bernoulli(torch.ones_like(halt_prob) * 0.5).bool()
            
            # Combine decisions
            halted = torch.where(explore_random, explore_bernoulli, halt_bernoulli)
            halted = halted | is_last_step
            
            # Accumulate the Q value
            if "accumulated_q_halt" not in outputs:
                outputs["accumulated_q_halt"] = q_halt_logits
                outputs["accumulated_q_continue"] = q_continue_logits
            else:
                prev_not_halted = ~carry.halted
                outputs["accumulated_q_halt"] = torch.where(prev_not_halted, 
                                                           outputs["accumulated_q_halt"] + q_halt_logits,
                                                           outputs["accumulated_q_halt"])
                outputs["accumulated_q_continue"] = torch.where(prev_not_halted,
                                                               outputs["accumulated_q_continue"] + q_continue_logits,
                                                               outputs["accumulated_q_continue"])

        return HierarchicalReasoningModel_ACTV4Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
