import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from models.hrm.hrm_act_v2 import (
    HierarchicalReasoningModel_ACTV2 as HRM_ACT_V2,
    HierarchicalReasoningModel_ACTV2Config,
    HierarchicalReasoningModel_ACTV2InnerCarry,
    HierarchicalReasoningModel_ACTV2Carry
)


class ARCHRMv2Model(nn.Module):
    """HRM v2 model adapted for ARC tasks"""

    def __init__(
        self,
        config_path: str,
        max_grid_size: int = 30,
        num_colors: int = 10,
        device: str = 'cuda'
    ):
        super().__init__()

        # Load base config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # ARC-specific configurations
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.pad_token = num_colors  # Use color 10 as padding
        self.device = device

        # Resolve template variables
        if 'puzzle_emb_ndim' in config_dict and isinstance(config_dict['puzzle_emb_ndim'], str):
            if config_dict['puzzle_emb_ndim'] == '${.hidden_size}':
                config_dict['puzzle_emb_ndim'] = config_dict['hidden_size']

        # Update config for ARC
        config_dict['vocab_size'] = num_colors + 3  # 0-9 colors + padding + start/end tokens
        config_dict['batch_size'] = 32  # Set a max batch size for the embedding buffer
        config_dict['seq_len'] = max_grid_size * max_grid_size * 2  # Input + output grids
        config_dict['num_puzzle_identifiers'] = 0  # Disable puzzle embeddings to avoid batch size issues
        config_dict['puzzle_emb_ndim'] = 0  # Disable puzzle embeddings

        # Create HRM v2 model
        self.config = HierarchicalReasoningModel_ACTV2Config(**config_dict)
        self.model = HRM_ACT_V2(config_dict)

        # Output projection for grid prediction
        self.output_proj = nn.Linear(config_dict['hidden_size'], num_colors + 1)  # +1 for padding

        # Special tokens
        self.start_token = num_colors + 1
        self.end_token = num_colors + 2

    def encode_grid_sequence(
        self,
        input_grid: torch.Tensor,
        output_grid: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode grids as sequences for the model
        Returns: (sequence, attention_mask)
        """
        batch_size = input_grid.shape[0]
        device = input_grid.device

        # Flatten grids to sequences
        input_seq = input_grid.view(batch_size, -1)

        if output_grid is not None:
            output_seq = output_grid.view(batch_size, -1)
            # Concatenate with separator token
            sep_token = torch.full((batch_size, 1), self.end_token, device=device, dtype=torch.long)
            full_seq = torch.cat([input_seq, sep_token, output_seq], dim=1)
        else:
            full_seq = input_seq

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (full_seq != self.pad_token).float()

        return full_seq, attention_mask

    def forward(
        self,
        input_grid: torch.Tensor,
        output_grid: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference

        Args:
            input_grid: (batch, H, W) input grids
            output_grid: (batch, H, W) target grids (for training)
            task_ids: (batch,) task identifiers for puzzle embeddings

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        batch_size = input_grid.shape[0]
        device = input_grid.device

        # Encode grids as sequences
        sequence, attention_mask = self.encode_grid_sequence(input_grid, output_grid)

        # Prepare batch dict for HRM model
        # Since puzzle embeddings are disabled, just use zeros
        batch_dict = {
            "inputs": sequence,
            "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=device)
        }

        # Initialize carry state
        carry = self._init_outer_carry(batch_size, sequence.shape[1], device, batch_dict)

        # Forward through HRM model
        carry, outputs_dict = self.model(carry, batch_dict)

        # Get hidden states and project to output space
        hidden = outputs_dict.get("hidden", outputs_dict.get("logits"))
        logits = self.output_proj(hidden)

        result = {'logits': logits}

        # Calculate loss if targets provided
        if output_grid is not None:
            # Extract output portion of logits
            input_len = self.max_grid_size * self.max_grid_size
            output_start = input_len + 1  # +1 for separator token
            output_logits = logits[:, output_start:output_start + input_len]

            # Reshape for grid prediction
            output_logits = output_logits.reshape(
                batch_size, self.max_grid_size, self.max_grid_size, -1
            )

            # Compute loss (ignore padding)
            loss = F.cross_entropy(
                output_logits.permute(0, 3, 1, 2),  # (B, C, H, W)
                output_grid,  # (B, H, W)
                ignore_index=self.pad_token
            )
            result['loss'] = loss

        return result

    def _init_carry(self, batch_size: int, device: torch.device) -> HierarchicalReasoningModel_ACTV2InnerCarry:
        """Initialize inner carry state"""
        hidden_size = self.config.hidden_size
        # Use actual batch size, not config batch size
        z_H = torch.zeros(self.config.H_layers, batch_size, hidden_size, device=device)
        z_L = torch.zeros(self.config.L_layers, batch_size, hidden_size, device=device)
        return HierarchicalReasoningModel_ACTV2InnerCarry(z_H=z_H, z_L=z_L)

    def _init_outer_carry(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        batch_dict: Dict[str, torch.Tensor]
    ) -> HierarchicalReasoningModel_ACTV2Carry:
        """Initialize outer carry state for adaptive computation"""
        inner_carry = self._init_carry(batch_size, device)
        steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        return HierarchicalReasoningModel_ACTV2Carry(
            inner_carry=inner_carry,
            steps=steps,
            halted=halted,
            current_data=batch_dict
        )

    def generate(
        self,
        input_grid: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate output grid given input grid

        Args:
            input_grid: (batch, H, W) input grid
            task_id: Optional task identifier
            max_steps: Maximum generation steps
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            (batch, H, W) generated output grid
        """
        if max_steps is None:
            max_steps = self.max_grid_size * self.max_grid_size

        batch_size = input_grid.shape[0]
        device = input_grid.device

        # Start with input sequence and separator
        input_seq = input_grid.view(batch_size, -1)
        sep_token = torch.full((batch_size, 1), self.end_token, device=device, dtype=torch.long)
        generated = torch.cat([input_seq, sep_token], dim=1)

        with torch.no_grad():
            for step in range(max_steps):
                # Prepare batch dict
                batch_dict = {
                    "inputs": generated,
                    "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=device)
                }

                # Initialize carry for this step
                carry = self._init_outer_carry(batch_size, generated.shape[1], device, batch_dict)

                # Forward pass
                _, outputs_dict = self.model(carry, batch_dict)
                hidden = outputs_dict.get("hidden", outputs_dict.get("logits"))
                logits = self.output_proj(hidden)

                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if we've generated enough tokens or hit end token
                if next_token.item() == self.end_token or step >= max_steps - 1:
                    break

        # Extract output grid from generated sequence
        input_len = self.max_grid_size * self.max_grid_size
        output_start = input_len + 1  # Skip input and separator
        output_seq = generated[:, output_start:output_start + input_len]

        # Pad if necessary
        if output_seq.shape[1] < input_len:
            padding = torch.full(
                (batch_size, input_len - output_seq.shape[1]),
                self.pad_token,
                device=device,
                dtype=torch.long
            )
            output_seq = torch.cat([output_seq, padding], dim=1)

        # Reshape to grid
        output_grid = output_seq.view(batch_size, self.max_grid_size, self.max_grid_size)

        # Clip to valid color range
        output_grid = torch.clamp(output_grid, 0, self.num_colors - 1)

        return output_grid
