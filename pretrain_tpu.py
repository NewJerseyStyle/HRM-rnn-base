"""
TPU-compatible training script for HRM models.
Modified for Google Cloud TPU v3-8.
"""

import os
import time
import dataclasses
from dataclasses import dataclass
from typing import Sequence, Any, Optional
import importlib

import torch
import torch.nn as nn
import torch.distributed as dist
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2

# TPU-specific imports
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    HAS_TPU = True
except ImportError:
    HAS_TPU = False
    print("Warning: torch_xla not found. TPU support disabled.")

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.losses import CrossEntropyWithoutBatch

# Global variables for distributed training
RANK = 0
WORLD_SIZE = 1
DEVICE = None


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    """Create dataloader with TPU support."""
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        test_set=split == "test",
        batch_size=config.batch_size,
        epochs=config.epochs,
        eval_interval=config.eval_interval,
        limit=config.limit,
        **kwargs
    ), shard_idx=rank, num_shards=world_size)
    
    if HAS_TPU:
        # Use ParallelLoader for TPU
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4
        )
        return loader, dataset.metadata
    else:
        return dataset, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int = 1):
    """Create model with TPU device placement."""
    global DEVICE
    
    # Model config
    model_cfg = {**config.arch, 
                 "batch_size": config.batch_size,
                 "seq_len": train_metadata.train_length_max,
                 "vocab_size": train_metadata.vocab_size,
                 "num_puzzle_identifiers": train_metadata.puzzle_count}
    
    # Load model classes
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    
    # Create model on appropriate device
    if HAS_TPU:
        device = xm.xla_device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    DEVICE = device
    
    model: nn.Module = model_cls(model_cfg)
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)
    model = model.to(device)
    
    # Skip torch.compile for TPU (XLA has its own optimization)
    if not HAS_TPU and "DISABLE_COMPILE" not in os.environ:
        model = torch.compile(model, dynamic=False)
    
    # Optimizers
    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers() if hasattr(model.model, 'puzzle_emb') and model.model.puzzle_emb else [],
            lr=1e-8,
            weight_decay=config.puzzle_emb_weight_decay,
            world_size=world_size
        ),
        AdamAtan2(
            model.parameters(),
            lr=1e-8,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]
    optimizer_lrs = [
        config.puzzle_emb_lr,
        config.lr
    ]
    
    return model, optimizers, optimizer_lrs


def train_batch(config: PretrainConfig, train_state: TrainState, batch, global_batch_size: int, rank: int, world_size: int):
    """Train on a single batch with TPU support."""
    
    # Move batch to device
    if HAS_TPU:
        device = xm.xla_device()
        batch = {k: v.to(device) for k, v in batch.items()}
    else:
        batch = {k: v.cuda() for k, v in batch.items()}
    
    # Initialize carry if needed
    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)
    
    # Forward pass
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
    
    # Normalize loss
    loss = loss / global_batch_size
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    for optimizer, lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()
    
    # TPU-specific: mark step
    if HAS_TPU:
        xm.mark_step()
    
    train_state.step += 1
    
    return metrics


def tpu_training_loop(index, flags):
    """TPU training loop for each TPU core."""
    global RANK, WORLD_SIZE, DEVICE
    
    RANK = index
    WORLD_SIZE = 8  # TPU v3-8 has 8 cores
    DEVICE = xm.xla_device()
    
    # Load config
    config = flags['config']
    
    # Create dataloaders
    train_loader, train_metadata = create_dataloader(
        config, "train", RANK, WORLD_SIZE,
        test_set_mode=False, 
        epochs_per_iter=config.train_epochs_per_iter,
        global_batch_size=config.global_batch_size
    )
    
    # Create model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, WORLD_SIZE)
    
    # Training state
    train_state = TrainState(
        step=0,
        total_steps=config.epochs * len(train_loader),
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )
    
    # Training loop
    if HAS_TPU:
        # Use ParallelLoader for efficient data loading
        para_loader = pl.ParallelLoader(train_loader, [DEVICE])
        
        for epoch in range(config.epochs):
            for batch in para_loader.per_device_loader(DEVICE):
                metrics = train_batch(config, train_state, batch, 
                                    config.global_batch_size, RANK, WORLD_SIZE)
                
                if train_state.step % config.eval_interval == 0:
                    # Log metrics (master only)
                    if xm.is_master_ordinal():
                        print(f"Step {train_state.step}: Loss = {metrics.get('loss', 0):.4f}")
    else:
        # Fallback to regular training
        for epoch in range(config.epochs):
            for batch in train_loader:
                metrics = train_batch(config, train_state, batch,
                                    config.global_batch_size, RANK, WORLD_SIZE)
                
                if train_state.step % config.eval_interval == 0:
                    print(f"Step {train_state.step}: Loss = {metrics.get('loss', 0):.4f}")


class PretrainConfig(pydantic.BaseModel):
    """Training configuration."""
    seed: int = 0
    data_path: str = "data/arc-1-aug-1000"
    epochs: int = 50000
    eval_interval: int = 1000
    train_epochs_per_iter: int = 10
    batch_size: int = 128  # TPU-friendly batch size
    global_batch_size: int = 1024  # 128 * 8 cores
    lr: float = 1e-4
    puzzle_emb_lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.1
    puzzle_emb_weight_decay: float = 0.1
    limit: Optional[int] = None
    checkpoint_path: Optional[str] = "checkpoints"
    arch: dict = {}


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def main(hydra_config: DictConfig) -> None:
    """Main entry point for TPU training."""
    
    # Convert Hydra config to PretrainConfig
    config = PretrainConfig(**hydra_config)
    
    if HAS_TPU:
        # Launch TPU training
        flags = {'config': config}
        xmp.spawn(tpu_training_loop, args=(flags,), nprocs=8, start_method='fork')
    else:
        # Fallback to single device training
        print("TPU not available, falling back to GPU/CPU training")
        tpu_training_loop(0, {'config': config})


if __name__ == "__main__":
    # Ensure flash attention is disabled for TPU
    os.environ['DISABLE_FLASH_ATTN'] = '1'
    main()