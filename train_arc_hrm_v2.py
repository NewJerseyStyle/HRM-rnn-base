#!/usr/bin/env python3
"""
Multi-GPU training script for HRM v2 on ARC dataset
Supports both single GPU and distributed training with 4 GPUs
"""

import os
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.amp import autocast, GradScaler

import numpy as np
from tqdm import tqdm

from data_loaders.arc_dataset import ARCDataset, arc_collate_fn, remove_padding
from models.arc_hrm_v2 import ARCHRMv2Model


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    # For single node multi-GPU, ensure we're using the right address
    if 'SLURM_NODELIST' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def get_data_paths(args) -> Tuple[str, str, str]:
    """Get data paths based on environment"""
    # Check for Kaggle environment
    kaggle_base = '/kaggle/input/arc-prize-2025'
    if os.path.exists(kaggle_base):
        return (
            os.path.join(kaggle_base, 'arc-agi_training_challenges.json'),
            os.path.join(kaggle_base, 'arc-agi_evaluation_challenges.json'),
            os.path.join(kaggle_base, 'arc-agi_test_challenges.json')
        )

    # Check for ARC directory
    arc_base = './ARC'
    if os.path.exists(arc_base):
        train_path = os.path.join(arc_base, 'arc-agi_training_challenges.json')
        val_path = os.path.join(arc_base, 'arc-agi_evaluation_challenges.json')
        test_path = os.path.join(arc_base, 'arc-agi_test_challenges.json')
        if os.path.exists(train_path):
            return train_path, val_path, test_path

    # Fallback to custom paths from args
    return args.train_path, args.val_path, args.test_path


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args,
    rank: int = 0
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    else:
        pbar = dataloader

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        input_grids = batch['input'].to(device)
        output_grids = batch['output'].to(device)

        # Create task IDs (hash of task_id string)
        task_ids = torch.tensor(
            [hash(tid) % 1000 for tid in batch['task_ids']],
            dtype=torch.long,
            device=device
        )

        # Mixed precision training
        with autocast('cuda'):
            outputs = model(input_grids, output_grids, task_ids)
            loss = outputs['loss']

        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

        # Update metrics
        total_loss += loss.item() * args.gradient_accumulation_steps
        num_batches += 1

        # Update progress bar
        if rank == 0 and batch_idx % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })

        # Periodic checkpoint
        if args.save_steps > 0 and (batch_idx + 1) % args.save_steps == 0 and rank == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f'checkpoint-epoch{epoch}-step{batch_idx+1}.pt'
            )
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, batch_idx, checkpoint_path)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    rank: int = 0
) -> Tuple[float, float]:
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', disable=rank != 0):
            input_grids = batch['input'].to(device)
            output_grids = batch['output'].to(device)
            task_ids = torch.tensor(
                [hash(tid) % 1000 for tid in batch['task_ids']],
                dtype=torch.long,
                device=device
            )

            with autocast('cuda'):
                outputs = model(input_grids, output_grids, task_ids)
                loss = outputs['loss']

            # Calculate accuracy
            logits = outputs['logits']
            input_len = model.module.max_grid_size * model.module.max_grid_size if hasattr(model, 'module') else model.max_grid_size * model.max_grid_size
            output_start = input_len + 1
            output_logits = logits[:, output_start:output_start + input_len]
            output_logits = output_logits.reshape(
                output_grids.shape[0],
                output_grids.shape[1],
                output_grids.shape[2],
                -1
            )
            predictions = torch.argmax(output_logits, dim=-1)

            # Mask for non-padding positions
            pad_token = model.module.pad_token if hasattr(model, 'module') else model.pad_token
            mask = output_grids != pad_token

            correct = (predictions == output_grids) & mask
            total_correct += correct.sum().item()
            total_pixels += mask.sum().item()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_pixels, 1)

    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    epoch: int,
    step: int,
    path: str
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict()
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def train_distributed(rank: int, world_size: int, args):
    """Distributed training function"""
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Get data paths
    train_path, val_path, test_path = get_data_paths(args)

    if rank == 0:
        print(f"Loading data from:")
        print(f"  Train: {train_path}")
        print(f"  Val: {val_path}")

    # Create datasets
    train_dataset = ARCDataset(train_path, mode='train')
    val_dataset = ARCDataset(val_path, mode='val')

    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=arc_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=arc_collate_fn,
        pin_memory=True
    )

    if rank == 0:
        print(f"Dataset sizes:")
        print(f"  Train: {len(train_dataset)} examples")
        print(f"  Val: {len(val_dataset)} examples")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Total batch size: {args.batch_size * world_size}")

    # Create model
    model = ARCHRMv2Model(args.config_path, device=device).to(device)

    # Wrap model in DDP with find_unused_parameters=True since we disabled puzzle embeddings
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif args.scheduler == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
    else:
        scheduler = None

    # Gradient scaler for mixed precision
    scaler = GradScaler('cuda')

    # Load checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        if rank == 0:
            print(f"Resumed from checkpoint: {args.resume_from}")

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(start_epoch, args.num_epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, args, rank
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, device, rank)

        # Gather metrics from all ranks
        if world_size > 1:
            metrics = torch.tensor([train_loss, val_loss, val_acc], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            train_loss, val_loss, val_acc = metrics.tolist()

        # Save checkpoint
        if rank == 0:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_path = os.path.join(args.output_dir, 'best_model.pt')
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, -1, best_model_path)
                print(f"  New best model saved! (loss: {best_val_loss:.4f}, acc: {best_val_acc:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % args.save_epochs == 0:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint-epoch{epoch+1}.pt')
                save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, -1, checkpoint_path)

    # Cleanup distributed training
    if rank == 0:
        print("Training completed successfully!")
    cleanup_distributed()


def train_single_gpu(args):
    """Single GPU training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data paths
    train_path, val_path, test_path = get_data_paths(args)

    print(f"Loading data from:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")

    # Create datasets
    train_dataset = ARCDataset(train_path, mode='train')
    val_dataset = ARCDataset(val_path, mode='val')

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=arc_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=arc_collate_fn,
        pin_memory=True
    )

    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Val: {len(val_dataset)} examples")

    # Create model
    model = ARCHRMv2Model(args.config_path, device=str(device)).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif args.scheduler == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
    else:
        scheduler = None

    # Gradient scaler for mixed precision
    scaler = GradScaler('cuda')

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, args, rank=0
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, device, rank=0)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, -1, best_model_path)
            print(f"  New best model saved! (loss: {best_val_loss:.4f})")


def generate_submission(model_path: str, test_path: str, output_path: str, config_path: str):
    """Generate submission file"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = ARCHRMv2Model(config_path, device=str(device)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Load test data
    test_dataset = ARCDataset(test_path, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Generating predictions for {len(test_dataset)} test examples...")

    predictions = {}

    with torch.no_grad():
        for batch in tqdm(test_loader):
            task_id = batch['task_ids'][0]
            input_grid = batch['input'].to(device)

            # Generate two attempts with different seeds
            torch.manual_seed(42)
            output_1 = model.generate(input_grid, temperature=0.8)

            torch.manual_seed(123)
            output_2 = model.generate(input_grid, temperature=1.0, top_k=5)

            # Convert to list format and remove padding
            output_1 = remove_padding(output_1[0].cpu().numpy().tolist())
            output_2 = remove_padding(output_2[0].cpu().numpy().tolist())

            # Store predictions
            if task_id not in predictions:
                predictions[task_id] = []

            predictions[task_id].append({
                "attempt_1": output_1,
                "attempt_2": output_2
            })

    # Save submission
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"Submission saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train HRM v2 on ARC dataset')

    # Data arguments
    parser.add_argument('--train_path', type=str, default='./ARC/arc-agi_training_challenges.json')
    parser.add_argument('--val_path', type=str, default='./ARC/arc-agi_evaluation_challenges.json')
    parser.add_argument('--test_path', type=str, default='./ARC/arc-agi_test_challenges.json')

    # Model arguments
    parser.add_argument('--config_path', type=str, default='config/arch/hrm_v2.yaml')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # Scheduler
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'linear', 'none'], default='cosine')

    # Hardware arguments
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=4)

    # Checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs/arc_hrm_v2')
    parser.add_argument('--save_steps', type=int, default=0, help='Save checkpoint every N steps (0 to disable)')
    parser.add_argument('--save_epochs', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')

    # Other arguments
    parser.add_argument('--generate_submission', action='store_true', help='Generate submission file')
    parser.add_argument('--submission_output', type=str, default='submission.json')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config_save_path = os.path.join(args.output_dir, 'training_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f)

    if args.generate_submission:
        # Generate submission
        model_path = os.path.join(args.output_dir, 'best_model.pt')
        test_path, _, _ = get_data_paths(args)
        generate_submission(model_path, args.test_path, args.submission_output, args.config_path)
    else:
        # Training
        if torch.cuda.is_available() and args.num_gpus > 1:
            # Multi-GPU training
            world_size = min(args.num_gpus, torch.cuda.device_count())
            print(f"Starting distributed training with {world_size} GPUs")
            mp.spawn(train_distributed, args=(world_size, args), nprocs=world_size, join=True)
        else:
            # Single GPU or CPU training
            train_single_gpu(args)


if __name__ == '__main__':
    main()
