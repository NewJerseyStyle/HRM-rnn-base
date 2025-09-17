import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
from typing import List, Dict, Tuple, Optional
import yaml
from hrm_act_v2 import (
    HierarchicalReasoningModel_ACTV2 as HRM_ACT_V2, 
    HierarchicalReasoningModel_ACTV2Config,
    HierarchicalReasoningModel_ACTV2InnerCarry,
    HierarchicalReasoningModel_ACTV2Carry
)
from pydantic import BaseModel
import argparse


class ARCDataset(Dataset):
    """Dataset for ARC competition tasks"""
    
    def __init__(self, data_path: str, max_grid_size: int = 30, mode: str = 'train'):
        """
        Args:
            data_path: Path to JSON file with ARC tasks
            max_grid_size: Maximum grid dimension (30x30 for ARC)
            mode: 'train', 'val', or 'test' - determines which examples to include
        """
        self.max_grid_size = max_grid_size
        self.mode = mode
        self.examples = []
        
        # Load data
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Process each task
            for task_id, task_data in data.items():
                if mode in ['train', 'val'] and 'train' in task_data:
                    # For training and validation, use training examples with outputs
                    for example in task_data['train']:
                        self.examples.append({
                            'task_id': task_id,
                            'input': self._process_grid(example['input']),
                            'output': self._process_grid(example['output']),
                            'type': 'train'
                        })
                
                elif mode == 'test' and 'test' in task_data:
                    # For testing, use test examples
                    for example in task_data['test']:
                        test_example = {
                            'task_id': task_id,
                            'input': self._process_grid(example['input']),
                            'type': 'test'
                        }
                        # Include output if available (for validation)
                        if 'output' in example:
                            test_example['output'] = self._process_grid(example['output'])
                        else:
                            # For test data without outputs, create dummy output
                            test_example['output'] = torch.zeros_like(test_example['input'])
                        
                        self.examples.append(test_example)
    
    def _process_grid(self, grid: List[List[int]]) -> torch.Tensor:
        """Convert grid to tensor and pad to max size"""
        grid_array = np.array(grid, dtype=np.int64)
        h, w = grid_array.shape
        
        # Pad to max_grid_size
        padded = np.zeros((self.max_grid_size, self.max_grid_size), dtype=np.int64)
        padded[:h, :w] = grid_array
        
        return torch.from_numpy(padded)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class ARCModel(nn.Module):
    """Wrapper for HRM model adapted for ARC tasks"""
    
    def __init__(self, config_path: str, max_grid_size: int = 30, num_colors: int = 10):
        super().__init__()
        
        # Load HRM config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Adapt config for ARC
        config_dict['vocab_size'] = num_colors + 2  # 10 colors + padding + special tokens
        config_dict['batch_size'] = 1  # Will be overridden during training
        config_dict['seq_len'] = max_grid_size * max_grid_size * 2  # input + output
        config_dict['num_puzzle_identifiers'] = 1  # For ARC tasks
        config_dict['puzzle_emb_ndim'] = config_dict.get('hidden_size', 512)
        
        # Initialize HRM model (pass config_dict directly)
        self.model = HRM_ACT_V2(config_dict)
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        
        # Output projection
        self.output_proj = nn.Linear(config_dict['hidden_size'], num_colors)
    
    def forward(self, input_grid: torch.Tensor, target_grid: Optional[torch.Tensor] = None):
        """
        Forward pass for ARC task
        Args:
            input_grid: (batch, height, width) tensor of input grids
            target_grid: (batch, height, width) tensor of target grids (for training)
        """
        batch_size = input_grid.shape[0]
        device = input_grid.device
        
        # Flatten grids to sequences
        input_seq = input_grid.view(batch_size, -1)
        
        if target_grid is not None:
            target_seq = target_grid.view(batch_size, -1)
            # Concatenate input and target for teacher forcing
            full_seq = torch.cat([input_seq, target_seq], dim=1)
        else:
            full_seq = input_seq
        
        # Create batch dictionary for HRM model
        batch_dict = {
            "inputs": full_seq,
            "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long, device=device)
        }
        
        # Initialize carry state for HRM
        seq_len = full_seq.shape[1]
        inner_carry = self._init_carry(batch_size, device)
        carry = self._init_outer_carry(batch_size, seq_len, device, batch_dict)
        
        # Forward through HRM model
        carry, outputs_dict = self.model(carry, batch_dict)
        
        # Get logits from outputs
        if "logits" in outputs_dict:
            logits = outputs_dict["logits"]
        else:
            # Fallback - project hidden states
            hidden = outputs_dict.get("hidden", outputs_dict.get("output", list(outputs_dict.values())[0]))
            logits = self.output_proj(hidden)
        
        # Reshape back to grid if needed
        if target_grid is not None:
            # During training, return logits for the target portion
            target_start = self.max_grid_size * self.max_grid_size
            if logits.shape[1] > target_start:
                target_logits = logits[:, target_start:, :]
            else:
                target_logits = logits
            
            # Reshape to grid format
            seq_len = min(target_logits.shape[1], self.max_grid_size * self.max_grid_size)
            target_logits = target_logits[:, :seq_len, :]
            pad_len = self.max_grid_size * self.max_grid_size - seq_len
            if pad_len > 0:
                padding = torch.zeros(batch_size, pad_len, self.num_colors, device=device)
                target_logits = torch.cat([target_logits, padding], dim=1)
            
            target_logits = target_logits.view(batch_size, self.max_grid_size, self.max_grid_size, self.num_colors)
            return target_logits
        else:
            # During inference, return logits for generation
            return logits
    
    def _init_carry(self, batch_size: int, device: torch.device):
        """Initialize inner carry state"""
        hidden_size = self.model.config.hidden_size
        z_H = torch.zeros(self.model.config.H_layers, batch_size, hidden_size, device=device)
        z_L = torch.zeros(self.model.config.L_layers, batch_size, hidden_size, device=device)
        
        return HierarchicalReasoningModel_ACTV2InnerCarry(z_H=z_H, z_L=z_L)
    
    def _init_outer_carry(self, batch_size: int, seq_len: int, device: torch.device, batch_dict: Dict[str, torch.Tensor]):
        """Initialize outer carry state"""
        inner_carry = self._init_carry(batch_size, device)
        steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        return HierarchicalReasoningModel_ACTV2Carry(
            inner_carry=inner_carry,
            steps=steps,
            halted=halted,
            current_data=batch_dict
        )
    
    def generate(self, input_grid: torch.Tensor, max_steps: int = None) -> torch.Tensor:
        """Generate output grid given input"""
        if max_steps is None:
            max_steps = self.max_grid_size * self.max_grid_size
        
        batch_size = input_grid.shape[0]
        device = input_grid.device
        
        # Start with input sequence
        generated = input_grid.view(batch_size, -1)
        
        # Generate tokens autoregressively
        for _ in range(max_steps):
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(generated)
                logits = self.output_proj(outputs)
                
                # Get next token
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        # Extract generated portion and reshape to grid
        output_start = self.max_grid_size * self.max_grid_size
        output_seq = generated[:, output_start:]
        output_grid = output_seq[:, :self.max_grid_size * self.max_grid_size]
        output_grid = output_grid.view(batch_size, self.max_grid_size, self.max_grid_size)
        
        return output_grid


def train_epoch(model, dataloader, optimizer, criterion, device, rank=0, gradient_accumulation_steps=1):
    """Train for one epoch with distributed support"""
    model.train()
    total_loss = 0
    num_batches = 0

    print(f"Starting training epoch with {len(dataloader)} batches...")

    for batch_idx, batch in enumerate(dataloader):
        # Progress tracking
        if batch_idx % 10 == 0:
            print(f"  Processing batch {batch_idx}/{len(dataloader)}")

        input_grid = batch['input'].to(device, non_blocking=True)
        target_grid = batch['output'].to(device, non_blocking=True)

        # Forward pass
        try:
            logits = model(input_grid, target_grid)
        except Exception as e:
            print(f"Error in forward pass at batch {batch_idx}: {e}")
            raise

        # Calculate loss
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            target_grid.view(-1)
        )

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % 50 == 0:
                print(f"    Completed gradient update at batch {batch_idx}, loss: {loss.item() * gradient_accumulation_steps:.4f}")

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        # Early termination for debugging - COMMENTED OUT FOR FULL TRAINING
        # if batch_idx >= 4:
        #     print(f"Early termination after {batch_idx+1} batches for debugging")
        #     break

    # Handle remaining gradients
    if num_batches % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(num_batches, 1)
    print(f"Completed training epoch with avg loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            
            input_grid = batch['input'].to(device)
            target_grid = batch['output'].to(device)
            
            # Forward pass
            logits = model(input_grid, target_grid)
            
            # Calculate loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_grid.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train_distributed(rank, world_size, args):
    """Main training function for distributed training"""
    setup_distributed(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    
    # Data paths (adjust these for Kaggle environment)
    train_path = '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json'
    val_path = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'
    test_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    
    # Check for local testing
    if not os.path.exists(train_path):
        # Use local paths for testing
        train_path = 'data/arc_train.json'
        val_path = 'data/arc_val.json'
        test_path = 'data/arc_test.json'
    
    # Create datasets
    if rank == 0:
        print("Loading datasets...")
    train_dataset = ARCDataset(train_path, mode='train')
    val_dataset = ARCDataset(val_path, mode='val')
    test_dataset = ARCDataset(test_path, mode='test')
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    if rank == 0:
        print("Initializing model...")
    model = ARCModel('hrm_v2.yaml').to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
    
    # Training loop
    if rank == 0:
        print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            rank, args.gradient_accumulation_steps
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Reduce losses across all processes
        if world_size > 1:
            train_loss_tensor = torch.tensor(train_loss, device=device)
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            train_loss = train_loss_tensor.item()
            val_loss = val_loss_tensor.item()
        
        # Update learning rate
        scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model (only on rank 0)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the underlying model (not DDP wrapper)
                torch.save(model.module.state_dict(), 'best_model.pth')
                print(f"  -> New best model saved!")
    
    # Generate predictions on rank 0 only
    if rank == 0:
        # Load best model for final predictions
        print("\nLoading best model for predictions...")
        model.module.load_state_dict(torch.load('best_model.pth'))
        
        # Create test dataloader without distributed sampling
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # Generate predictions for test set
        print("Generating test predictions...")
        generate_submission(model.module, test_loader, device, 'submission.json')
        print("Submission file created!")
    
    cleanup_distributed()


def single_gpu_train(args):
    """Single GPU training fallback"""
    # Force GPU detection and debugging
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        device = torch.device('cuda:0')
    else:
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Set environment variable for SRU debugging
    os.environ['DEBUG_DEVICE'] = '1'
    
    # Data paths (adjust these for Kaggle environment)
    train_path = '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json'
    val_path = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'
    test_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    
    # Check for local testing
    if not os.path.exists(train_path):
        # Use local paths for testing
        train_path = 'data/arc_train.json'
        val_path = 'data/arc_val.json'
        test_path = 'data/arc_test.json'
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = ARCDataset(train_path, mode='train')
    val_dataset = ARCDataset(val_path, mode='val')
    test_dataset = ARCDataset(test_path, mode='test')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    print(f"Using device: {device}")

    # Check GPU availability and memory
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()

    model = ARCModel('hrm_v2.yaml')
    print(f"Moving model to {device}...")
    model = model.to(device)

    # Ensure all submodules and parameters are on the correct device
    def move_to_device(module, dev):
        for param in module.parameters():
            if param.device != dev:
                param.data = param.data.to(dev)
        for buffer in module.buffers():
            if buffer.device != dev:
                buffer.data = buffer.data.to(dev)

    # Apply device move to all modules
    for name, module in model.named_modules():
        try:
            module.to(device)
            move_to_device(module, device)
            print(f"  Moved {name} to {device}")
        except Exception as e:
            print(f"  Warning: Could not move {name} to {device}: {e}")

    print(f"Model initialized on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify model is on correct device
    print("\nVerifying model device placement:")
    on_gpu = 0
    on_cpu = 0
    for name, param in model.named_parameters():
        if param.device.type == 'cuda':
            on_gpu += 1
        else:
            on_cpu += 1
            print(f"  WARNING: {name} is still on CPU!")

    print(f"  Parameters on GPU: {on_gpu}, on CPU: {on_cpu}")

    # Check GPU memory usage after model loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    if on_cpu > 0:
        print("ERROR: Some model parameters are still on CPU! Training will be slow.")
        print("Attempting to force move all parameters to GPU...")
        for name, param in model.named_parameters():
            if param.device.type != 'cuda':
                param.data = param.data.to(device)
                print(f"  Moved {name} to GPU")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
    
    # Training loop
    print("Starting training...")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            0, args.gradient_accumulation_steps
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                model_path = 'best_model.pth'
                torch.save(model.state_dict(), model_path)
                # Verify the save worked
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path) / (1024*1024)  # MB
                    print(f"  -> New best model saved! ({file_size:.1f} MB)")
                else:
                    print(f"  -> ERROR: Model file not found after save!")
            except Exception as e:
                print(f"  -> ERROR saving model: {e}")

        # Save checkpoint every epoch for safety
        try:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"  -> Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"  -> ERROR saving checkpoint: {e}")
    
    # Load best model for final predictions
    print("\nLoading best model for predictions...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Generate predictions for test set
    print("Generating test predictions...")
    generate_submission(model, test_loader, device, 'submission.json')
    print("Submission file created!")


def main():
    parser = argparse.ArgumentParser(description='Train HRM model on ARC tasks')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, 
                       help='Gradient accumulation steps for larger effective batch size')
    parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs')
    
    args = parser.parse_args()
    
    # Check if CUDA is available and we have multiple GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() >= args.world_size:
        print(f"Starting distributed training on {args.world_size} GPUs")
        print(f"Effective batch size: {args.batch_size * args.world_size * args.gradient_accumulation_steps}")
        mp.spawn(train_distributed, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        print(f"CUDA devices available: {torch.cuda.device_count()}")
        print("Falling back to single GPU training...")
        single_gpu_train(args)


def generate_submission(model, test_loader, device, output_path):
    """Generate submission file for Kaggle"""
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch in test_loader:
            task_id = batch['task_id'][0]  # Batch size 1 for test
            input_grid = batch['input'].to(device)
            
            # Generate two attempts with different strategies
            # Attempt 1: Greedy decoding
            output_grid_1 = model.generate(input_grid)
            
            # Attempt 2: With slight temperature
            # (You could implement temperature sampling here)
            output_grid_2 = model.generate(input_grid)
            
            # Convert to list format
            output_1 = output_grid_1[0].cpu().numpy().tolist()
            output_2 = output_grid_2[0].cpu().numpy().tolist()
            
            # Remove padding (find actual size)
            output_1 = remove_padding(output_1)
            output_2 = remove_padding(output_2)
            
            # Add to predictions
            if task_id not in predictions:
                predictions[task_id] = []
            
            predictions[task_id].append({
                "attempt_1": output_1,
                "attempt_2": output_2
            })
    
    # Save submission
    with open(output_path, 'w') as f:
        json.dump(predictions, f)


def remove_padding(grid):
    """Remove padding from grid (find actual size)"""
    grid = np.array(grid)
    
    # Find non-zero rows and columns
    non_zero_rows = np.any(grid != 0, axis=1)
    non_zero_cols = np.any(grid != 0, axis=0)
    
    if not np.any(non_zero_rows) or not np.any(non_zero_cols):
        # Return minimal grid if all zeros
        return [[0]]
    
    # Get bounding box
    row_indices = np.where(non_zero_rows)[0]
    col_indices = np.where(non_zero_cols)[0]
    
    min_row = row_indices[0]
    max_row = row_indices[-1] + 1
    min_col = col_indices[0]
    max_col = col_indices[-1] + 1
    
    # Crop grid
    cropped = grid[min_row:max_row, min_col:max_col]
    
    return cropped.tolist()


if __name__ == "__main__":
    main()