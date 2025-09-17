#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import os

# Simple ARC Dataset for testing
class SimpleARCDataset(Dataset):
    def __init__(self, num_samples=100):
        self.data = []
        for i in range(num_samples):
            # Generate random 5x5 grids
            input_grid = torch.randint(0, 10, (5, 5))
            output_grid = torch.randint(0, 10, (5, 5))  # Random target
            self.data.append({
                'input': input_grid.float(),
                'output': output_grid.long()
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(25, 128),  # 5x5 = 25 inputs
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 250)   # 5x5x10 = 250 outputs (5x5 grid, 10 classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.flatten(x)  # [batch, 25]
        x = self.layers(x)   # [batch, 250]
        x = x.view(batch_size, 5, 5, 10)  # [batch, 5, 5, 10]
        return x

def main():
    print("=== Simple GPU Training Test ===")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create simple dataset
    dataset = SimpleARCDataset(100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Created dataset with {len(dataset)} samples")

    # Create simple model
    model = SimpleModel().to(device)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Verify model is on GPU
    gpu_params = sum(1 for p in model.parameters() if p.device.type == 'cuda')
    total_params = sum(1 for p in model.parameters())
    print(f"Parameters on GPU: {gpu_params}/{total_params}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("\nStarting training...")
    model.train()

    for epoch in range(3):  # Just 3 epochs for testing
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_grid = batch['input'].to(device)
            target_grid = batch['output'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_grid)  # [batch, 5, 5, 10]

            # Reshape for loss calculation
            loss = criterion(outputs.view(-1, 10), target_grid.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx == 0:
                print(f"  Epoch {epoch+1}, batch {batch_idx}: loss = {loss.item():.4f}")
                print(f"    Input device: {input_grid.device}")
                print(f"    Output device: {outputs.device}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}")

        # Save a checkpoint
        checkpoint_path = f'simple_checkpoint_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / (1024*1024)
            print(f"  Saved checkpoint: {checkpoint_path} ({size_mb:.1f} MB)")

    print("\n=== Training Complete ===")

    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 5, 5).to(device)
        test_output = model(test_input)
        print(f"Inference successful: {test_output.shape}")

if __name__ == "__main__":
    main()