#!/usr/bin/env python3
"""Test script to verify SRU integration with HRM model."""

import torch
from models.hrm.hrm_act_v2 import HierarchicalReasoningModel_ACTV2

def test_sru_integration():
    # Create a test configuration
    config = {
        'batch_size': 4,
        'seq_len': 32,
        'puzzle_emb_ndim': 0,
        'num_puzzle_identifiers': 100,
        'vocab_size': 1000,
        'H_layers': 2,
        'L_layers': 2,
        'hidden_size': 256,
        'expansion': 2.0,
        'num_heads': 8,
        'pos_encodings': 'learned',
        'halt_max_steps': 5,
        'halt_epsilon': 0.01
    }
    
    print("Creating HRM model with SRU layers...")
    model = HierarchicalReasoningModel_ACTV2(config)
    
    # Test with different dtypes
    dtypes = [torch.float32, torch.bfloat16]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for dtype in dtypes:
        print(f"\nTesting with dtype: {dtype}")
        
        # Move model to device and dtype
        model = model.to(device).to(dtype)
        
        # Create dummy batch
        batch = {
            'inputs': torch.randint(0, 1000, (4, 32), device=device),
            'puzzle_identifiers': torch.randint(0, 100, (4,), device=device)
        }
        
        # Initialize carry
        carry = model.initial_carry(batch)
        
        # Test forward pass
        try:
            with torch.no_grad():
                new_carry, outputs = model(carry, batch)
            
            print(f"✓ Forward pass successful!")
            print(f"  Output shape: {outputs['logits'].shape}")
            print(f"  Output dtype: {outputs['logits'].dtype}")
            print(f"  Q-halt logits shape: {outputs['q_halt_logits'].shape}")
            
            # Test multiple steps
            for step in range(3):
                batch['inputs'] = torch.randint(0, 1000, (4, 32), device=device)
                new_carry, outputs = model(new_carry, batch)
            print(f"✓ Multiple forward passes successful!")
            
        except Exception as e:
            print(f"✗ Error with dtype {dtype}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_sru_integration()