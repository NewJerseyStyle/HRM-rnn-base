#!/bin/bash

echo "=== TPU v3-8 Training Setup for HRM v3 ==="
echo "Environment configured for TPU:"
echo "- Flash Attention: Disabled (TPU fallback)"
echo "- Precision: bfloat16"
echo "- Architecture: hrm_v3_tpu with SRUpp"
echo "==========================================="

# Set TPU environment variables
export DISABLE_FLASH_ATTN=1  # TPU doesn't support flash attention
export XLA_USE_BF16=1        # Enable bfloat16 for TPU

# Run v3 with TPU-optimized config
# Using larger batch size for TPU efficiency
python pretrain.py \
    arch=hrm_v3_tpu \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 \
    eval_interval=2000 \
    global_batch_size=1024 \
    lr=1e-4 \
    "$@"