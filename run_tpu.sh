#!/bin/bash

echo "=== TPU v3-8 Training Setup ==="
echo "Environment configured for TPU:"
echo "- Flash Attention: Disabled"
echo "- Precision: bfloat16"
echo "- Architecture: hrm_v4_tpu"
echo "==============================="

# Set TPU environment variables
export DISABLE_FLASH_ATTN=1 XLA_USE_BF16=1 python pretrain.py arch=hrm_v4_tpu data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 "$@"
