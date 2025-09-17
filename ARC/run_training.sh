#!/bin/bash

# ARC HRM Training Script for 4 L4 GPUs
# This script will automatically detect available GPUs and run distributed training

echo "Starting ARC HRM Training..."
echo "Detecting available GPUs..."

# Check GPU count
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Found $GPU_COUNT GPUs"

if [ $GPU_COUNT -ge 4 ]; then
    echo "Running distributed training on 4 GPUs"
    python train_arc.py \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --num_epochs 100 \
        --gradient_accumulation_steps 2 \
        --world_size 4
else
    echo "Running on available GPUs: $GPU_COUNT"
    python train_arc.py \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --num_epochs 100 \
        --gradient_accumulation_steps 4 \
        --world_size $GPU_COUNT
fi

echo "Training completed!"
echo "Submission file should be generated as submission.json"