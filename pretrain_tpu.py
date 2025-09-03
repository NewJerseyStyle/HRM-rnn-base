"""
TPU-compatible training script for HRM models.
For Google Cloud TPU v3-8, simply use the regular pretrain.py with TPU environment setup.
"""

import os

# Set up TPU environment
os.environ['DISABLE_FLASH_ATTN'] = '1'  # TPU doesn't support flash attention
os.environ['XLA_USE_BF16'] = '1'  # Enable bfloat16 for TPU

# Import and run the original training script
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the main training function from pretrain.py
from pretrain import launch

if __name__ == "__main__":
    print("=== TPU Training Setup ===")
    print("Environment configured for TPU v3-8:")
    print("- Flash Attention: Disabled")
    print("- Precision: bfloat16")
    print("- Using arch: hrm_v4_tpu")
    print("===========================")
    
    # Call the original launch function
    launch()
