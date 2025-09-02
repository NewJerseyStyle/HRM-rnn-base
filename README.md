# Hierarchical Reasoning Model with Simple Recurrent Unit (SRU)

|Task|Model|Score|
|----|-----|-----|
|sudoku extreme 1000|HRM (Transformers)|0.55|
|sudoku extreme 1000|HRM (4 layers SRU dim=512)|0.43|
|sudoku extreme 1000|HRM (4 layers SRU++ dim=128)|0.40 <!-- 0.39765 -->|

Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI.
Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency.
HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes.
Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities.
These results underscore HRM's potential as a transformative advancement toward universal computation and general-purpose reasoning systems.

## 🔬 SRU Experimental Version

This fork implements the **Simple Recurrent Unit (SRU)** as an alternative to the transformer layers in the original HRM architecture. Key modifications include:

- **SRU Integration**: Replaced transformer-based H_level and L_level modules with SRU layers for potentially improved sequential processing
- **Enhanced Compatibility**: Added fallback implementations for hardware compatibility issues
- **Mixed Precision Support**: Proper dtype handling for bfloat16/float32 training

## Quick Start Guide 🚀

### Prerequisites ⚙️

⚠️ **Important Hardware Compatibility Notice**: 

Due to hardware limitations during development (RTX 2060, pre-Ampere architecture), this codebase includes several fallback implementations that may impact performance on modern hardware:

### Fallback Implementations (Performance Impact)

| Component | Original (Fast) | Fallback (Compatible) | Performance Impact | Affected GPUs |
|-----------|----------------|----------------------|-------------------|---------------|
| **Flash Attention** | `flash-attn` CUDA kernels | Standard PyTorch attention | ~2-3x slower | Pre-Ampere GPUs (RTX 20xx, GTX series) |
| **Adam-Atan2** | CUDA-optimized `adam-atan2` | Pure PyTorch `adam-atan2-pytorch` | ~1.5x slower optimizer steps | GPUs with incompatible CUDA kernels |
| **Mixed Precision** | bfloat16 | float16 | Minimal, better compatibility | GPUs without native bfloat16 |

### For Users with Modern Hardware (RTX 30xx/40xx, A100, H100)

If you have an Ampere or newer GPU, you can enable optimized implementations for significantly faster training:

```bash
# Install optimized Flash Attention (for Ampere+ GPUs)
pip install flash-attn --no-build-isolation

# Run without fallbacks (automatic detection)
python pretrain.py arch=hrm_v3 ...
```

### For Users with Older Hardware (RTX 20xx, GTX series)

The fallbacks are automatically enabled for compatibility:

```bash
# Force fallback mode (if automatic detection fails)
DISABLE_FLASH_ATTN=1 python pretrain.py arch=hrm_v3 ...
```

### Environment Variables

- `DISABLE_FLASH_ATTN=1`: Force use of PyTorch attention fallback (for pre-Ampere GPUs)
- `WANDB_DISABLED=1`: Disable Weights & Biases logging
- `OMP_NUM_THREADS=N`: Set OpenMP threads (recommended: number of GPUs)

**Note**: The fallback implementations ensure broad hardware compatibility but are not optimized for maximum performance. Users with modern hardware are encouraged to use the original optimized versions when possible.

```bash
# Install CUDA 12.6
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override

export CUDA_HOME=/usr/local/cuda-12.6

# Install PyTorch with CUDA 12.6
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Additional packages for building extensions
pip3 install packaging ninja wheel setuptools setuptools-scm
```

Then install FlashAttention. For Hopper GPUs, install FlashAttention 3

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

For Ampere or earlier GPUs, install FlashAttention 2

```bash
pip3 install flash-attn
```

## Install Python Dependencies 🐍

```bash
pip install -r requirements.txt
```

## W&B Integration 📈

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
wandb login
```

## Run Experiments

### Quick Demo: Sudoku Solver 💻🗲

Train a master-level Sudoku AI capable of solving extremely difficult puzzles on a modern laptop GPU. 🧩

```bash
# Download and build Sudoku dataset
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

# Start training (single GPU, smaller batch size)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

Runtime: ~10 hours on a RTX 4070 laptop GPU

## Trained Checkpoints 🚧

 - [ARC-AGI-2](https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2)
 - [Sudoku 9x9 Extreme (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme)
 - [Maze 30x30 Hard (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-maze-30x30-hard)

To use the checkpoints, see Evaluation section below.

## Full-scale Experiments 🔵

Experiments below assume an 8-GPU setup.

### Dataset Preparation

```bash
# Initialize submodules
git submodule update --init --recursive

# ARC-1
python dataset/build_arc_dataset.py  # ARC offical + ConceptARC, 960 examples
# ARC-2
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000  # ARC-2 official, 1120 examples

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py  # Full version
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples

# Maze
python dataset/build_maze_dataset.py  # 1000 examples
```

### Dataset Visualization

Explore the puzzles visually:

* Open `puzzle_visualizer.html` in your browser.
* Upload the generated dataset folder located in `data/...`.

## Launch experiments

### Multi-GPU Training 🔥

The codebase supports distributed training across multiple GPUs using PyTorch's `torchrun`. The training automatically detects the distributed environment and scales accordingly.

**Single GPU:**
```bash
python pretrain.py [args...]
```

**Multiple GPUs (recommended for full-scale experiments):**
```bash
# Use all available GPUs
torchrun --nproc-per-node=$(nvidia-smi -L | wc -l) pretrain.py [args...]

# Use specific number of GPUs (e.g., 4 GPUs)
torchrun --nproc-per-node=4 pretrain.py [args...]

# For 8 GPUs (as in original experiments)
OMP_NUM_THREADS=8 torchrun --nproc-per-node=8 pretrain.py [args...]
```

**Environment Variables:**
- `OMP_NUM_THREADS`: Controls CPU threading (recommended: number of GPUs)
- `CUDA_VISIBLE_DEVICES`: Restricts which GPUs to use (e.g., `CUDA_VISIBLE_DEVICES=0,1,2,3`)
- `WANDB_DISABLED=1 WANDB_MODE=offline`: Disable wandb for testing
- `DISABLE_FLASH_ATTN=1`: Force fallback mode for flash attention
### Train SRU
By changing argument passed to the training script
```bash
python pretrain.py arch=hrm_v2 # experiment2 for SRU
python pretrain.py arch=hrm_v3 # experiment3 for SRU++
```

### Small-sample (1K)

ARC-1:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py 
```

*Runtime:* ~24 hours

ARC-2:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-2-aug-1000
```

*Runtime:* ~24 hours (checkpoint after 8 hours is often sufficient)

Sudoku Extreme (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

*Runtime:* ~10 minutes

Maze 30x30 Hard (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/maze-30x30-hard-1k epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

*Runtime:* ~1 hour

### Full Sudoku-Hard

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-hard-full epochs=100 eval_interval=10 lr_min_ratio=0.1 global_batch_size=2304 lr=3e-4 puzzle_emb_lr=3e-4 weight_decay=0.1 puzzle_emb_weight_decay=0.1 arch.loss.loss_type=softmax_cross_entropy arch.L_cycles=8 arch.halt_max_steps=8 arch.pos_encodings=learned
```

*Runtime:* ~2 hours

## Evaluation

Evaluate your trained models:

* Check `eval/exact_accuracy` in W&B.
* For ARC-AGI, follow these additional steps:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>
```

* Then use the provided `arc_eval.ipynb` notebook to finalize and inspect your results.

## Notes

### SRU Implementation Details

 - **Architecture**: The SRU layers replace the transformer-based reasoning modules in the original HRM, potentially offering better sequential modeling for hierarchical reasoning tasks
 - **Memory Efficiency**: SRU has lower memory requirements compared to self-attention mechanisms, enabling training on hardware with limited GPU memory
 - **Performance**: While SRU may have different convergence characteristics compared to transformers, it maintains the core hierarchical reasoning capabilities

### Hardware Compatibility

 - **Optimizer Fallback**: Due to CUDA kernel compatibility issues, this version uses `adam-atan2-pytorch` (pure PyTorch) instead of the optimized CUDA version. This ensures broader hardware compatibility but may result in slightly slower optimization steps
 - **Flash Attention Fallback**: If flash-attn is not available or incompatible, the code automatically falls back to standard PyTorch attention implementation
 - **Mixed Precision**: Proper dtype casting ensures compatibility across different precision settings (float32, bfloat16)

### Training Notes

 - Small-sample learning typically exhibits accuracy variance of around ±2 points.
 - For Sudoku-Extreme (1,000-example dataset), late-stage overfitting may cause numerical instability during training and Q-learning. It is advisable to use early stopping once the training accuracy approaches 100%.

## Citation 📜

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

```bibtex
@misc{njs2025hrm-rnn-based,
      title={Hierarchical Reasoning Model with Simple Recurrent Unit}, 
      author={NewJerseyStyle},
      year={2025},
      primaryClass={cs.AI},
      url={https://github.com/NewJerseyStyle/HRM-rnn-base}, 
}
```
