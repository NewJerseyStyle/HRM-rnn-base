import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple


class ARCDataset(Dataset):
    """Dataset for ARC (Abstraction and Reasoning Corpus) tasks"""

    def __init__(
        self,
        data_path: str,
        max_grid_size: int = 30,
        mode: str = 'train',
        num_colors: int = 10
    ):
        """
        Args:
            data_path: Path to JSON file with ARC tasks
            max_grid_size: Maximum grid dimension (30x30 for ARC)
            mode: 'train', 'val', or 'test'
            num_colors: Number of colors in ARC (0-9, plus padding)
        """
        self.max_grid_size = max_grid_size
        self.mode = mode
        self.num_colors = num_colors
        self.examples = []

        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)

            # Process each task
            for task_id, task_data in data.items():
                if mode in ['train', 'val']:
                    if 'train' in task_data:
                        for idx, example in enumerate(task_data['train']):
                            self.examples.append({
                                'task_id': task_id,
                                'example_id': f"{task_id}_{idx}",
                                'input': self._process_grid(example['input']),
                                'output': self._process_grid(example['output']),
                                'input_shape': torch.tensor(np.array(example['input']).shape),
                                'output_shape': torch.tensor(np.array(example['output']).shape),
                                'type': 'train'
                            })

                elif mode == 'test':
                    if 'test' in task_data:
                        for idx, example in enumerate(task_data['test']):
                            test_example = {
                                'task_id': task_id,
                                'example_id': f"{task_id}_test_{idx}",
                                'input': self._process_grid(example['input']),
                                'input_shape': torch.tensor(np.array(example['input']).shape),
                                'type': 'test'
                            }

                            if 'output' in example:
                                test_example['output'] = self._process_grid(example['output'])
                                test_example['output_shape'] = torch.tensor(np.array(example['output']).shape)
                            else:
                                # For test without labels
                                test_example['output'] = torch.zeros(
                                    self.max_grid_size, self.max_grid_size, dtype=torch.long
                                )
                                test_example['output_shape'] = torch.tensor([0, 0])

                            self.examples.append(test_example)
        else:
            print(f"WARNING: Data path not found: {data_path}")

    def _process_grid(self, grid: List[List[int]]) -> torch.Tensor:
        """Convert grid to padded tensor"""
        grid_array = np.array(grid, dtype=np.int64)
        h, w = grid_array.shape

        # Pad to max_grid_size
        padded = np.full((self.max_grid_size, self.max_grid_size),
                         self.num_colors, dtype=np.int64)  # Use num_colors as padding token
        padded[:h, :w] = grid_array

        return torch.from_numpy(padded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def arc_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for ARC dataset"""
    collated = {
        'input': torch.stack([item['input'] for item in batch]),
        'output': torch.stack([item['output'] for item in batch]),
        'input_shape': torch.stack([item['input_shape'] for item in batch]),
        'output_shape': torch.stack([item['output_shape'] for item in batch]),
    }

    # Include metadata if needed
    collated['task_ids'] = [item['task_id'] for item in batch]
    collated['example_ids'] = [item['example_id'] for item in batch]

    return collated


def remove_padding(grid: List[List[int]], pad_value: int = 10) -> List[List[int]]:
    """Remove padding from grid to get actual dimensions"""
    if not grid or not grid[0]:
        return [[0]]

    # Convert to numpy for easier manipulation
    arr = np.array(grid)

    # Find non-padding rows and columns
    non_pad_rows = np.any(arr != pad_value, axis=1)
    non_pad_cols = np.any(arr != pad_value, axis=0)

    if not np.any(non_pad_rows) or not np.any(non_pad_cols):
        return [[0]]

    # Get bounding box
    row_indices = np.where(non_pad_rows)[0]
    col_indices = np.where(non_pad_cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return [[0]]

    min_row, max_row = row_indices[0], row_indices[-1] + 1
    min_col, max_col = col_indices[0], col_indices[-1] + 1

    # Extract non-padded region
    result = arr[min_row:max_row, min_col:max_col].tolist()

    return result if result else [[0]]