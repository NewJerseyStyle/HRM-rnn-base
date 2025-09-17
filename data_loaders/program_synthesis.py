"""Program Synthesis Dataset Loader with Language-Aware Tokenization."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer
import ast
import re


class ProgramSynthesisDataset(Dataset):
    """Dataset for program synthesis tasks with language-aware tokenization."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 512,
        use_ast_aware: bool = True,
        language: str = "python"
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.use_ast_aware = use_ast_aware
        self.language = language
        
        # Initialize tokenizer (CodeT5, CodeBERT, or custom)
        if tokenizer is None:
            # Use CodeT5 tokenizer by default
            self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        else:
            self.tokenizer = tokenizer
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load program synthesis data."""
        data = []
        
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                raw_data = json.load(f)
        elif self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r') as f:
                raw_data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Process each example
        for item in raw_data:
            processed = self._process_item(item)
            if processed:
                data.append(processed)
        
        return data
    
    def _process_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data item."""
        # Expected format: {"prompt": str, "code": str, "test_cases": list}
        if 'prompt' not in item or 'code' not in item:
            return None
        
        # Tokenize with language awareness
        if self.use_ast_aware and self.language == "python":
            tokens = self._ast_aware_tokenize(item['code'])
        else:
            tokens = self.tokenizer.encode(
                item['code'],
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(
            item['prompt'],
            max_length=self.max_length // 2,
            truncation=True,
            padding='max_length'
        )
        
        return {
            'prompt': item['prompt'],
            'prompt_tokens': torch.tensor(prompt_tokens),
            'code': item['code'],
            'code_tokens': torch.tensor(tokens),
            'test_cases': item.get('test_cases', []),
            'metadata': item.get('metadata', {})
        }
    
    def _ast_aware_tokenize(self, code: str) -> List[int]:
        """AST-aware tokenization for Python code."""
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            # Extract structured tokens
            tokens = []
            for node in ast.walk(tree):
                # Add node type as special token
                node_type = f"<{node.__class__.__name__}>"
                if node_type in self.tokenizer.vocab:
                    tokens.append(self.tokenizer.vocab[node_type])
                
                # Add actual code tokens
                if hasattr(node, 'lineno'):
                    # Get the actual code for this node
                    node_code = ast.get_source_segment(code, node)
                    if node_code:
                        node_tokens = self.tokenizer.encode(
                            node_code,
                            add_special_tokens=False
                        )
                        tokens.extend(node_tokens[:self.max_length // 10])  # Limit per node
            
            # Ensure we don't exceed max length
            tokens = tokens[:self.max_length]
            
            # Pad if necessary
            if len(tokens) < self.max_length:
                tokens.extend([self.tokenizer.pad_token_id] * (self.max_length - len(tokens)))
            
            return tokens
            
        except SyntaxError:
            # Fallback to regular tokenization if AST parsing fails
            return self.tokenizer.encode(
                code,
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


class ProgramReasoningDataset(ProgramSynthesisDataset):
    """Extended dataset for program reasoning with step-by-step solutions."""
    
    def _process_item(self, item: Dict) -> Optional[Dict]:
        """Process item with reasoning steps."""
        base_item = super()._process_item(item)
        if not base_item:
            return None
        
        # Add reasoning steps if available
        if 'reasoning_steps' in item:
            steps = []
            for step in item['reasoning_steps']:
                step_tokens = self.tokenizer.encode(
                    step,
                    max_length=self.max_length // 4,
                    truncation=True,
                    padding='max_length'
                )
                steps.append(torch.tensor(step_tokens))
            base_item['reasoning_steps'] = torch.stack(steps) if steps else None
        
        # Add execution trace if available
        if 'execution_trace' in item:
            trace_tokens = self.tokenizer.encode(
                str(item['execution_trace']),
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            base_item['execution_trace'] = torch.tensor(trace_tokens)
        
        return base_item


def create_program_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    use_reasoning: bool = False,
    **kwargs
) -> DataLoader:
    """Create a DataLoader for program synthesis."""
    
    DatasetClass = ProgramReasoningDataset if use_reasoning else ProgramSynthesisDataset
    
    dataset = DatasetClass(
        data_path=data_path,
        tokenizer=tokenizer,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )


# Example usage for different program synthesis benchmarks
class HumanEvalDataset(ProgramSynthesisDataset):
    """Dataset for HumanEval benchmark."""
    
    def _load_data(self) -> List[Dict]:
        """Load HumanEval data."""
        # Implement HumanEval specific loading
        pass


class MBPPDataset(ProgramSynthesisDataset):
    """Dataset for MBPP (Mostly Basic Python Problems)."""
    
    def _load_data(self) -> List[Dict]:
        """Load MBPP data."""
        # Implement MBPP specific loading
        pass


class APPSDataset(ProgramSynthesisDataset):
    """Dataset for APPS (Automated Programming Progress Standard)."""
    
    def _load_data(self) -> List[Dict]:
        """Load APPS data."""
        # Implement APPS specific loading
        pass