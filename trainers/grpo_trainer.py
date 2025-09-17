"""GRPO (Group Relative Policy Optimization) Trainer for Program Reasoning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import wandb


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # GRPO specific
    group_size: int = 4  # Number of samples per group
    reward_scale: float = 1.0
    kl_penalty: float = 0.01
    baseline_type: str = "group_mean"  # "group_mean", "moving_average", "value_function"
    
    # Reasoning specific
    reward_shaping: bool = True
    partial_credit: bool = True
    reasoning_weight: float = 0.3  # Weight for intermediate reasoning steps
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_interval: int = 100
    save_interval: int = 1000


class RewardFunction:
    """Reward function for program synthesis."""
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        
    def compute_reward(
        self,
        generated_code: str,
        test_cases: List[Dict],
        reasoning_steps: Optional[List[str]] = None,
        execution_trace: Optional[str] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward for generated program."""
        
        rewards = {}
        
        # 1. Correctness reward (test case passing)
        test_reward = self._evaluate_test_cases(generated_code, test_cases)
        rewards['test_cases'] = test_reward
        
        # 2. Syntax validity reward
        syntax_reward = self._check_syntax(generated_code)
        rewards['syntax'] = syntax_reward
        
        # 3. Reasoning alignment reward
        if reasoning_steps and self.config.reward_shaping:
            reasoning_reward = self._evaluate_reasoning(
                generated_code, reasoning_steps, execution_trace
            )
            rewards['reasoning'] = reasoning_reward
        else:
            rewards['reasoning'] = 0.0
        
        # 4. Code quality metrics (optional)
        if self.config.reward_shaping:
            quality_reward = self._evaluate_code_quality(generated_code)
            rewards['quality'] = quality_reward
        else:
            rewards['quality'] = 0.0
        
        # Combine rewards
        total_reward = (
            rewards['test_cases'] * 0.5 +
            rewards['syntax'] * 0.2 +
            rewards['reasoning'] * self.config.reasoning_weight +
            rewards['quality'] * (0.3 - self.config.reasoning_weight)
        )
        
        return total_reward * self.config.reward_scale, rewards
    
    def _evaluate_test_cases(self, code: str, test_cases: List[Dict]) -> float:
        """Evaluate code against test cases."""
        if not test_cases:
            return 0.0
        
        passed = 0
        for test in test_cases:
            try:
                # Create execution environment
                exec_globals = {}
                exec(code, exec_globals)
                
                # Run test
                result = exec_globals[test['function']](test['input'])
                if result == test['expected']:
                    passed += 1
                elif self.config.partial_credit:
                    # Partial credit for partially correct answers
                    passed += self._compute_partial_credit(result, test['expected'])
            except:
                continue
        
        return passed / len(test_cases)
    
    def _check_syntax(self, code: str) -> float:
        """Check syntax validity."""
        try:
            compile(code, '<string>', 'exec')
            return 1.0
        except SyntaxError:
            return 0.0
    
    def _evaluate_reasoning(
        self,
        code: str,
        reasoning_steps: List[str],
        execution_trace: Optional[str]
    ) -> float:
        """Evaluate reasoning alignment."""
        # Implement reasoning evaluation logic
        # This could involve checking if the code follows the reasoning steps
        return 0.5  # Placeholder
    
    def _evaluate_code_quality(self, code: str) -> float:
        """Evaluate code quality metrics."""
        # Simple heuristics for code quality
        lines = code.split('\n')
        
        # Penalize very long or very short solutions
        length_score = 1.0 - abs(len(lines) - 10) / 50
        length_score = max(0, min(1, length_score))
        
        # Check for good practices
        has_comments = any('//' in line or '#' in line for line in lines)
        has_functions = 'def ' in code
        
        quality = length_score * 0.5
        if has_comments:
            quality += 0.25
        if has_functions:
            quality += 0.25
        
        return quality
    
    def _compute_partial_credit(self, result: Any, expected: Any) -> float:
        """Compute partial credit for partially correct answers."""
        # Implement partial credit logic
        return 0.0


class GRPOTrainer:
    """GRPO trainer for HRM with program reasoning."""
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,  # Reference model for KL penalty
        config: GRPOConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.device = device
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Initialize components
        self.reward_fn = RewardFunction(config)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        # Tracking
        self.global_step = 0
        self.best_reward = -float('inf')
        self.baseline = 0.0  # Moving average baseline
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single GRPO training step."""
        
        # 1. Generate multiple samples per prompt (group)
        groups = self._generate_groups(batch)
        
        # 2. Compute rewards for each sample
        rewards, reward_info = self._compute_rewards(groups)
        
        # 3. Compute advantages using group baseline
        advantages = self._compute_advantages(rewards)
        
        # 4. Compute policy gradient loss with KL penalty
        loss, loss_info = self._compute_loss(groups, advantages)
        
        # 5. Optimization step
        loss.backward()
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        # Combine metrics
        metrics = {
            'loss': loss.item(),
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantages.mean().item(),
            **loss_info,
            **reward_info
        }
        
        return metrics
    
    def _generate_groups(self, batch: Dict[str, torch.Tensor]) -> List[Dict]:
        """Generate multiple samples per prompt."""
        groups = []
        
        with torch.no_grad():
            for i in range(batch['prompt_tokens'].size(0)):
                prompt = batch['prompt_tokens'][i].unsqueeze(0)
                
                # Generate group_size samples
                samples = []
                for _ in range(self.config.group_size):
                    # Use HRM to generate
                    output = self._generate_sample(prompt)
                    samples.append(output)
                
                groups.append({
                    'prompt': prompt,
                    'samples': samples,
                    'test_cases': batch.get('test_cases', [[]])[i],
                    'reasoning_steps': batch.get('reasoning_steps', [[]])[i]
                })
        
        return groups
    
    def _generate_sample(self, prompt: torch.Tensor) -> Dict:
        """Generate a single sample using HRM."""
        # Initialize carry
        batch = {'inputs': prompt}
        carry = self.model.initial_carry(batch)
        
        generated_tokens = []
        log_probs = []
        
        # Generate tokens autoregressively
        for _ in range(512):  # Max length
            carry, outputs = self.model(carry, batch)
            
            # Sample from logits
            logits = outputs['logits'][:, -1, :]
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
            
            # Track log probability
            log_prob = torch.log(probs.gather(-1, token))
            log_probs.append(log_prob)
            
            generated_tokens.append(token)
            
            # Check for halting
            if outputs.get('halted', torch.tensor([False]))[0]:
                break
            
            # Update batch for next step
            batch['inputs'] = torch.cat([batch['inputs'], token], dim=1)
        
        return {
            'tokens': torch.cat(generated_tokens, dim=1),
            'log_probs': torch.cat(log_probs, dim=1)
        }
    
    def _compute_rewards(
        self, 
        groups: List[Dict]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rewards for all samples."""
        all_rewards = []
        reward_info = defaultdict(list)
        
        for group in groups:
            group_rewards = []
            
            for sample in group['samples']:
                # Decode tokens to code
                code = self._decode_tokens(sample['tokens'])
                
                # Compute reward
                reward, info = self.reward_fn.compute_reward(
                    code,
                    group['test_cases'],
                    group.get('reasoning_steps')
                )
                
                group_rewards.append(reward)
                for k, v in info.items():
                    reward_info[k].append(v)
            
            all_rewards.extend(group_rewards)
        
        # Average reward info
        reward_info = {k: np.mean(v) for k, v in reward_info.items()}
        
        return torch.tensor(all_rewards, device=self.device), reward_info
    
    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute advantages using group baseline."""
        advantages = []
        
        # Process rewards in groups
        for i in range(0, len(rewards), self.config.group_size):
            group_rewards = rewards[i:i + self.config.group_size]
            
            if self.config.baseline_type == "group_mean":
                baseline = group_rewards.mean()
            elif self.config.baseline_type == "moving_average":
                self.baseline = 0.9 * self.baseline + 0.1 * group_rewards.mean()
                baseline = self.baseline
            else:
                baseline = 0.0
            
            group_advantages = group_rewards - baseline
            advantages.append(group_advantages)
        
        return torch.cat(advantages)
    
    def _compute_loss(
        self,
        groups: List[Dict],
        advantages: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss with KL penalty."""
        
        policy_losses = []
        kl_divs = []
        
        adv_idx = 0
        for group in groups:
            for sample in group['samples']:
                # Policy gradient loss
                advantage = advantages[adv_idx]
                log_probs = sample['log_probs']
                policy_loss = -(log_probs * advantage).mean()
                policy_losses.append(policy_loss)
                
                # KL divergence penalty
                with torch.no_grad():
                    ref_output = self._get_ref_logits(
                        group['prompt'],
                        sample['tokens']
                    )
                    ref_log_probs = F.log_softmax(ref_output, dim=-1)
                
                current_log_probs = F.log_softmax(
                    self._get_current_logits(group['prompt'], sample['tokens']),
                    dim=-1
                )
                
                kl_div = F.kl_div(
                    current_log_probs,
                    ref_log_probs.exp(),
                    reduction='batchmean'
                )
                kl_divs.append(kl_div)
                
                adv_idx += 1
        
        # Combine losses
        policy_loss = torch.stack(policy_losses).mean()
        kl_loss = torch.stack(kl_divs).mean()
        total_loss = policy_loss + self.config.kl_penalty * kl_loss
        
        loss_info = {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item()
        }
        
        return total_loss, loss_info
    
    def _decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode tokens to code string."""
        # Implement token decoding
        return ""  # Placeholder
    
    def _get_ref_logits(self, prompt: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Get logits from reference model."""
        # Implement reference model forward pass
        return torch.randn(tokens.size(0), tokens.size(1), 50000)  # Placeholder
    
    def _get_current_logits(self, prompt: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Get logits from current model."""
        # Implement current model forward pass
        return torch.randn(tokens.size(0), tokens.size(1), 50000)  # Placeholder
    
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()
        
        total_rewards = []
        total_test_accuracy = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Generate single sample per prompt for evaluation
                # Compute rewards
                # Track metrics
                pass
        
        self.model.train()
        
        return {
            'eval_reward': np.mean(total_rewards),
            'eval_test_accuracy': np.mean(total_test_accuracy)
        }