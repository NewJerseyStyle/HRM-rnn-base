"""Experiment script for HRM + GRPO on Program Synthesis."""

import torch
import torch.nn as nn
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from typing import Optional
import copy

from models.hrm.hrm_act_v4_simple import HierarchicalReasoningModel_ACTV4Simple
from data_loaders.program_synthesis import create_program_dataloader
from trainers.grpo_trainer import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer


@hydra.main(version_base=None, config_path="../config", config_name="program_synthesis")
def main(cfg: DictConfig):
    """Main experiment function."""
    
    # Initialize wandb
    if cfg.use_wandb:
        wandb.init(
            project=cfg.project_name,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg)
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    vocab_size = len(tokenizer)
    
    # Update model config with vocab size
    model_config = OmegaConf.to_container(cfg.model)
    model_config['vocab_size'] = vocab_size
    
    # Initialize model
    model = HierarchicalReasoningModel_ACTV4Simple(model_config).to(device)
    
    # Initialize reference model for GRPO (frozen copy)
    ref_model = copy.deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Create data loaders
    train_loader = create_program_dataloader(
        data_path=cfg.data.train_path,
        batch_size=cfg.training.batch_size,
        tokenizer=tokenizer,
        use_reasoning=cfg.data.use_reasoning,
        max_length=cfg.data.max_length,
        use_ast_aware=cfg.data.use_ast_aware
    )
    
    eval_loader = create_program_dataloader(
        data_path=cfg.data.eval_path,
        batch_size=cfg.training.batch_size,
        tokenizer=tokenizer,
        use_reasoning=cfg.data.use_reasoning,
        max_length=cfg.data.max_length,
        use_ast_aware=cfg.data.use_ast_aware,
        shuffle=False
    )
    
    # Initialize GRPO trainer
    grpo_config = GRPOConfig(**cfg.grpo)
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        config=grpo_config,
        device=device
    )
    
    # Training loop
    print("Starting GRPO training...")
    global_step = 0
    best_eval_reward = -float('inf')
    
    for epoch in range(cfg.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")
        
        epoch_metrics = []
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            metrics = trainer.train_step(batch)
            epoch_metrics.append(metrics)
            
            # Log metrics
            if cfg.use_wandb and global_step % cfg.training.log_interval == 0:
                wandb.log(metrics, step=global_step)
            
            # Evaluate
            if global_step % cfg.training.eval_interval == 0:
                eval_metrics = trainer.evaluate(eval_loader)
                print(f"Step {global_step}: {eval_metrics}")
                
                if cfg.use_wandb:
                    wandb.log(eval_metrics, step=global_step)
                
                # Save best model
                if eval_metrics['eval_reward'] > best_eval_reward:
                    best_eval_reward = eval_metrics['eval_reward']
                    save_path = Path(cfg.output_dir) / "best_model.pt"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'global_step': global_step,
                        'best_eval_reward': best_eval_reward,
                        'config': cfg
                    }, save_path)
                    print(f"Saved best model with reward: {best_eval_reward:.4f}")
            
            global_step += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                avg_metrics = {
                    k: sum(m[k] for m in epoch_metrics[-10:]) / min(10, len(epoch_metrics))
                    for k in epoch_metrics[0].keys()
                }
                print(f"Batch {batch_idx}/{len(train_loader)}: "
                      f"Loss: {avg_metrics['loss']:.4f}, "
                      f"Reward: {avg_metrics['mean_reward']:.4f}")
    
    print("\nTraining completed!")
    
    # Final evaluation
    final_metrics = trainer.evaluate(eval_loader)
    print(f"Final evaluation: {final_metrics}")
    
    if cfg.use_wandb:
        wandb.log({"final": final_metrics})
        wandb.finish()


if __name__ == "__main__":
    main()