# HRM + GRPO for Program Synthesis: Research Validation

## Why This is a Valid Experiment

### 1. **Hierarchical Structure Matches Program Reasoning**

Programs have natural hierarchical structure that aligns well with HRM:
- **Low-level (L-level/SRU)**: Handles syntax, token sequences, local patterns
- **High-level (H-level/Refinement)**: Handles program logic, control flow, algorithmic reasoning

This mirrors how humans write code:
1. First understanding the problem (high-level reasoning)
2. Then implementing details (low-level syntax)

### 2. **GRPO Advantages for Program Synthesis**

GRPO (Group Relative Policy Optimization) is particularly well-suited for program synthesis because:

- **Exploration**: Generates multiple solutions per problem, exploring different approaches
- **Relative Comparison**: Learns from comparing solutions within a group, not absolute performance
- **Partial Credit**: Can reward partially correct programs and good reasoning steps
- **Reduced Variance**: Group baseline reduces reward variance compared to single-sample RL

### 3. **Language-Aware Tokenization Benefits**

Using AST-aware tokenization provides:
- **Structural Understanding**: Preserves program structure beyond linear tokens
- **Better Generalization**: Model learns program patterns, not just token sequences
- **Error Localization**: Can identify which parts of the program are incorrect

## Research Contributions

### 1. **Novel Architecture Application**
- First application of HRM to program synthesis (to our knowledge)
- Demonstrates versatility of hierarchical reasoning beyond puzzles

### 2. **Improved Training Method**
- GRPO + HRM combines benefits of:
  - Hierarchical reasoning (HRM)
  - Exploration-based learning (GRPO)
  - Structural understanding (AST-aware tokenization)

### 3. **Better Interpretability**
- Can analyze what L-level vs H-level learns
- Halting mechanism shows when model is "confident"
- Reasoning steps provide insight into problem-solving process

## Expected Improvements Over Baselines

### vs. Standard Transformers
- **Better sample efficiency**: Hierarchical structure provides inductive bias
- **Iterative refinement**: Can improve solutions through multiple passes
- **Adaptive computation**: Halting mechanism saves computation on easy problems

### vs. Standard RL (PPO/REINFORCE)
- **Lower variance**: Group baseline more stable than single-sample
- **Better exploration**: Explicitly generates diverse solutions
- **Faster convergence**: Relative rewards easier to learn than absolute

### vs. Supervised Fine-tuning
- **Learns to reason**: Not just memorizing solutions
- **Handles novel problems**: GRPO explores solution space
- **Improves from feedback**: Can learn from test case results

## Experimental Setup

### Datasets to Test On
1. **HumanEval**: 164 hand-written Python problems
2. **MBPP**: 974 crowd-sourced Python problems
3. **APPS**: 10,000 competition-level problems

### Metrics to Track
- **Pass@k**: Percentage of problems solved in k attempts
- **Reasoning Alignment**: How well generated code follows reasoning steps
- **Sample Efficiency**: Problems solved vs. training samples
- **Computation Efficiency**: FLOPs per problem solved

### Ablation Studies
1. **HRM vs Transformer backbone**: Same GRPO, different architecture
2. **GRPO vs supervised**: Same HRM, different training
3. **AST-aware vs regular tokenization**: Impact of structural understanding
4. **With/without reasoning steps**: Value of intermediate supervision

## Implementation Notes

### Training Tips
1. **Start with smaller problems**: Train on basic problems first
2. **Curriculum learning**: Gradually increase problem difficulty
3. **Warm-start from supervised**: Pre-train with supervised learning, then GRPO
4. **Balance exploration/exploitation**: Adjust group_size and temperature

### Hyperparameter Recommendations
```yaml
# For program synthesis
hidden_size: 512-1024  # Larger for complex reasoning
H_layers: 4-6          # More refinement for harder problems
L_layers: 6-8          # Handle longer sequences
group_size: 4-8        # More exploration for harder problems
reasoning_weight: 0.3-0.5  # Higher for problems with reasoning steps
```

## Potential Extensions

1. **Multi-language**: Extend to Java, C++, JavaScript
2. **Program repair**: Fix buggy code instead of generating from scratch
3. **Code translation**: Translate between programming languages
4. **Test generation**: Generate test cases for given code
5. **Explanation generation**: Explain what code does

## Citations and Related Work

- HRM: Inspired by adaptive computation and hierarchical reasoning
- GRPO: Builds on group-based policy optimization
- Program Synthesis: Extends work on neural program synthesis
- AST-aware: Follows structured code representation research

## Running the Experiment

```bash
# Prepare data
python prepare_program_data.py --dataset humaneval

# Train with GRPO
python experiments/program_synthesis_grpo.py \
    data.dataset=humaneval \
    model.hidden_size=512 \
    grpo.group_size=4

# Evaluate
python evaluate_program_synthesis.py \
    --checkpoint outputs/program_synthesis/best_model.pt \
    --dataset humaneval \
    --num_samples 100
```

## Expected Timeline

- **Week 1-2**: Data preparation and baseline implementation
- **Week 3-4**: HRM + GRPO training
- **Week 5-6**: Evaluation and ablation studies
- **Week 7-8**: Analysis and paper writing

This experiment combines cutting-edge techniques in a novel way that could lead to significant improvements in program synthesis!