# MAGIC Quick Start Guide

## Installation

```bash
# 1. Activate environment
conda activate mopo

# 2. Install cvxpy (required for quadratic programming)
pip install cvxpy

# 3. Verify installation
python -c "import cvxpy; print('✓ cvxpy installed:', cvxpy.__version__)"
```

## Basic Usage

### Option 1: Command Line (Recommended)

```bash
python helpers/evaluate_magic.py \
  --task abiomed \
  --policy-path saved_models/abiomed/mopo/policy.pth \
  --J -1 0 1 3 5 10 \
  --kappa 200 \
  --eval-episodes 100 \
  --device cuda \
  --devid 0
```

### Option 2: Python API

```python
import torch
from algo.magic import MAGIC_Practical
from common.buffer import ReplayBuffer

# Load your components
buffer = load_offline_data()         # ReplayBuffer
eval_policy = load_policy("...")     # Policy to evaluate
world_model = load_dynamics_model()  # TransitionModel
action_space = env.action_space

# Run MAGIC
results = MAGIC_Practical(
    buffer=buffer,
    eval_policy=eval_policy,
    world_model=world_model,
    action_space=action_space,
    J=[-1, 0, 1, 3, 5, 10],
    kappa=200,
    delta=0.1,
    gamma=0.99,
    device="cuda"
)

# Results
print(f"MAGIC Estimate: {results['magic_estimate']:.4f}")
print(f"95% CI: {results['confidence_interval']}")
print(f"Weights: {results['weights']}")
```

## Examples

### Run Synthetic Example

```bash
cd /home/ubuntu/abiomed/mopo_abiomed
python examples/magic_example.py
```

This demonstrates:
- Trajectory reconstruction
- Behavior policy learning
- Multiple OPE estimators
- j-step return computation

## Key Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `J` | Return lengths to combine | `[-1, 0, 1, 3, 5]` | Include -1 (IS), 0 (model), and 1-10 (hybrid) |
| `kappa` | Bootstrap samples | `200` | 100-500 depending on dataset size |
| `delta` | Confidence level | `0.1` | 0.05 (95% CI) or 0.1 (90% CI) |
| `gamma` | Discount factor | `0.99` | Match your MDP discount |

### J Parameter Guide

- `-1`: Pure importance sampling (high variance, unbiased)
- `0`: Pure model-based (low variance, potentially biased)
- `1-10`: Hybrid (j steps of IS, then model)
- Recommended: `[-1, 0, 1, 3, 5, 10]` for good coverage

## Expected Output

```
================================================================================
MAGIC Practical Algorithm
================================================================================

Step 0a: Reconstructing trajectories from buffer...
Reconstructed 1523 trajectories from 76234 transitions
Average trajectory length: 50.05

Step 0b: Learning behavior policy...
Training behavior policy on 76234 samples...
Early stopping at epoch 27

Step 0c: Using provided world model...
Model has 7 ensemble models

Running MAGIC algorithm...

Step 1: Computing j-step returns...
  j=-1: 245.34
  j=0: 198.77
  j=1: 232.12
  j=3: 224.57
  j=5: 220.35

Step 2: Estimating covariance matrix...
Covariance matrix computed (condition number: 1.23e+02)

Step 3: Computing bootstrap confidence interval...
Bootstrap CI: [210.12, 240.57]

Step 4: Estimating bias vector...
Bias vector: [4.78 0.   0.   0.   0.  ]

Step 5: Solving quadratic program...
Optimal weights: [0.15 0.20 0.35 0.20 0.10]

Step 6: Computing final estimate...

MAGIC estimate: 225.46

================================================================================
Final MAGIC Estimate: 225.46
95% CI: [210.12, 240.57]
================================================================================
```

## Common Use Cases

### 1. Evaluate a Trained MOPO Policy

```bash
python helpers/evaluate_magic.py \
  --task abiomed \
  --policy-path saved_models/abiomed/mopo/seed_1_*/policy_abiomed.pth \
  --dynamics-path saved_models/abiomed/mopo/seed_1_*/dynamics_model \
  --eval-episodes 100
```

### 2. Evaluate on D4RL Environment

```bash
python helpers/evaluate_magic.py \
  --task halfcheetah-random-v0 \
  --policy-path saved_models/halfcheetah/mopo/policy.pth \
  --J -1 0 1 3 5 \
  --kappa 200
```

### 3. Quick Test (No True Evaluation)

```bash
python helpers/evaluate_magic.py \
  --task abiomed \
  --policy-path saved_models/policy.pth \
  --no-true-eval  # Skip environment rollouts
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cvxpy'"

**Solution:**
```bash
pip install cvxpy
```

### Issue: "QP solver failed"

**Solution:** Install additional solver
```bash
pip install cvxopt
```

### Issue: "High importance weights detected"

**Cause:** Poor behavior policy fit

**Solutions:**
1. Increase behavior policy training epochs
2. Use PDIS instead of IS (set J without -1)
3. Check data quality

### Issue: Memory error

**Solutions:**
1. Reduce `kappa` (bootstrap samples)
2. Reduce number of J values
3. Process trajectories in batches

## File Locations

```
algo/magic.py                    # Main implementation
algo/MAGIC_README.md             # Detailed documentation
helpers/evaluate_magic.py        # Evaluation script
examples/magic_example.py        # Example code
MAGIC_IMPLEMENTATION_SUMMARY.md  # Implementation summary
MAGIC_QUICKSTART.md             # This file
```

## Next Steps

1. **Read Documentation**: See `algo/MAGIC_README.md` for details
2. **Run Examples**: Try `examples/magic_example.py`
3. **Evaluate Your Policy**: Use `helpers/evaluate_magic.py`
4. **Customize**: Modify J, kappa, or other parameters

## Getting Help

- Check `algo/MAGIC_README.md` for detailed documentation
- Run examples to understand component usage
- See implementation summary for algorithm details

## Quick Reference

```python
# Import
from algo.magic import MAGIC_Practical

# Run
results = MAGIC_Practical(
    buffer=buffer,
    eval_policy=policy,
    world_model=model,
    action_space=env.action_space,
    J=[-1, 0, 1, 3, 5, 10],
    kappa=200,
    delta=0.1,
    gamma=0.99
)

# Access results
estimate = results['magic_estimate']
ci_lower, ci_upper = results['confidence_interval']
weights = results['weights']
j_estimates = results['j_estimates']
```

That's it! You're ready to use MAGIC for offline policy evaluation.
