# MAGIC Estimator for Offline Policy Evaluation

Implementation of the **MAGIC (Model and Guided Importance Sampling Combining)** algorithm for offline policy evaluation in the abiomed environment.

## Overview

MAGIC is a state-of-the-art offline policy evaluation (OPE) algorithm that combines:
- **Model-based estimation**: Using learned dynamics models
- **Importance sampling**: Reweighting trajectories from behavior policy
- **Adaptive weighting**: Optimally combining different estimators to minimize MSE

The algorithm learns optimal weights for different j-step estimators (where j controls the mix of IS and model rollouts) using quadratic programming.

## Features

- **Trajectory Reconstruction**: Automatically reconstructs trajectories from raw transition data
- **Behavior Policy Learning**: Learns behavior policy via behavior cloning
- **Multiple Estimators**: Implements IS, WIS, PDIS, DR, and hybrid j-step returns
- **Bootstrap Confidence Intervals**: Provides uncertainty estimates
- **Optimal Weighting**: Combines estimators using quadratic programming

## Installation

### Prerequisites

The MAGIC implementation requires `cvxpy` for quadratic programming. Install it with:

```bash
# Activate your conda environment
conda activate mopo

# Install cvxpy
pip install cvxpy

# Optional: Install specific solvers for better performance
pip install cvxopt  # For CVXOPT solver
pip install scs     # For SCS solver (default)
```

### Verify Installation

```python
import cvxpy as cp
print(f"cvxpy version: {cp.__version__}")
```

## Usage

### Basic Example

```python
from algo.magic import MAGIC_Practical
from models.transition_model import TransitionModel
from common.buffer import ReplayBuffer

# Load your policy, buffer, and dynamics model
eval_policy = load_policy("path/to/policy.pth")
buffer = load_offline_data()
world_model = load_or_train_dynamics_model()

# Run MAGIC evaluation
results = MAGIC_Practical(
    buffer=buffer,
    eval_policy=eval_policy,
    world_model=world_model,
    action_space=env.action_space,
    J=[-1, 0, 1, 3, 5, 10],  # Return lengths to try
    kappa=200,               # Bootstrap samples
    delta=0.1,               # Confidence level
    gamma=0.99,              # Discount factor
    device="cuda"
)

print(f"MAGIC Estimate: {results['magic_estimate']:.4f}")
print(f"95% CI: [{results['confidence_interval'][0]:.4f}, "
      f"{results['confidence_interval'][1]:.4f}]")
```

### Command-Line Evaluation

Use the provided evaluation script:

```bash
# Evaluate a policy using MAGIC
python helpers/evaluate_magic.py \
  --task abiomed \
  --policy-path saved_models/abiomed/mopo/policy.pth \
  --J -1 0 1 3 5 10 \
  --kappa 200 \
  --eval-episodes 100 \
  --device cuda \
  --devid 0
```

### Parameters

#### MAGIC Algorithm Parameters

- **J** (list of int): Return lengths to combine
  - `-1`: Pure importance sampling (IS)
  - `0`: Pure model-based
  - `1+`: j-step hybrid (j steps of IS, then model rollout)
  - Example: `[-1, 0, 1, 3, 5, 10]`

- **kappa** (int): Number of bootstrap samples for confidence intervals
  - Default: `200`
  - Higher values give tighter CIs but slower computation

- **delta** (float): Confidence level
  - Default: `0.1` (90% confidence)
  - Common values: `0.05` (95% CI), `0.1` (90% CI)

- **gamma** (float): Discount factor
  - Default: `0.99`
  - Should match the MDP discount factor

## Algorithm Details

### Step 0: Data Preparation

1. **Trajectory Reconstruction**: Groups raw transitions into episodes
   - Detects episode boundaries using terminal flags
   - Handles state jumps (environment resets)
   - Configurable distance threshold

2. **Behavior Policy Learning**: Learns behavior policy π̂_b via BC
   - Uses Gaussian policy for continuous actions
   - Trained with negative log-likelihood loss
   - Early stopping to prevent overfitting

3. **Dynamics Model**: Uses existing learned model
   - Ensemble of probabilistic models
   - Predicts next state and reward
   - Includes uncertainty estimates

### Core MAGIC Algorithm

1. **Compute j-step returns** for all j ∈ J
   - Each estimator uses different mix of IS and model rollouts

2. **Estimate covariance matrix** Ω
   - Sample covariance of different estimators
   - Used to measure correlation between estimates

3. **Bootstrap confidence interval**
   - Resample trajectories κ times
   - Compute percentile-based CI
   - Provides bounds [l, u] on true value

4. **Estimate bias vector** b̂
   - For each estimator, compute bias relative to CI
   - Penalizes estimates outside confidence bounds

5. **Solve quadratic program**
   - Minimize: x^T M x where M = Ω + bb^T
   - Subject to: x ≥ 0, Σx = 1
   - Finds optimal weights for combining estimators

6. **Return weighted combination**
   - Final estimate: Σⱼ xⱼ* g^(j)

## Components

### TrajectoryReconstructor

Reconstructs trajectories from transition data:

```python
from algo.magic import TrajectoryReconstructor

reconstructor = TrajectoryReconstructor(
    distance_threshold=100.0,  # State distance for detecting resets
    min_episode_length=5       # Min length before checking resets
)
trajectories = reconstructor.reconstruct_trajectories(buffer)
```

### BehaviorPolicyLearner

Learns behavior policy via behavior cloning:

```python
from algo.magic import BehaviorPolicyLearner

learner = BehaviorPolicyLearner(
    state_dim=17,
    action_dim=6,
    action_space=env.action_space,
    hidden_dims=[256, 256],
    lr=3e-4,
    device="cuda"
)
learner.learn(states, actions, epochs=50)
```

### OffPolicyEstimator

Computes various OPE estimators:

```python
from algo.magic import OffPolicyEstimator

estimator = OffPolicyEstimator(gamma=0.99)

# Vanilla importance sampling
is_estimate = estimator.importance_sampling(
    trajectories, eval_policy, behavior_policy
)

# Weighted importance sampling
wis_estimate = estimator.weighted_importance_sampling(
    trajectories, eval_policy, behavior_policy
)

# Per-decision IS
pdis_estimate = estimator.per_decision_importance_sampling(
    trajectories, eval_policy, behavior_policy
)

# j-step return
j_estimate, per_traj = estimator.compute_j_step_return(
    trajectories, eval_policy, behavior_policy, world_model, j=5
)
```

### MAGICEstimator

Core MAGIC algorithm:

```python
from algo.magic import MAGICEstimator

magic = MAGICEstimator(gamma=0.99)
results = magic.estimate(
    trajectories=trajectories,
    eval_policy=eval_policy,
    behavior_policy=behavior_policy,
    world_model=world_model,
    J=[-1, 0, 1, 3, 5],
    kappa=200,
    delta=0.1
)
```

## Output Format

The `MAGIC_Practical` function returns a dictionary with:

```python
{
    'magic_estimate': float,              # Final MAGIC estimate
    'j_estimates': dict,                  # Individual j-step estimates
    'weights': dict,                      # Optimal weights for each j
    'confidence_interval': tuple,         # (lower, upper) bounds
    'covariance_matrix': np.ndarray,      # Covariance matrix
    'bias_vector': np.ndarray            # Estimated bias vector
}
```

## Example Output

```
================================================================================
MAGIC Practical Algorithm
================================================================================

Step 0a: Reconstructing trajectories from buffer...
Reconstructed 1523 trajectories from 76234 transitions
Average trajectory length: 50.05

Step 0b: Learning behavior policy...
Training behavior policy on 76234 samples...
Epoch 10/50, Loss: 0.2341
Epoch 20/50, Loss: 0.1876
Early stopping at epoch 27

Step 0c: Using provided world model...
Model has 7 ensemble models

Running MAGIC algorithm...

Step 1: Computing j-step returns...
  j=-1: 245.3421
  j=0: 198.7654
  j=1: 232.1234
  j=3: 224.5678
  j=5: 220.3456

Step 2: Estimating covariance matrix...
Covariance matrix computed (condition number: 1.23e+02)

Step 3: Computing bootstrap confidence interval...
Bootstrap CI: [210.1234, 240.5678]

Step 4: Estimating bias vector...
Bias vector: [4.7787 0.     0.     0.     0.    ]

Step 5: Solving quadratic program...
Optimal weights: [0.15 0.20 0.35 0.20 0.10]

Step 6: Computing final estimate...

MAGIC estimate: 225.4567

================================================================================
Final MAGIC Estimate: 225.4567
95% CI: [210.1234, 240.5678]
================================================================================
```

## Troubleshooting

### QP Solver Issues

If the quadratic program solver fails:

1. Install a different solver:
   ```bash
   pip install cvxopt  # Try CVXOPT solver
   ```

2. Check covariance matrix condition number
   - High condition numbers (>1e6) indicate ill-conditioning
   - May need regularization or different J values

3. The code automatically falls back to uniform weights if QP fails

### High Importance Weights

If you see warnings about high importance weights:

- The behavior policy may be poorly learned
- Try increasing behavior policy training epochs
- Consider using PDIS instead of IS (less variance)

### Memory Issues

For large datasets:

- Reduce `kappa` (number of bootstrap samples)
- Use smaller set of J values
- Process trajectories in batches

## References

1. Thomas, P., & Brunskill, E. (2016). "Data-efficient off-policy policy evaluation for reinforcement learning." In ICML.

2. Jiang, N., & Li, L. (2016). "Doubly robust off-policy value evaluation for reinforcement learning." In ICML.

3. Thomas, P., Theocharous, G., & Ghavamzadeh, M. (2015). "High-confidence off-policy evaluation." In AAAI.

## Files

- `algo/magic.py`: Main MAGIC implementation
- `helpers/evaluate_magic.py`: Evaluation script
- `algo/MAGIC_README.md`: This documentation

## License

This implementation follows the same license as the parent MOPO project.
