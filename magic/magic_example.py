"""
Example script demonstrating MAGIC usage for offline policy evaluation

This example shows how to:
1. Load a trained policy
2. Load offline data
3. Train a dynamics model (or load existing one)
4. Run MAGIC evaluation
5. Compare with true evaluation
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.magic import MAGIC_Practical, TrajectoryReconstructor, BehaviorPolicyLearner, OffPolicyEstimator
from models.policy_models import MLP, DiagGaussian, ActorProb
from common.buffer import ReplayBuffer


def simple_example():
    """
    Simple synthetic example to demonstrate MAGIC
    """
    print("=" * 80)
    print("MAGIC Estimator - Simple Example")
    print("=" * 80)

    # For this example, we'll create synthetic data
    # In practice, you would load real offline data

    # Synthetic environment parameters
    state_dim = 4
    action_dim = 2
    n_trajectories = 100
    traj_length = 50
    gamma = 0.99

    print(f"\nCreating synthetic data:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Number of trajectories: {n_trajectories}")
    print(f"  Trajectory length: {traj_length}")

    # Create synthetic trajectories
    from algo.magic import Trajectory

    trajectories = []
    for i in range(n_trajectories):
        states = np.random.randn(traj_length, state_dim)
        actions = np.random.randn(traj_length, action_dim)
        rewards = np.random.randn(traj_length)
        next_states = states + 0.1 * np.random.randn(traj_length, state_dim)
        terminals = np.zeros(traj_length)
        terminals[-1] = 1.0  # Last state is terminal

        traj = Trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminals=terminals
        )
        trajectories.append(traj)

    print(f"✓ Created {len(trajectories)} synthetic trajectories")

    # Create synthetic policies for demonstration
    class SimplePolicyWrapper:
        """Simple wrapper for demonstration"""
        def __init__(self, state_dim, action_dim):
            self.state_dim = state_dim
            self.action_dim = action_dim

        def log_prob(self, states, actions):
            # Return random log probabilities for demonstration
            return np.random.randn(len(states)) - 2.0

        def sample_action(self, state, deterministic=False):
            return np.random.randn(self.action_dim)

    eval_policy = SimplePolicyWrapper(state_dim, action_dim)
    behavior_policy = SimplePolicyWrapper(state_dim, action_dim)

    print("✓ Created synthetic policies")

    # Create simple model wrapper
    class SimpleModelWrapper:
        """Simple wrapper for demonstration"""
        def predict(self, state, action, deterministic=False):
            next_state = state + 0.1 * np.random.randn(state_dim)
            reward = np.random.randn()
            terminal = 0.0
            info = {}
            return next_state, reward, terminal, info

    world_model = SimpleModelWrapper()

    print("✓ Created synthetic dynamics model")

    # Run different estimators
    print("\n" + "=" * 80)
    print("Running Offline Policy Estimators")
    print("=" * 80)

    estimator = OffPolicyEstimator(gamma=gamma)

    # Importance Sampling
    is_estimate = estimator.importance_sampling(
        trajectories, eval_policy, behavior_policy
    )
    print(f"\nImportance Sampling (IS): {is_estimate:.4f}")

    # Weighted Importance Sampling
    wis_estimate = estimator.weighted_importance_sampling(
        trajectories, eval_policy, behavior_policy
    )
    print(f"Weighted IS (WIS):        {wis_estimate:.4f}")

    # Per-Decision IS
    pdis_estimate = estimator.per_decision_importance_sampling(
        trajectories, eval_policy, behavior_policy
    )
    print(f"Per-Decision IS (PDIS):   {pdis_estimate:.4f}")

    # Model-based estimate
    mb_estimate = estimator.model_based_return(
        trajectories, eval_policy, world_model, rollout_length=10
    )
    print(f"Model-Based (MB):         {mb_estimate:.4f}")

    # Compute j-step returns
    print("\n" + "-" * 80)
    print("j-step Return Estimates")
    print("-" * 80)

    J = [-1, 0, 1, 3, 5]
    j_estimates = {}

    for j in J:
        mean_est, _ = estimator.compute_j_step_return(
            trajectories, eval_policy, behavior_policy, world_model, j
        )
        j_estimates[j] = mean_est
        print(f"j={j:3d}: {mean_est:8.4f}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nThis example demonstrated:")
    print("  ✓ Trajectory reconstruction")
    print("  ✓ Multiple OPE estimators (IS, WIS, PDIS, MB)")
    print("  ✓ j-step return computation")
    print("\nNote: This uses synthetic data. For real evaluation, use:")
    print("  - Real offline data (D4RL or abiomed)")
    print("  - Trained evaluation policy")
    print("  - Learned behavior policy (via BC)")
    print("  - Learned dynamics model")
    print("\nSee helpers/evaluate_magic.py for full example with real data")


def demonstrate_trajectory_reconstruction():
    """
    Demonstrate trajectory reconstruction from buffer
    """
    print("\n" + "=" * 80)
    print("Trajectory Reconstruction Example")
    print("=" * 80)

    # Create a simple buffer with transitions
    state_dim = 4
    action_dim = 2
    buffer_size = 500

    # Mock action space
    class MockActionSpace:
        def __init__(self, dim):
            self.shape = (dim,)
            self.low = -np.ones(dim)
            self.high = np.ones(dim)

    buffer = ReplayBuffer(
        buffer_size=buffer_size,
        obs_shape=(state_dim,),
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32
    )

    # Add some transitions (simulating 5 episodes of length 100)
    print(f"\nAdding {buffer_size} transitions to buffer...")

    for ep in range(5):
        for t in range(100):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = state + 0.1 * np.random.randn(state_dim).astype(np.float32)

            # Mark last transition of episode as terminal
            terminal = 1.0 if t == 99 else 0.0

            buffer.add(state, next_state, action, reward, terminal)

    print(f"Buffer contains {buffer.size} transitions")

    # Reconstruct trajectories
    reconstructor = TrajectoryReconstructor(
        distance_threshold=100.0,
        min_episode_length=5
    )

    trajectories = reconstructor.reconstruct_trajectories(buffer)

    print(f"\nReconstructed {len(trajectories)} trajectories")
    print(f"Trajectory lengths: {[len(t) for t in trajectories]}")
    print(f"Average length: {np.mean([len(t) for t in trajectories]):.2f}")


def demonstrate_behavior_policy_learning():
    """
    Demonstrate behavior policy learning
    """
    print("\n" + "=" * 80)
    print("Behavior Policy Learning Example")
    print("=" * 80)

    state_dim = 4
    action_dim = 2
    n_samples = 1000

    # Generate synthetic state-action pairs
    states = np.random.randn(n_samples, state_dim).astype(np.float32)
    actions = np.random.randn(n_samples, action_dim).astype(np.float32)

    # Mock action space
    class MockActionSpace:
        def __init__(self, dim):
            self.shape = (dim,)
            self.low = -np.ones(dim)
            self.high = np.ones(dim)

    action_space = MockActionSpace(action_dim)

    print(f"\nTraining behavior policy on {n_samples} samples...")

    learner = BehaviorPolicyLearner(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=action_space,
        hidden_dims=[64, 64],
        lr=3e-4,
        device="cpu"
    )

    # Train
    stats = learner.learn(
        states=states,
        actions=actions,
        batch_size=64,
        epochs=20,
        early_stop_threshold=0.001
    )

    print(f"\nTraining complete!")
    print(f"  Final loss: {stats['final_loss']:.4f}")
    print(f"  Epochs: {stats['epochs']}")

    # Test log probability computation
    test_states = states[:10]
    test_actions = actions[:10]

    log_probs = learner.log_prob(test_states, test_actions)
    probs = learner.prob(test_states, test_actions)

    print(f"\nSample log probabilities: {log_probs[:5]}")
    print(f"Sample probabilities: {probs[:5]}")


if __name__ == "__main__":
    # Run examples
    simple_example()
    demonstrate_trajectory_reconstruction()
    demonstrate_behavior_policy_learning()

    print("\n" + "=" * 80)
    print("All Examples Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Install cvxpy: pip install cvxpy")
    print("  2. Run full evaluation: python helpers/evaluate_magic.py --help")
    print("  3. Read documentation: algo/MAGIC_README.md")
