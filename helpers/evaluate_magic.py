"""
Evaluate policies using MAGIC estimator on abiomed environment
"""

import argparse
import os
import sys
import numpy as np
import torch
import gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.magic import MAGIC_Practical
from algo.sac import SACPolicy
from models.policy_models import MLP, DiagGaussian, ActorProb
from models.transition_model import TransitionModel
from common.buffer import ReplayBuffer
from common import util

# For abiomed environment
try:
    from noisy_mujoco.envs import AbiomedRLEnvFactory
    ABIOMED_AVAILABLE = True
except ImportError:
    ABIOMED_AVAILABLE = False
    print("Warning: noisy_mujoco not available, abiomed environment will not work")


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate policy using MAGIC")

    # Environment
    parser.add_argument("--task", type=str, default="abiomed",
                       help="Task name (abiomed or d4rl environment)")

    # Policy to evaluate
    parser.add_argument("--policy-path", type=str, required=True,
                       help="Path to saved policy to evaluate")
    parser.add_argument("--policy-type", type=str, default="sac",
                       choices=["sac", "bc"],
                       help="Type of policy")

    # Dynamics model
    parser.add_argument("--dynamics-path", type=str, default=None,
                       help="Path to saved dynamics model (if None, will train new one)")

    # MAGIC parameters
    parser.add_argument("--J", type=int, nargs='+', default=[-1, 0, 1, 3, 5, 10],
                       help="Return lengths for MAGIC")
    parser.add_argument("--kappa", type=int, default=200,
                       help="Number of bootstrap samples")
    parser.add_argument("--delta", type=float, default=0.1,
                       help="Confidence level")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")

    # Data
    parser.add_argument("--dataset-path", type=str, default=None,
                       help="Path to offline dataset (if None, load from environment)")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    parser.add_argument("--devid", type=int, default=0,
                       help="GPU device ID")

    # True evaluation (for comparison)
    parser.add_argument("--eval-episodes", type=int, default=100,
                       help="Number of episodes for true evaluation")
    parser.add_argument("--no-true-eval", action="store_true",
                       help="Skip true policy evaluation")

    return parser.parse_args()


def load_policy(policy_path, env, device):
    """Load a saved policy"""
    state_dim = env.observation_space.shape[0]
    action_dim = np.prod(env.action_space.shape)

    # Create policy network
    actor_backbone = MLP(input_dim=state_dim, hidden_dims=[256, 256])
    dist_net = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    policy_net = ActorProb(actor_backbone, dist_net, device=str(device))

    # Load weights
    policy_net.load_state_dict(torch.load(policy_path, map_location=device))
    policy_net.eval()

    print(f"Loaded policy from {policy_path}")

    return policy_net


def load_dataset(env, task, dataset_path=None):
    """Load offline dataset"""
    if dataset_path is not None:
        # Load from file
        dataset = np.load(dataset_path, allow_pickle=True)
        print(f"Loaded dataset from {dataset_path}")
    else:
        # Load from environment
        if hasattr(env, 'get_dataset'):
            dataset = env.get_dataset()
            print(f"Loaded dataset from environment")
        else:
            raise ValueError("Environment does not have get_dataset method and no dataset path provided")

    return dataset


def create_buffer_from_dataset(dataset, env):
    """Create replay buffer from dataset"""
    if isinstance(dataset, dict):
        # D4RL format
        n_samples = len(dataset['observations'])
    else:
        # Abiomed format
        n_samples = len(dataset[0].data) if isinstance(dataset, list) else len(dataset.data)

    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)

    buffer = ReplayBuffer(
        buffer_size=n_samples,
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32
    )

    # Load dataset into buffer
    buffer.load_dataset(dataset, env=env)

    print(f"Created buffer with {buffer.size} transitions")

    return buffer


def train_dynamics_model(buffer, env, device):
    """Train a dynamics model on the offline data"""
    from trainer import Trainer
    from config import abiomed

    # Get default config
    if hasattr(abiomed, 'default_config'):
        config = abiomed.default_config
    else:
        # Minimal config
        config = {
            'transition_params': {
                'model': {
                    'ensemble_size': 7,
                    'num_elite': 5,
                    'hidden_dims': [200, 200, 200, 200]
                },
                'lr': 0.001
            }
        }

    # Get static functions
    from static_fns import abiomed as static_abiomed
    static_fns = static_abiomed.StaticFns

    # Create transition model
    transition_model = TransitionModel(
        obs_space=env.observation_space,
        action_space=env.action_space,
        static_fns=static_fns,
        **config['transition_params']
    )

    print("Training dynamics model...")

    # Simple training loop
    n_epochs = 50
    batch_size = 256

    for epoch in range(n_epochs):
        # Sample batch
        batch = buffer.sample(min(batch_size, buffer.size))

        # Update model
        train_info = transition_model.update(batch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {train_info['loss/train_model_loss_mse']:.4f}")

    print("Dynamics model training complete")

    return transition_model


def true_policy_evaluation(policy, env, n_episodes=100):
    """Evaluate policy by rolling out in environment"""
    print(f"\nRunning true policy evaluation ({n_episodes} episodes)...")

    policy.eval()
    episode_returns = []
    episode_lengths = []

    for ep in range(n_episodes):
        # Check if using Gym or Gymnasium API
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            # Gymnasium API
            obs, info = reset_result
        else:
            # Gym API
            obs = reset_result

        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Sample action
            with torch.no_grad():
                action = policy.sample_action(obs.reshape(1, -1), deterministic=True).flatten()

            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                # Gymnasium API
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Gym API
                next_obs, reward, done, info = step_result

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            # Safety check for infinite episodes
            if episode_length >= 1000:
                break

        episode_returns.append(episode_reward)
        episode_lengths.append(episode_length)

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{n_episodes}, Return: {episode_reward:.2f}")

    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)

    print(f"\nTrue Policy Value: {mean_return:.4f} ± {std_return:.4f}")

    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths
    }


def main():
    args = get_args()

    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.devid}")
        util.device = device
    else:
        device = torch.device("cpu")
        util.device = device

    print(f"Using device: {device}")

    # Create environment
    print(f"\nCreating environment: {args.task}")

    if args.task == "abiomed":
        if not ABIOMED_AVAILABLE:
            raise RuntimeError("Abiomed environment not available")

        # Create abiomed environment
        env = AbiomedRLEnvFactory.create_env(
            env_id="synthetic1",
            gamma1=0.0,  # No shaping for evaluation
            gamma2=0.0,
            gamma3=0.0
        )
    else:
        # D4RL environment
        import d4rl
        env = gym.make(args.task)

    print(f"Environment created: obs_dim={env.observation_space.shape[0]}, "
          f"action_dim={np.prod(env.action_space.shape)}")

    # Load policy
    print(f"\nLoading policy from {args.policy_path}")
    eval_policy = load_policy(args.policy_path, env, device)

    # Load dataset
    print(f"\nLoading offline dataset...")
    dataset = load_dataset(env, args.task, args.dataset_path)

    # Create buffer
    buffer = create_buffer_from_dataset(dataset, env)

    # Load or train dynamics model
    if args.dynamics_path is not None:
        print(f"\nLoading dynamics model from {args.dynamics_path}")
        # TODO: Implement dynamics model loading
        raise NotImplementedError("Dynamics model loading not yet implemented")
    else:
        print(f"\nTraining dynamics model...")
        world_model = train_dynamics_model(buffer, env, device)

    # Run MAGIC evaluation
    print("\n" + "=" * 80)
    print("Running MAGIC Evaluation")
    print("=" * 80)

    magic_results = MAGIC_Practical(
        buffer=buffer,
        eval_policy=eval_policy,
        world_model=world_model,
        action_space=env.action_space,
        J=args.J,
        kappa=args.kappa,
        delta=args.delta,
        gamma=args.gamma,
        device=str(device)
    )

    # True policy evaluation (if requested)
    if not args.no_true_eval:
        true_results = true_policy_evaluation(eval_policy, env, args.eval_episodes)

        print("\n" + "=" * 80)
        print("Comparison of Estimates")
        print("=" * 80)
        print(f"MAGIC Estimate:     {magic_results['magic_estimate']:.4f}")
        print(f"True Policy Value:  {true_results['mean_return']:.4f} ± {true_results['std_return']:.4f}")
        print(f"Estimation Error:   {abs(magic_results['magic_estimate'] - true_results['mean_return']):.4f}")
        print("=" * 80)

        print("\nIndividual j-step estimates:")
        for j, estimate in magic_results['j_estimates'].items():
            weight = magic_results['weights'][j]
            error = abs(estimate - true_results['mean_return'])
            print(f"  j={j:3d}: {estimate:8.4f} (weight={weight:.4f}, error={error:.4f})")

    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
