"""
Evaluate policies using MAGIC estimator on abiomed environment
"""

import argparse
import os
import sys
import numpy as np
import torch
import gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from magic import MAGIC_Practical
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from common.buffer import ReplayBuffer
from common import util
from mbpo_kde.mopo import get_args
from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory
import yaml


policy_paths_lookup = {
    "SVR": "/abiomed/models/policy_models/SVR/abiomed/svr_seed_1_2025-08-27_01-55-26_200001.pth",
    "SVR_reshaped": "/abiomed/models/policy_models/SVR/abiomed/svr_seed_1_2025-09-02_23-19-51_200001.pth",
    "MOPO": "/abiomed/models/policy_models/mbpo/abiomed/seed_1_0902_194505-abiomed_mbpo/policy_abiomed.pth",
    "MBPO_KDE": "/abiomed/models/policy_models/mbpo_kde/abiomed/seed_1_0831_184223-abiomed_mbpo_kde/policy_abiomed.pth",
}


def get_args_magic():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="../config/synthetic3/mbpo_kde_acp.yaml")
    config_args, remaining_argv = config_parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {k.replace("-", "_"): v for k, v in config.items()}
    else:
        config = {}
    parser = argparse.ArgumentParser(description="Evaluate policy using MAGIC")
    # Policy to evaluate
    parser.add_argument("--policy-name", type=str, default="MBPO_KDE", 
                       help="Name of policy to evaluate")

    # MAGIC parameters
    parser.add_argument("--J", type=int, nargs='+', default=[-1, 0, 1, 3, 6],
                       help="Return lengths for MAGIC")
    parser.add_argument("--kappa", type=int, default=200,
                       help="Number of bootstrap samples")
    parser.add_argument("--delta", type=float, default=0.1,
                       help="Confidence level")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")

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

    parser.set_defaults(**config)
        # 5. Final parse (command line still wins over YAML)
    args = parser.parse_args(remaining_argv)
    args.config = config_args.config
    print(args.config)
    return args



def load_policy(env, args):
    policy_path = policy_paths_lookup[args.policy_name]
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    # create policy model
    actor_backbone = MLP(input_dim=np.prod(obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(obs_shape) + action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(obs_shape) + action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # create policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        device=args.device
    )

    policy_state_dict = torch.load(policy_path)
    sac_policy.load_state_dict(policy_state_dict)
    print(f"Loaded policy from {policy_path}")

    return sac_policy


def load_dataset(args, env):
    dataset1 = env.world_model.data_train
    dataset2 = env.world_model.data_val
    dataset3 = env.world_model.data_test
    dataset = [dataset1, dataset2, dataset3]
    buffer_len  = len(dataset1.data) + len(dataset2.data) + len(dataset3.data)

    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size = buffer_len,
        obs_shape=env.observation_space.shape,
        obs_dtype=np.float32,
        action_dim=env.action_space.shape[0],
        action_dtype=np.float32
    )
    #since dataset is not in RL format, it handles the transfer and defines buffer_size
    offline_buffer.load_dataset(dataset, env)
    
    return offline_buffer

def transition_model_wrapper(env):
    """Wrapper for transition model"""

    class TransitionModelWrapper:
        def __init__(self, env):
            self.env = env
            self.device = env.world_model.device

        def predict(self, obs, act, deterministic=False):
            #predict: 
            # obs, act -> next_obs, penalized_rewards, terminals, info
            obs = torch.tensor(obs).to(self.device)
            obs = obs.reshape(1, 6, 12)
            act = torch.tensor(act).to(self.device)
            p_level = self.env._action_to_p_level(act)

            with torch.no_grad():
                next_state = self.env.world_model.step(obs, p_level, deterministic).squeeze(0)
            
            reward = self.env._compute_reward(next_state)
            terminals = torch.tensor(False)
            info = {}
            next_obs = next_state
            penalized_rewards = reward

            return next_obs, penalized_rewards, terminals, info

    transition_model = TransitionModelWrapper(env)
    return transition_model


def main():
    args = get_args()
    magic_args = get_args_magic()
    # Merge arguments: copy attributes from magic_args into args
    for key, value in vars(magic_args).items():
        setattr(args, key, value)

    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.devid}")
        args.device = device
        util.device = device
    else:
        device = torch.device("cpu")
        util.device = device

    print(f"Using device: {device}")

    # Create environment
    print(f"\nCreating environment: {args.model_name}")

    env = AbiomedRLEnvFactory.create_env(
                                        model_name=args.model_name,
                                        model_path=args.model_path_wm,
                                        data_path=args.data_path_wm,
                                        max_steps=args.max_steps,
                                        gamma1=args.gamma1,
                                        gamma2=args.gamma2,
                                        gamma3=args.gamma3,
                                        action_space_type='continuous',
                                        reward_type="smooth",
                                        normalize_rewards=True,
                                        noise_rate=args.noise_rate,
                                        noise_scale=args.noise_scale,
                                        seed=42,
                                        device= f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu"
                                        )
    print(f"Environment created: obs_dim={env.observation_space.shape[0]}, "
          f"action_dim={np.prod(env.action_space.shape)}")

    # Load policy
    print(f"\nLoading policy from {args.policy_name}")
    eval_policy = load_policy(env, args)

    # Load dataset
    print(f"\nLoading offline dataset...")
    buffer = load_dataset(args, env)

    world_model = transition_model_wrapper(env)

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

    print("\n" + "=" * 80)
    print("Comparison of Estimates")
    print("=" * 80)
    print(f"MAGIC Estimate:     {magic_results['magic_estimate']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
