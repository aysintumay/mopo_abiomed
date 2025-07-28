import argparse
import datetime
import os
import random
import time
import importlib
import wandb 
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import dsrl
import sys

# import gym
import gymnasium as gym
# import d4rl
import numpy as np
import torch
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from common.logger import Logger
from common.util import set_device_and_logger
from common import util

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from noisy_mujoco.wrappers import (
                        RandomNormalNoisyActions, 
                        RandomNormalNoisyTransitions,
                        RandomNormalNoisyTransitionsActions
                        )
from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory


"""
This script supports gymnasium environments but not D4RL package. See the difference:
gymnasium: >= 0.26.3
    observation, info = env.reset()
    observation, reward, terminal,truncated, info =  env.step(action)

gym: < 0.26.3
    observation = env.reset()
    observation, reward, terminal, info = env.step(action)

D4RL uses old gym whereas gymnasium uses the new Gymnasium API.
"""
def get_mopo():


    # import configs
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task.lower()}"
    # static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"config.{task.lower()}"
    # config = importlib.import_module(config_path).default_config
    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        # target_entropy = args.target_entropy if args.target_entropy \
        #     else -np.prod(env.action_space.shape)
        target_entropy = -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

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
    
    return sac_policy

def _evaluate(policy, eval_env, episodes):
        

        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        policy.eval()
        obs, _ = eval_env.reset()

        while num_episodes < episodes:
            action = policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, truncated,  _ = eval_env.step(action)
           
            
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal or truncated:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                ) 
                
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                # if args.task[0].isupper():

                obs, _ = eval_env.reset()
                # else:
                #     obs = eval_env.reset()
        eval_info = {
                            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
                            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],

                        }

        ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
        ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
        print(f"Mean Return: {ep_reward_mean:.2f} ± {ep_reward_std:.2f}")

        return {    
                'mean_return': ep_reward_mean,
                'std_return': ep_reward_std,
                'mean_length': ep_length_mean,
                'std_length': ep_length_std
            }


def get_env():
    if args.task.lower()=="abiomed":
        env = AbiomedRLEnvFactory.create_env(
                                        model_name=args.model_name,
                                        model_path=args.model_path_wm,
                                        data_path=args.data_path_wm,
                                        max_steps=args.max_steps,
                                        action_space_type="continuous",
                                        reward_type="smooth",
                                        normalize_rewards=False,
                                        seed=42
                                        )
        args.obs_shape = env.observation_space.shape[0]
        args.action_dim = env.action_space.shape[0]
    else:
        env = gym.make(args.task)
        args.obs_shape = env.observation_space.shape
        args.action_dim = np.prod(env.action_space.shape)

        if args.action and not args.transition:
            print("Environment with noisy actions")
            env = RandomNormalNoisyActions(env=env, noise_rate=args.noise_rate_action, loc = args.loc, scale = args.scale_action)
        elif args.transition and not args.action:
            print("Environment with noisy transitions")
            env = RandomNormalNoisyTransitions(env=env, noise_rate=args.noise_rate_transition, loc = args.loc, scale = args.scale_transition)
        elif args.transition and args.action:
            print("Environment with noisy actions and transitions")
            env = RandomNormalNoisyTransitionsActions(env=env, noise_rate_action=args.noise_rate_action, loc = args.loc, scale_action = args.scale_action,\
                                                            noise_rate_transition=args.noise_rate_transition, scale_transition = args.scale_transition)
        else:
            print("Environment without noise")
            env = env

    return env

def mopo_args(parser):
    g = parser.add_argument_group("MOPO hyperparameters")
    g.add_argument("--actor-lr", type=float, default=3e-4)
    g.add_argument("--critic-lr", type=float, default=3e-4)
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--tau", type=float, default=0.005)
    g.add_argument("--alpha", type=float, default=0.2)
    g.add_argument('--auto-alpha', default=True)
    g.add_argument('--target-entropy', type=int, default=-1) #-action_dim
    g.add_argument('--alpha-lr', type=float, default=3e-4)

    # dynamics model's arguments
    g.add_argument("--dynamics-lr", type=float, default=0.001)
    g.add_argument("--n-ensembles", type=int, default=7)
    g.add_argument("--n-elites", type=int, default=5)
    g.add_argument("--reward-penalty-coef", type=float, default=5e-3) #1e=6
    g.add_argument("--rollout-length", type=int, default=5) #1 
    g.add_argument("--rollout-batch-size", type=int, default=5000) #50000
    g.add_argument("--rollout-freq", type=int, default=1000)
    g.add_argument("--model-retain-epochs", type=int, default=5)
    g.add_argument("--real-ratio", type=float, default=0.05)
    g.add_argument("--dynamics-model-dir", type=str, default=None)
    g.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


    g.add_argument("--epoch", type=int, default=600) #1000
    g.add_argument("--step-per-epoch", type=int, default=1000) 
    #1000
    g.add_argument("--batch-size", type=int, default=256)
    return parser
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument(
        "--algo-name",
        choices=["mbpo","mopo",'uambpo',"bcq","bc"],
        default="mbpo",
        help="Which algorithm’s flags to load"
    )
    args_partial, remaining_argv = base.parse_known_args()
    parser = argparse.ArgumentParser(
        # keep the base flags and auto‐help
        parents=[base],
        description="Test your RL method"
    )

    # parser.add_argument("--algo-name", type=str, default="mbpo")
    # parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="offline")
    parser.add_argument("--task", type=str, default="Abiomed-v0")
    parser.add_argument("--policy_path" , type=str,
                         default="/home/ubuntu/mopo/saved_models/Abiomed-v0/mbpo/seed_2_0424_174352-Abiomed_v0_mbpo/policy_Abiomed-v0.pth")
    parser.add_argument("--model_path" , type=str, default="saved_models")
    parser.add_argument(
                    "--devid", 
                    type=int,
                    default=0,
                    help="Which GPU device index to use"
                )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_episodes", type=int, default=1000)
    # parser.add_argument("--terminal_counter", type=int, default=1) 
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    

    #world transformer arguments
    parser.add_argument('-seq_dim', '--seq_dim', type=int, metavar='<dim>', default=12,
                        help='Specify the sequence dimension.')
    parser.add_argument('-output_dim', '--output_dim', type=int, metavar='<dim>', default=11*12,
                        help='Specify the sequence dimension.')
    parser.add_argument('-bc', '--bc', type=int, metavar='<size>', default=64,
                        help='Specify the batch size.')
    parser.add_argument('-nepochs', '--nepochs', type=int, metavar='<epochs>', default=20,
                        help='Specify the number of epochs to train for.')
    parser.add_argument('-encoder_size', '--encs', type=int, metavar='<size>', default=2,
                help='Set the number of encoder layers.') 
    parser.add_argument('-lr', '--lr', type=float, metavar='<size>', default=0.001,
                        help='Specify the learning rate.')
    parser.add_argument('-encoder_dropout', '--encoder_dropout', type=float, metavar='<size>', default=0.1,
                help='Set the tunable dropout.')
    parser.add_argument('-decoder_dropout', '--decoder_dropout', type=float, metavar='<size>', default=0.1,
                help='Set the tunable dropout.')
    parser.add_argument('-dim_model', '--dim_model', type=int, metavar='<size>', default=256,
                help='Set the number of encoder layers.')
    parser.add_argument('-path', '--path', type=str, metavar='<cohort>', 
                        default='/data/abiomed_tmp/processed',
                        help='Specify the path to read data.')
    #=================Abiomed Environment Arguments================
    parser.add_argument("--model_name", type=str, default="10min_1hr_window")
    parser.add_argument("--model_path_wm", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=24)
    #===============Noisy Environment Arguments================
    parser.add_argument("--noise_rate_action", type=float, help="Portion of action to be noisy with probability", default=0.01)
    parser.add_argument("--noise_rate_transition", type=float, help="Portion of transitions to be noisy with probability", default=0.01)
    parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
    parser.add_argument("--scale_action", type=float, default=0.001, help="Standard deviation of the action noise distribution")
    parser.add_argument("--scale_transition", type=float, default=0.001, help="Standard deviation of the transition noise distribution")
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")
    
    if args_partial.algo_name == "mopo":
        mopo_args(parser)
    elif args_partial.algo_name == "mbpo":
        mopo_args(parser)
        
    elif args_partial.algo_name == "uambpo":
        mopo_args(parser)
    # elif args_partial.algo_name == "bcq":
    #     bcq_args(parser)
    # else:
    #     bc_args(parser)
    args = parser.parse_args()

    
    results = []
    # for seed in args.seeds:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    # log_file = 'seed_1_0415_200911-walker2d_random_v0_mopo'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)

    model_path = os.path.join(args.model_path, args.task, args.algo_name, log_file)
    # writer = SummaryWriter(log_path)
    # writer.add_text("args", str(args))
    # logger = Logger(writer=writer,log_path=log_path)
    # model_logger = Logger(writer=writer,log_path=model_path)

    # Devid = args.devid if args.device == 'cuda' else -1
    args.device = f'cuda:{args.devid}'

    # args.device = set_device_and_logger(Devid, logger, model_logger)
    
    env = get_env() 
    policy = get_mopo()

    policy_state_dict = torch.load(args.policy_path, map_location=args.device)
    
    policy.load_state_dict(policy_state_dict)
    eval_info = _evaluate(policy, env, args.eval_episodes) 
    mean_return = eval_info["mean_return"]
    std_return = eval_info["std_return"]
    mean_length = eval_info["mean_length"]
    std_length = eval_info["std_length"]
  
    results.append({
        # 'seed': seed,
        'mean_return': mean_return,
        'std_return': std_return,
        'mean_length': mean_length,
        'std_length': std_length,
    })
    
    print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    # Save results to CSV
    os.makedirs(os.path.join('results', args.task, args.algo_name), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join('results', args.task, args.algo_name, f"{args.task}_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")