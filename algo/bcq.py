import argparse
import gym
import numpy as np
import os
import torch

import d4rl
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import continuous_bcq.BCQ
import continuous_bcq.DDPG as DDPG
import continuous_bcq.utils as utils

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.policy_models import MLP, DiagGaussian, BasicActor
from common.logger import Logger
from common.util import set_device_and_logger

# python algo/bcq.py --task halfcheetah-random-v0 --seeds 1 2 3 --model-dir saved_models/BCQ  --device_id 5
# we can do --max_timesteps 1 --eval_episodes 1 for testing


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="halfcheetah-random-v0")               # OpenAI gym environment name
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument("--model-dir", type=str, default="bc_models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="results")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--algo-name", type=str, default="bcq")                      
    parser.add_argument("--data-name", type=str, default="train")
        
    parser.add_argument("--eval_freq", default=2e4, type=float)     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ

    return parser.parse_args()

# Trains BCQ offline
def train_BCQ(env, state_dim, action_dim, max_action, device, output_dir, seed, args):
    # For saving files
    setting = f"{args.task}_{seed}"

    # Initialize policy
    policy = continuous_bcq.BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    # dataset = d4rl.qlearning_dataset(env)
    dataset = env.data

    N = dataset['rewards'].shape[0]
    print('Loading buffer! total size:', N)
    for i in range(N-1):
        obs = dataset['observations'][i]
        if args.task == 'Abiomed-v0':
            new_obs = dataset['next_observations'][i]

            # new_obs = new_obs[:90]
            new_obs = new_obs.reshape(-1)
            # obs = obs[:90]
            obs = obs.reshape(-1)
        else:
            new_obs = dataset['observations'][i+1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        replay_buffer.add(obs, action, new_obs, reward, done_bool)
    print('Loaded buffer')

    evaluations = []
    episode_num = 0
    done = True 
    training_iters = 0
    
    while training_iters < args.max_timesteps: 
            print('Train step:', training_iters)
            pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
            # pickle save

            ev = eval_policy(policy, args.task, seed, args.eval_episodes)
            evaluations.append(ev)
            
            print(f'Iteration {training_iters} Actor loss: {pol_vals:.2f}, Eval Rewards: {np.mean(ev["eval/episode_reward"]):.2f}')

            training_iters += args.eval_freq
            print(f"Training iterations: {training_iters}")

    # return evaluations
    return pol_vals


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=1000):
    eval_env = gym.make(env_name, **kwargs) 
    eval_env.seed(seed)# + 100)
    eval_ep_info_buffer = []

    state, done = eval_env.reset(), False
    for _ in range(eval_episodes):
            # state, done = eval_env.reset(), False
            ep_reward = 0.0
            episode_length = 0

            # while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            ep_reward += reward
            episode_length += 1

            eval_ep_info_buffer.append(
                {"episode_reward": ep_reward, "episode_length": episode_length}
            )
            state = eval_env.get_obs() 
    return {
        "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
        "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
    }
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = get_args()

    print("---------------------------------------")	
    print(f"Setting: Training BCQ, Env: {args.task}, Seed: {args.seeds}")
    print("---------------------------------------")

    os.makedirs(args.model_dir, exist_ok=True)

    t0 = datetime.now().strftime("%m%d_%H%M%S")

    log_file = f'seed__{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
    model_path = os.path.join(args.model_dir, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)
    model_logger = Logger(writer=writer,log_path=model_path)
    
    Devid = args.device_id if args.device == 'cuda' else -1
    set_device_and_logger(Devid, logger, model_logger)

    # Create environment and get dataset
    scaler_info = {'rwd_stds': None, 'rwd_means':None, 'scaler': None}
    if args.task == "Abiomed-v0":
        gym.envs.registration.register(
            id='Abiomed-v0',
            entry_point='abiomed_env:AbiomedEnv',
            max_episode_steps=1000,
        )
        kwargs = {"args": args, 'scaler_info': scaler_info}
        env = gym.make(args.task, **kwargs)
    else:
        env = gym.make(args.task)

    t0 = datetime.now().strftime("%m%d_%H%M%S")

    results = []
    for seed in args.seeds:
        # env = gym.make(args.task)

        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        state_dim = env.observation_space.shape[0]
        print("STATE DIM: ", state_dim)
        action_dim = env.action_space.shape[0] 
        max_action = float(env.action_space.high[0])

        device = torch.device(args.device_id if torch.cuda.is_available() else "cpu")

        policy = train_BCQ(env, state_dim, action_dim, max_action, device, args.model_dir, seed, args)
        np.save(os.path.join(args.model_dir, f"{args.task}_{t0}_{seed}"), policy)

        # change the input of the env to test - data name and scalar info needs to change
        # call eval policy again - 
        env.scaler_info = {'rwd_stds': env.rwd_stds, 'rwd_means':env.rwd_means, 'scaler': env.scaler}
        args.data_name = 'test'
        kwargs = {"args": args, "logger": logger, 'scaler_info': env.scaler_info}

        # eval_results = evals[-1]
        eval_results = eval_policy(policy, env, seed, 1000)
        # Evaluate
        mean_return = np.mean(eval_results["eval/episode_reward"])
        std_return = np.std(eval_results["eval/episode_reward"])
        mean_length = np.mean(eval_results["eval/episode_length"])
        std_length = np.std(eval_results["eval/episode_length"])
        results.append({
            'seed': seed,
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_length': mean_length,
            'std_length': std_length
        })
        
        print(f"Seed {seed} - Mean Return: {mean_return:.2f} ± {std_return:.2f}")

    results_df = pd.DataFrame(results)
    t0 = datetime.now().strftime("%m%d_%H%M%S")

    os.makedirs(os.path.join(args.logdir, args.task, "bcq"), exist_ok=True)
    results_path = os.path.join(args.logdir, args.task, "bcq", f"bcq_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
