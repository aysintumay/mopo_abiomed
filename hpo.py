import argparse
import time
import os
import sys
import datetime
import random
import wandb
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import gymnasium as gym
from tqdm import tqdm
from itertools import product
from torch.utils.tensorboard import SummaryWriter

from helpers.evaluate_d4rl import _evaluate as evaluate_d4rl
from train import train
from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer
from common.util import set_device_and_logger
from common import util
from mopo import get_args, main
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from noisy_mujoco.wrappers import (RandomNormalNoisyActions,
                                      RandomNormalNoisyTransitions,
                                        RandomNormalNoisyTransitionsActions
                                    )

from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory

def create_search_visualization(results_df, save_path):
    """
    Create visualizations of the hyperparameter search results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Custom Score vs Trial
    axes[0, 0].plot(results_df.index, results_df['custom_score'], 'b-', alpha=0.7)
    axes[0, 0].scatter(results_df.index, results_df['custom_score'], alpha=0.5)
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Custom Score')
    axes[0, 0].set_title('Custom Score vs Trial')
    axes[0, 0].grid(True)
    
    # Plot 2: Score vs gamma1
    axes[0, 1].scatter(results_df['gamma1'], results_df['custom_score'], alpha=0.6)
    axes[0, 1].set_xlabel('Gamma1')
    axes[0, 1].set_ylabel('Custom Score')
    axes[0, 1].set_title('Custom Score vs Gamma1')
    axes[0, 1].grid(True)
    
    # Plot 3: Score vs gamma2
    axes[0, 2].scatter(results_df['gamma2'], results_df['custom_score'], alpha=0.6)
    axes[0, 2].set_xlabel('Gamma2')
    axes[0, 2].set_ylabel('Custom Score')
    axes[0, 2].set_title('Custom Score vs Gamma2')
    axes[0, 2].grid(True)
    
    # Plot 4: Score vs gamma3
    axes[1, 0].scatter(results_df['gamma3'], results_df['custom_score'], alpha=0.6)
    axes[1, 0].set_xlabel('Gamma3')
    axes[1, 0].set_ylabel('Custom Score')
    axes[1, 0].set_title('Custom Score vs Gamma3')
    axes[1, 0].grid(True)
    
    # Plot 5: Component breakdown for top result
    top_result = results_df.iloc[0]
    components = ['Reward', 'WS', 'ACP', 'AIR']
    values = [top_result['avg_reward'], top_result['avg_ws'], top_result['avg_acp'], top_result['avg_air']]
    weights = [0.3, 0.3, 0.2, 0.2]
    weighted_values = [v * w for v, w in zip(values, weights)]
    
    x_pos = np.arange(len(components))
    axes[1, 1].bar(x_pos, values, alpha=0.7, label='Raw Values')
    axes[1, 1].bar(x_pos, weighted_values, alpha=0.7, label='Weighted Values')
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_title('Best Model Components')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(components)
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot 6: Top 10 scores
    top_10 = results_df.head(10)
    axes[1, 2].barh(range(len(top_10)), top_10['custom_score'])
    axes[1, 2].set_yticks(range(len(top_10)))
    axes[1, 2].set_yticklabels([f"T{row['trial_id']}" for _, row in top_10.iterrows()])
    axes[1, 2].set_xlabel('Custom Score')
    axes[1, 2].set_title('Top 10 Trials')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'hyperparameter_search_results.png'), dpi=300, bbox_inches='tight')
    wandb.log({"hyperparameter_search_results": wandb.Image(plt)})
    plt.close()


def run_hyperparameter_search(args, gamma_ranges):
    """
    Run hyperparameter search over gamma values
    """
    
    gamma_combinations = list(product(np.array(gamma_ranges['gamma1']), np.array(gamma_ranges['gamma2']), np.array(gamma_ranges['gamma3'])))
    
    (f"Total combinations to evaluate: {len(gamma_combinations)}")
    seed = args.seeds
    results = []
    best_score = float('-inf')
    best_params = None
    best_model_path = None
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for trial_id, (gamma1, gamma2, gamma3) in enumerate(gamma_combinations):
        try:
            
            # Create a copy of args for this trial
            trial_args = argparse.Namespace(**vars(args))
            trial_args.gamma1 = float(gamma1)
            trial_args.gamma2 = float(gamma2)
            trial_args.gamma3 = float(gamma3)
            
            
            run = wandb.init(
                project="MOPO_hpo",
                group=f"MOPO_{t}",
                name=f"trial_{trial_id}_gamma1_{trial_args.gamma1}_gamma2_{trial_args.gamma2}_gamma3_{trial_args.gamma3}",
                config={
                    "gamma1": trial_args.gamma1,
                    "gamma2": trial_args.gamma2,
                    "gamma3": trial_args.gamma3,
                    "trial_id": trial_id,
                    "seed": seed,
                    "env": trial_args.task,
                },
                reinit=True,
            )
            if args.device != "cpu":
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            # log
            t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
            log_file = f'seed_{seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
            log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)

            model_path = os.path.join(args.model_path, args.algo_name+"_hpo", args.task, t,  log_file)
            if not os.path.exists(os.path.join(args.model_path, args.algo_name+"_hpo", args.task, t)):
                os.makedirs(os.path.join(args.model_path, args.algo_name+"_hpo", args.task, t))
            writer = SummaryWriter(log_path)
            writer.add_text("args", str(args))
            logger = Logger(writer=writer,log_path=log_path)
            model_logger = Logger(writer=writer,log_path=model_path)

            Devid = args.devid if args.device == 'cuda' else -1
            set_device_and_logger(Devid, logger, model_logger)

       

     
            if args.task == 'abiomed':
                env = AbiomedRLEnvFactory.create_env(
                                            model_name=args.model_name,
                                            model_path=args.model_path_wm,
                                            data_path=args.data_path_wm,
                                            max_steps=args.max_steps,
                                            gamma1 = trial_args.gamma1,
                                            gamma2 = trial_args.gamma2,
                                            gamma3 = trial_args.gamma3,
                                            action_space_type="continuous",
                                            reward_type="smooth",
                                            normalize_rewards=True,
                                            seed=42,
                                            device= f"cuda:{Devid}" if torch.cuda.is_available() else "cpu"
                                            )
        
            else:
                env = gym.make(args.task)
                
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
            print(f"Starting trial {trial_id} with gamma1={gamma1}, gamma2={gamma2}, gamma3={gamma3}")

            policy, trainer = train(env, run, logger, seed, trial_args)
            env = AbiomedRLEnvFactory.create_env(
                                            model_name=args.model_name,
                                            model_path=args.model_path_wm,
                                            data_path=args.data_path_wm,
                                            max_steps=args.max_steps,
                                            gamma1 = 0.0,
                                            gamma2 = 0.0,
                                            gamma3 = 0.0,
                                            action_space_type="continuous",
                                            reward_type="smooth",
                                            normalize_rewards=True,
                                            seed=42,
                                            device= f"cuda:{Devid}" if torch.cuda.is_available() else "cpu"
                                            )
            metrics = evaluate_d4rl(policy, env, args.eval_episodes, args= trial_args, plot=True)
            custom_metric =  metrics['custom_metric']
            wandb.log({
                'custom_score':custom_metric,
                'avg_reward': metrics['mean_return'],
                'avg_ws': metrics['mean_wean_score'],
                'avg_acp': metrics['mean_acp'],
                'avg_air': metrics['mean_aggregate_air'],
            })

            
            result = {
                'trial_id': trial_id,
                'gamma1': gamma1,
                'gamma2': gamma2,
                'gamma3': gamma3,
                'custom_score':  custom_metric,
               'avg_reward': metrics['mean_return'],
                'avg_ws': metrics['mean_wean_score'],
                'avg_acp': metrics['mean_acp'],
                'avg_air': metrics['mean_aggregate_air'],
                'model_path': model_path
            }
            reward = metrics['mean_return']
            results.append(result)
            
            if reward > best_score:
                best_score = reward
                best_params = (gamma1, gamma2, gamma3)
                best_model_path = model_path
            
            wandb.run.summary["best_custom_score"] = best_score
            wandb.run.summary["best_params"] = {
                "gamma1": best_params[0],
                "gamma2": best_params[1],
                "gamma3": best_params[2],
            }
            
            
        except Exception as e:
            print(f"Trial {trial_id} failed: {str(e)}")
            wandb.alert(title="Trial Failed", text=f"Trial {trial_id} failed:\n{e}")
            continue
    
    # Save results
    result_path = args.model_path
    results_df = pd.DataFrame(results)
    results_path = os.path.join(result_path, "hyperparameter_search_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Sort by custom score and display top results

    results_df_sorted = results_df.sort_values('custom_score', ascending=False)
    
    print("=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 80)
    print(f"Best Parameters: gamma1={best_params[0]:.4f}, gamma2={best_params[1]:.4f}, gamma3={best_params[2]:.4f}")
    print(f"Best Score: {best_score:.4f}")
    print(f"Best Model Path: {best_model_path}")
    print("\nTop 5 Results:")
    print(results_df_sorted.head().to_string(index=False))
    
    # Create visualization
    

    create_search_visualization(results_df_sorted, result_path)
    wandb.run.summary["best_model_path"] = best_model_path
    run.finish()
    return results_df_sorted, best_params, best_model_path 

if __name__ == "__main__":
    print("Running", __file__)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gamma1", type=float, nargs='+', default=[0.3, 0.5])
    parser.add_argument("--gamma2", type=float, nargs='+', default=[0.5, 0.8])
    parser.add_argument("--gamma3", type=float, nargs='+', default=[0.5, 1])
    # parser.add_argument("--gamma1", type=float, nargs='+', default=[0.0])
    # parser.add_argument("--gamma2", type=float, nargs='+', default=[0.0])
    # parser.add_argument("--gamma3", type=float, nargs='+', default=[0.0])

    args = get_args()
    args_gamma = parser.parse_args()
    device = torch.device(f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu")
    print(device)
    print("---------------------------------------")
    print(f"Env: {args.task}, Seed: {args.seeds}")
    print("---------------------------------------")
    #start wandb logger
    run = wandb.init(
                project='MOPO_hpo',
                group="MOPO",
                config=vars(args),
                )
    seed = args.seeds[0]
    args.seeds = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gamma_ranges = {
        'gamma1': args_gamma.gamma1, #ACP: max 8 min 0
        'gamma2': args_gamma.gamma2, #WS: max 2 min -1
        'gamma3': args_gamma.gamma3 #AIR: max 1 min 0
    }

    results_df, best_params, best_model_path = run_hyperparameter_search(args, gamma_ranges)