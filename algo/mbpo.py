import argparse
import os
import random
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import gym
import d4rl
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import *
from test import *



def evaluate(policy, seed, trainer, args):

    trainer.eval_env = gym.make(args.task)
    trainer.eval_env.seed(seed)
    trainer.algo.policy = policy

    eval_info = trainer._evaluate()
    ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
    ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
    
    print(f"Seed {seed} - Mean Return: {ep_reward_mean:.2f} ± {ep_reward_std:.2f}")

    return {    
                'seed': seed,
                'mean_return': ep_reward_mean,
                'std_return': ep_reward_std,
                'mean_length': ep_length_mean,
                'std_length': ep_length_std
            }
        

def main():
    args = get_args()

    results = []
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    experiment_name = "mbpo" if args.reward_penalty_coef == 0 else "mopo"
    print(f"Based on reward penalty, experiment name: {experiment_name}")

    
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # log
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)

    devid = args.device.split(':')[-1]
    Devid = int(devid) if len(devid)==1 else -1
    set_device_and_logger(Devid,logger, logger)

    run = None # no wandb for baselines
    policy, trainer = train(run, logger, args)
    trainer.algo.save_dynamics_model(f"dynamics_model")
    
    for seed in args.seeds:
        results.append(evaluate(policy, seed, trainer, args))

    
    # Save results to CSV
    if len(results) > 0:
        os.makedirs(os.path.join(args.baseline_logdir, args.task, experiment_name), exist_ok=True)
        results_df = pd.DataFrame(results)
        results_path = os.path.join(args.baseline_logdir, args.task, experiment_name, f"{experiment_name}_results_{t0}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()