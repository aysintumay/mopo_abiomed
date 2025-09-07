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
from train import train
import yaml
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helpers.evaluate_d4rl import _evaluate as evaluate_d4rl

from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer
from common.util import set_device_and_logger
from common import util
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from noisy_mujoco.wrappers import (RandomNormalNoisyActions,
                                      RandomNormalNoisyTransitions,
                                        RandomNormalNoisyTransitionsActions
                                    )

from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory
import warnings

warnings.filterwarnings("ignore")



def get_args():
    print("Running", __file__)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="config/synthetic3/mbpo_kde_acp.yaml")
    config_args, remaining_argv = config_parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {k.replace("-", "_"): v for k, v in config.items()}
    else:
        config = {}
    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--algo-name", type=str, default="mbpo_kde")
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="offline")
    # parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--policy_path" , type=str, default="")
    parser.add_argument("--model_path" , type=str, default="/abiomed/models/policy_models")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
                    "--devid", 
                    type=int,
                    default=0,
                    help="Which GPU device index to use"
                )

    parser.add_argument("--task", type=str, default="abiomed")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=True)
    parser.add_argument('--target-entropy', type=int, default=-1) #-action_dim
    parser.add_argument('--alpha-lr', type=float, default=3e-4)

    # dynamics model's arguments
    parser.add_argument("--dynamics-lr", type=float, default=0.001)
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=0.5) #1e=6
    parser.add_argument("--rollout-length", type=int, default=5) #1 
    parser.add_argument("--rollout-batch-size", type=int, default=10000) #50000
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=100) #1000
    parser.add_argument("--step-per-epoch", type=int, default=1000) #1000
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--terminal_counter", type=int, default=1) 
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    

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
    parser.add_argument("--classifier_model_name", type=str, default="abiomed/trained_kde_abiomed")
    #============ noisy mujoco arguments ============
    parser.add_argument("--noise_rate_action", type=float, help="Portion of action to be noisy with probability", default=0.01)
    parser.add_argument("--noise_rate_transition", type=float, help="Portion of transitions to be noisy with probability", default=0.01)
    parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
    parser.add_argument("--scale_action", type=float, default=0.001, help="Standard deviation of the action noise distribution")
    parser.add_argument("--scale_transition", type=float, default=0.001, help="Standard deviation of the transition noise distribution")
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")
    #============ abiomed environment arguments ============
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path_wm", type=str, default=None)
    parser.add_argument("--data_path_wm", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--gamma1", type=float , default=0.0)
    parser.add_argument("--gamma2", type=float, default=0.0)
    parser.add_argument("--gamma3", type=float, default=0.0)
    parser.add_argument("--fs", action="store_true", help = "doing feature selection")
    #======================================================
    


    parser.add_argument(
        '--root-dir', 
        #default='log/hopper-medium-replay-v0/mopo',
         default='log', help='root dir'
    )

    parser.set_defaults(**config)

    # 5. Final parse (command line still wins over YAML)
    args = parser.parse_args(remaining_argv)
    args.config = config_args.config
    print(args.config)

    return args



def main(args):
    # print("ROLLOUT BATCH SIZE", args.rollout_batch_size)
    run = wandb.init(
                project=args.task,
                group=args.algo_name,
                config=vars(args),
                )
    results = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.device != "cpu":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # log
        taskname = args.task
        if args.task == "abiomed":
            if args.noise_rate > 0:
                taskname += f"_nr{args.noise_rate}_ns{args.noise_scale}"
            if args.data_path and "5000eps" in args.data_path:
                taskname += "_5000eps"
            if args.data_path and "200000eps" in args.data_path:
                taskname += "_200000eps"
        t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
        log_file = f'seed_{seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
        log_path = os.path.join(args.logdir, taskname, args.algo_name, log_file)

        model_path = os.path.join(args.model_path, args.algo_name, taskname, log_file)
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger = Logger(writer=writer,log_path=log_path)
        model_logger = Logger(writer=writer,log_path=model_path)

        Devid = args.devid if args.device == 'cuda' else -1
        set_device_and_logger(Devid, logger, model_logger)

        args.model_path = model_path
        args.data_name = 'train'
        # create env and dataset

     
        scaler_info = {'rwd_stds': None, 'rwd_means':None, 'scaler': None}
        if args.task == 'abiomed':
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
                                        device= f"cuda:{Devid}" if torch.cuda.is_available() else "cpu"
                                        )
            # dataset = env.world_model.data_train
      
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
         
        policy, trainer = train(env, run, logger, seed, args)
        trainer.algo.save_dynamics_model(f"dynamics_model")

        # results.append(evaluate_d4rl(policy, env, args.eval_episodes))
        eval_res = evaluate_d4rl(policy, env, args.eval_episodes, args=args, plot=True)
        eval_res['seed']= seed
        results.append(eval_res)
        

        
    # Save results to CSV
    os.makedirs(os.path.join('results', taskname, args.algo_name), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join('results', taskname, args.algo_name, f"{args.task}_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    wandb.finish()

if __name__ == "__main__":

  
    main(args=get_args())
