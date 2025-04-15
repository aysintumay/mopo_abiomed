import argparse
import time
import os
import datetime
import random
import wandb
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
import pickle

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from test import test
from train import train, get_args
from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer
from common.util import set_device_and_logger

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mopo")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="offline")
    # parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--policy_path" , type=str, default="")
    parser.add_argument("--model_path" , type=str, default="saved_models")

    parser.add_argument("--task", type=str, default="Abiomed-v0")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3])
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
    parser.add_argument("--reward-penalty-coef", type=float, default=1.0) #1e=6
    parser.add_argument("--rollout-length", type=int, default=5) #1 
    parser.add_argument("--rollout-batch-size", type=int, default=50000) #50000
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=1) #1000
    parser.add_argument("--step-per-epoch", type=int, default=1) #1000
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device_id", type=int, default=7)
    #world transformer arguments
    parser.add_argument('-seq_dim', '--seq_dim', type=int, metavar='<dim>', default=12,
                        help='Specify the sequence dimension.')
    parser.add_argument('-output_dim', '--output_dim', type=int, metavar='<dim>', default=11*12,
                        help='Specify the sequence dimension.')
    parser.add_argument('-bc', '--bc', type=int, metavar='<size>', default=64,
                        help='Specify the batch size.')
    parser.add_argument('-nepochs', '--nepochs', type=int, metavar='<epochs>', default=1,
                        help='Specify the number of epochs to train for.')
    parser.add_argument('-encoder_size', '--encs', type=int, metavar='<size>', default=2,
                help='Set the number of encoder layers.') 
    parser.add_argument('-lr', '--lr', type=float, metavar='<size>', default=0.001,
                        help='Specify the learning rate.')
    parser.add_argument('-encoder_dropout', '--encoder_dropout', type=float, metavar='<size>', default=0.1,
                help='Set the tunable dropout.')
    parser.add_argument('-decoder_dropout', '--decoder_dropout', type=float, metavar='<size>', default=0,
                help='Set the tunable dropout.')
    parser.add_argument('-dim_model', '--dim_model', type=int, metavar='<size>', default=256,
                help='Set the number of encoder layers.')
    parser.add_argument('-path', '--path', type=str, metavar='<cohort>', 
                        default='/data/abiomed_tmp/processed',
                        help='Specify the path to read data.')
    
    parser.add_argument(
        '--root-dir', 
        #default='log/hopper-medium-replay-v0/mopo',
         default='log', help='root dir'
    )
   
    parser.add_argument(
        '--algos', default="mopo", help='algos'
    )
    
    parser.add_argument(
        '--xlabel', default='Timesteps', help='matplotlib figure xlabel'
    )
    parser.add_argument(
        '--ylabel', default='episode_reward', help='matplotlib figure ylabel'
    )

    parser.add_argument(
        '--ylabel2', default='normalized_episode_reward', help='matplotlib figure ylabel'
    )

    args = parser.parse_args()

    return args



def main(args):

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
        t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
        log_file = f'seed_{seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
        log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)

        model_path = os.path.join(args.model_path, args.task, args.algo_name, log_file)
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger = Logger(writer=writer,log_path=log_path)
        model_logger = Logger(writer=writer,log_path=model_path)

        Devid = args.device_id if args.device == 'cuda' else -1
        set_device_and_logger(Devid, logger, model_logger)

        args.model_path = model_path
        args.pretrained = False #to be fast
        args.data_name = 'train'      
        scaler_info = train(run, logger, seed, args)

        args.data_name = 'test'
        eval_info, dset = test(run, logger, model_logger, scaler_info, seed, args)

        


        mean_return = np.mean(eval_info["eval/episode_reward"])
        std_return = np.std(eval_info["eval/episode_reward"])
        mean_length = np.mean(eval_info["eval/episode_length"])
        std_length = np.std(eval_info["eval/episode_length"])
        results.append({
            'seed': seed,
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_length': mean_length,
            'std_length': std_length
        })
        
        print(f"Seed {seed} - Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    # Save results to CSV
    os.makedirs(os.path.join('results', args.task, 'mopo'), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join('results', args.task, 'mopo', f"{args.task}_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":

  
    main(args=get_args())
