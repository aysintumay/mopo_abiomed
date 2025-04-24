import argparse
import datetime
import os
import random
import importlib
import wandb 
import pickle

import gym
import d4rl
import abiomed_env
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mopo import MOPO
from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer,plot_accuracy
from common.util import set_device_and_logger


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--algo-name", type=str, default="mopo")
#     # parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
#     parser.add_argument("--policy_path" , type=str, default="log/halfcheetah_medium_plot/mopo/seed_5_0403_230811-halfcheetah_medium_replay_v0_mopo/policy_halfcheetah-medium-replay-v0.pth")
    
#     parser.add_argument("--task", type=str, default="halfcheetah-medium-v2")
#     parser.add_argument("--seed", type=int, default=1)
#     parser.add_argument("--actor-lr", type=float, default=3e-4)
#     parser.add_argument("--critic-lr", type=float, default=3e-4)
#     parser.add_argument("--gamma", type=float, default=0.99)
#     parser.add_argument("--tau", type=float, default=0.005)
#     parser.add_argument("--alpha", type=float, default=0.2)
#     parser.add_argument('--auto-alpha', default=True)
#     parser.add_argument('--target-entropy', type=int, default=-1) #-actio_dim
#     parser.add_argument('--alpha-lr', type=float, default=3e-4)

#     # dynamics model's arguments
#     parser.add_argument("--dynamics-lr", type=float, default=0.001)
#     parser.add_argument("--n-ensembles", type=int, default=7)
#     parser.add_argument("--n-elites", type=int, default=5)
#     parser.add_argument("--reward-penalty-coef", type=float, default=1.0) #1e=6
#     parser.add_argument("--rollout-length", type=int, default=5) #1 
#     parser.add_argument("--rollout-batch-size", type=int, default=5000) #50000
#     parser.add_argument("--rollout-freq", type=int, default=1000)
#     parser.add_argument("--model-retain-epochs", type=int, default=5)
#     parser.add_argument("--real-ratio", type=float, default=0.05)
#     parser.add_argument("--dynamics-model-dir", type=str, default=None)

#     parser.add_argument("--epoch", type=int, default=1) #1000
#     parser.add_argument("--step-per-epoch", type=int, default=1000)
#     parser.add_argument("--eval_episodes", type=int, default=3)
#     parser.add_argument("--batch-size", type=int, default=256)
#     parser.add_argument("--logdir", type=str, default="log")
#     parser.add_argument("--log-freq", type=int, default=1000)
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

#     #world transformer arguments
#     parser.add_argument('-seq_dim', '--seq_dim', type=int, metavar='<dim>', default=12,
#                         help='Specify the sequence dimension.')
#     parser.add_argument('-output_dim', '--output_dim', type=int, metavar='<dim>', default=11*12,
#                         help='Specify the sequence dimension.')
#     parser.add_argument('-bc', '--bc', type=int, metavar='<size>', default=64,
#                         help='Specify the batch size.')
#     parser.add_argument('-nepochs', '--nepochs', type=int, metavar='<epochs>', default=20,
#                         help='Specify the number of epochs to train for.')
#     parser.add_argument('-encoder_size', '--encs', type=int, metavar='<size>', default=2,
#                 help='Set the number of encoder layers.') 
#     parser.add_argument('-lr', '--lr', type=float, metavar='<size>', default=0.001,
#                         help='Specify the learning rate.')
#     parser.add_argument('-encoder_dropout', '--encoder_dropout', type=float, metavar='<size>', default=0.1,
#                 help='Set the tunable dropout.')
#     parser.add_argument('-decoder_dropout', '--decoder_dropout', type=float, metavar='<size>', default=0,
#                 help='Set the tunable dropout.')
#     parser.add_argument('-dim_model', '--dim_model', type=int, metavar='<size>', default=256,
#                 help='Set the number of encoder layers.')
#     parser.add_argument('-path', '--path', type=str, metavar='<cohort>', 
#                         default='/data/abiomed_tmp/processed',
#                         help='Specify the path to read data.')
#     return parser.parse_args()


def get_eval(policy, env, logger, trainer, args):

    reward_l, acc_l, off_acc = [], [], []
    reward_std_l, acc_std_l, off_acc_std = [], [], []

    trainer.eval_env = env
    trainer.algo.policy = policy

    if args.task == 'Abiomed-v0':
        eval_info,dset = trainer.evaluate()
        #to plot the results
        # with open(os.path.join('intermediate_data',f'dataset_test_1.pkl'), 'wb') as f:
        #     pickle.dump(dset, f)
    else:
        eval_info, _ = trainer._evaluate()
    ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
    ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
    if args.task == 'Abiomed-v0':
        ep_accuracy_mean, ep_accuracy_std = np.mean(eval_info["eval/episode_accuracy"]), np.std(eval_info["eval/episode_accuracy"])
        ep_1_off_accuracy_mean, ep_1_off_accuracy_std = np.mean(eval_info["eval/episode_1_off_accuracy"]), np.std(eval_info["eval/episode_1_off_accuracy"])

    if args.task == 'Abiomed-v0':
        logger.record("eval/episode_accuracy", ep_accuracy_mean, args.eval_episodes, printed=False)
        logger.record("eval/episode_1_off_accuracy", ep_1_off_accuracy_mean, args.eval_episodes, printed=False)
        logger.print(f"episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f},\
                            episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f},\
                            episode_accuracy: {ep_accuracy_mean:.3f} ± {ep_accuracy_std:.3f},\
                            episode_1_off_accuracy: {ep_1_off_accuracy_mean:.3f} ± {ep_1_off_accuracy_std:.3f}")
    else:
        logger.record("eval/episode_reward", ep_reward_mean, args.eval_episodes, printed=False)
        logger.record("eval/episode_length", ep_length_mean, args.eval_episodes, printed=False)
        logger.print(f"episode_reward: {ep_reward_mean:.3f} ± {ep_reward_std:.3f},\
                            episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")
    # if env.name == 'Abiomed-v0':
    #     plot_accuracy(np.array(reward_l), np.array(reward_std_l)/args.eval_episodes, 'Average Return')
    #     plot_accuracy(np.array(acc_l), np.array(acc_std_l)/args.eval_episodes, 'Accuracy')
    #     plot_accuracy(np.array(off_acc), np.array(off_acc_std)/args.eval_episodes, '1-off Accuracy')

    if args.task != 'Abiomed-v0':
        dset_name = env.unwrapped.spec.name+'-v0'
        normalized_score_mean = d4rl.get_normalized_score(dset_name, ep_reward_mean)*100
        normalized_score_std = d4rl.get_normalized_score(dset_name, ep_reward_std)*100
        logger.record("normalized_episode_reward", normalized_score_mean, ep_length_mean, printed=False)
        logger.print(f"normalized_episode_reward: {normalized_score_mean:.3f} ± {normalized_score_std:.3f},\
                            episode_length: {ep_length_mean:.3f} ± {ep_length_std:.3f}")
    else:
        logger.record("episode_reward", ep_reward_mean/ep_length_mean, args.eval_episodes, printed=False)

    # if args.task == 'Abiomed-v0':
    #     logger.record("avg episode_accuracy", np.array(acc_l).mean(), args.eval_episodes, printed=False)
    #     logger.record("avg episode_1_off_accuracy", np.array(off_acc).mean(), args.eval_episodes, printed=False)
    
    return eval_info
    
    

def test(trainer, sac_policy, run, logger, model_logger, norm_info, seed, args):


    # create env and dataset
    if args.task == "Abiomed-v0":
        gym.envs.registration.register(
        id='Abiomed-v0',
        entry_point='abiomed_env:AbiomedEnv',  
        max_episode_steps = 1000,
        )
        kwargs = {"args": args, "logger": logger, 'scaler_info': norm_info}
        env = gym.make(args.task, **kwargs)
    else:
        env = gym.make(args.task)
    # dataset = d4rl.qlearning_dataset(env)

    #CHANGE
    # dataset = {k: v[:5] for k, v in dataset.items()}

    # args.obs_shape = env.observation_space.shape
    # args.action_dim = np.prod(env.action_space.shape)
    

    env.seed(seed)


    # import configs
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"config.{task}"
    config = importlib.import_module(config_path).default_config

    
    
    # policy_state_dict = torch.load(os.path.join(model_logger.log_path, f'policy_{args.task}.pth'))
    # sac_policy.load_state_dict(policy_state_dict)

    eval_info = get_eval(sac_policy, env, logger, trainer, args)

    
    return eval_info

    
if __name__ == "__main__":

    
    test()
