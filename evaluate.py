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

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from models.transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mopo import MOPO
from common.buffer import ReplayBuffer
from common.logger import Logger
from common.util import set_device_and_logger
from common import util

import warnings
warnings.filterwarnings("ignore")

def get_mopo():


    # import configs
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task}"
    static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"config.{task}"
    config = importlib.import_module(config_path).default_config
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

    actor = ActorProb(actor_backbone, dist, util.device)
    critic1 = Critic(critic1_backbone, util.device)
    critic2 = Critic(critic2_backbone, util.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        # target_entropy = args.target_entropy if args.target_entropy \
        #     else -np.prod(env.action_space.shape)
        target_entropy = -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
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
        device=util.device
    )
    
    return sac_policy


def plot_predictions_rl(eval_env, src, tgt_full, pred, pl, pred_pl,iter=1):


    input_color = 'tab:blue'
    pred_color = 'tab:orange' #label="input",
    gt_color = 'tab:green'
    rl_color = 'tab:red'

    fig, ax1 = plt.subplots(figsize = (8,5.8), dpi=300)
                                    
    default_x_ticks = range(0, 181, 18)
    x_ticks = np.array(list(range(0, 31, 3)))
    plt.xticks(default_x_ticks, x_ticks)

    ax1.axvline(x=90, linestyle='--', c='black', alpha =0.7)

    plt.plot(range(90), eval_env.unnormalize(src.reshape(90,12)[:,0], idx = 0), color=input_color)
    plt.plot(range(90,180), eval_env.unnormalize(tgt_full.reshape(90,12)[:,0], idx = 0), label ="ground truth MAP", color=input_color)
    plt.plot(range(90,180), eval_env.unnormalize(pred.reshape(90,12)[:,0], idx = 0),  label ='prediction MAP', color=pred_color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(range(90,180), np.round(eval_env.unnormalize(pl.reshape(-1,1), idx = 12))*1000,'--',label ='ground truth PL', color=gt_color)
    ax2.plot(range(90,180), np.round(eval_env.unnormalize(pred_pl.reshape(-1,1), idx = 12))*1000,'--',label ='BCQ PL', color=rl_color)
    ax2.set_ylim((500,10000))
    ax1.legend(loc=3)
    ax2.legend(loc=1)

    ax1.set_ylabel('MAP (mmHg)',  fontsize=20)
    ax2.set_ylabel('Pump Speed',  fontsize=20)
    ax1.set_xlabel('Time (min)', fontsize=20)
    ax1.set_title(f"MAP Prediction and P-level")
    # wandb.log({f"plot_batch_{iter}": wandb.Image(fig)})

    plt.show()


def eval_acc(eval_env, y_pred_test, y_test):

    pred_unreg =  eval_env.unnormalize(np.array(y_pred_test), idx=12)
    real_unreg = eval_env.unnormalize(y_test, idx=12) 


    pl_pred_fl = np.round(pred_unreg.flatten())
    pl_true_fl = np.round(real_unreg.flatten())
    n = len(pl_pred_fl)


    accuracy = sum(pl_pred_fl == pl_true_fl)/n
    accuracy_1_off = (sum(pl_pred_fl == pl_true_fl) + sum(pl_pred_fl+1 == pl_true_fl)+sum(pl_pred_fl-1 == pl_true_fl))/n

    return accuracy, accuracy_1_off


def evaluate(policy, eval_env, eval_episodes, _terminal_counter):  
    policy.eval()
    obs = eval_env.reset()
    eval_ep_info_buffer = []
    num_episodes = 0
    episode_reward, episode_length = 0, 0
    terminal_counter = 0
    start_time = time.time()
    print(' # of eval episodes:', eval_episodes) 

    while num_episodes < eval_episodes:

        next_state_gt = eval_env.get_next_obs() 
        action = policy.sample_action(obs, deterministic=True) 
        action = action.repeat(90) #repeat the action for 90 steps
        full_pl = eval_env.get_full_pl() #for plotting

        #use next_obs only for evaluation
        next_obs, reward, terminal, _ = eval_env.step(action) #next state predictions            
        
        episode_reward += reward
        episode_length += 1

        terminal_counter += 1
        acc, acc_1_off = eval_acc(eval_env, action, full_pl)
        
        if num_episodes % 10 == 0:
            plot_predictions_rl(eval_env, obs.reshape(1,90,12), next_state_gt.reshape(1,90,12), next_obs.reshape(1,90,12), action.reshape(1,90), full_pl.reshape(1,90), num_episodes)
        
        #obs: (0,90) next_state_gt:(90,180) next_obs: (90,180), action: (90,180) act: (90,180)
        if terminal_counter == _terminal_counter:
            #plot the last round
            
            eval_ep_info_buffer.append(
                {"episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "episode_accurcy": acc, 
                    "episode_1_off_accuracy": acc_1_off}
            )
            terminal_counter = 0
            episode_reward, episode_length = 0, 0
            num_episodes +=1

        if num_episodes == env.data['observations'].shape[0]+1:
            break
        else:
            obs = eval_env.get_obs().reshape(1,-1)
    print("EVAL TIME: {:.3f}s".format(time.time() - start_time))
    return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval/episode_accuracy": [ep_info["episode_accurcy"] for ep_info in eval_ep_info_buffer],
            "eval/episode_1_off_accuracy": [ep_info["episode_1_off_accuracy"] for ep_info in eval_ep_info_buffer],
        }

def get_env():

    args.data_name = 'train' #to obtain norm_info
    norm_info = {'rwd_stds': None, 'rwd_means':None, 'scaler': None}
 
    # create env and dataset
    gym.envs.registration.register(
    id='Abiomed-v0',
    entry_point='abiomed_env:AbiomedEnv',  
    max_episode_steps = 1000,
    )
    kwargs = {"args": args, "logger": logger, 'scaler_info': norm_info}
    env = gym.make(args.task, **kwargs) #get the norm_info 
    
    env.scaler_info = {'rwd_stds': env.rwd_stds, 'rwd_means':env.rwd_means, 'scaler': env.scaler}
    args.data_name = 'test'
    kwargs = {"args": args, "logger": logger, 'scaler_info': env.scaler_info}
    env = gym.make(args.task, **kwargs)
   

    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    
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

    g.add_argument("--epoch", type=int, default=600) #1000
    g.add_argument("--step-per-epoch", type=int, default=1000) #1000
    g.add_argument("--batch-size", type=int, default=256)
    return parser
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument(
        "--algo-name",
        choices=["mbpo","mopo","bcq","bc"],
        default="mbpo",
        help="Which algorithm’s flags to load"
    )
    args_partial, remaining_argv = base.parse_known_args()
    parser = argparse.ArgumentParser(
        # keep the base flags and auto‐help
        parents=[base],
        description="Train your RL method"
    )

    # parser.add_argument("--algo-name", type=str, default="mbpo")
    parser.add_argument("--pretrained", type=bool, default=True)
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

    parser.add_argument("--seeds", type=int, nargs='+', default=[1,2])
    
    parser.add_argument("--eval_episodes", type=int, default=10)
    
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
    
    
    if args_partial.algo_name == "mopo":
        mopo_args(parser)
    elif args_partial.algo_name == "mbpo":
        mopo_args(parser)
    # elif args_partial.algo_name == "bcq":
    #     bcq_args(parser)
    # else:
    #     bc_args(parser)
    args = parser.parse_args()

    
    results = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
        log_file = f'seed_{seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
        # log_file = 'seed_1_0415_200911-walker2d_random_v0_mopo'
        log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)

        model_path = os.path.join(args.model_path, args.task, args.algo_name, log_file)
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger = Logger(writer=writer,log_path=log_path)
        model_logger = Logger(writer=writer,log_path=model_path)

        Devid = args.devid if args.device == 'cuda' else -1
        set_device_and_logger(Devid, logger, model_logger)

     
        #if you saved the whole model: policy = torch.load(os.path.join(model_logger.log_path, f'policy_{args.task}.pth'))
        env = get_env() 
        #if you saved the state_dict define the model class and load the model : write your own function
        policy = get_mopo()
        #change the policy path
        #load the state_dict and model
        policy_state_dict = torch.load(args.policy_path, map_location=f'cuda:{args.devid}')
        policy.load_state_dict(policy_state_dict)

        eval_info = evaluate(policy, env, args.eval_episodes, args.terminal_counter) 
        mean_return = np.mean(eval_info["eval/episode_reward"])
        std_return = np.std(eval_info["eval/episode_reward"])
        mean_length = np.mean(eval_info["eval/episode_length"])
        std_length = np.std(eval_info["eval/episode_length"])
        mean_accuracy = np.mean(eval_info["eval/episode_accuracy"])
        std_accuracy = np.std(eval_info["eval/episode_accuracy"])
        mean_1_off_accuracy = np.mean(eval_info["eval/episode_1_off_accuracy"])
        std_1_off_accuracy = np.std(eval_info["eval/episode_1_off_accuracy"])
        results.append({
            'seed': seed,
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_length': mean_length,
            'std_length': std_length,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_1_off_accuracy': mean_1_off_accuracy,
            'std_1_off_accuracy': std_1_off_accuracy,
        })
        
        print(f"Seed {seed} - Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    # Save results to CSV
    os.makedirs(os.path.join('results', args.task, args.algo_name), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join('results', args.task, args.algo_name, f"{args.task}_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")