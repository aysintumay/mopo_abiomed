import argparse
import datetime
import os
import random
import importlib
import wandb 

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mopo")
    # parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--policy_path" , type=str, default="log/halfcheetah-expert-v2/mopo/seed_1_0404_173844-halfcheetah_expert_v2_mopo/policy_halfcheetah-expert-v2.pth")
    
    parser.add_argument("--task", type=str, default="halfcheetah-expert-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=True)
    parser.add_argument('--target-entropy', type=int, default=-1) #-actio_dim
    parser.add_argument('--alpha-lr', type=float, default=3e-4)

    # dynamics model's arguments
    parser.add_argument("--dynamics-lr", type=float, default=0.001)
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=1.0) #1e=6
    parser.add_argument("--rollout-length", type=int, default=5) #1 
    parser.add_argument("--rollout-batch-size", type=int, default=5000) #50000
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=1) #1000
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
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
    parser.add_argument('-decoder_dropout', '--decoder_dropout', type=float, metavar='<size>', default=0,
                help='Set the tunable dropout.')
    parser.add_argument('-dim_model', '--dim_model', type=int, metavar='<size>', default=256,
                help='Set the number of encoder layers.')
    parser.add_argument('-path', '--path', type=str, metavar='<cohort>', 
                        default='/data/abiomed_tmp/processed',
                        help='Specify the path to read data.')
    return parser.parse_args()

def _evaluate(policy, eval_env, episodes):
        policy.eval()
        obs = eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < episodes:
            action = policy.sample_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = eval_env.step(action)
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                

                #d4rl don't have REF_MIN_SCORE and REF_MAX_SCORE for v2 environments
                dset_name = eval_env.unwrapped.spec.name+'-v0'
                print(d4rl.get_normalized_score(dset_name, np.array(episode_reward))*100)

                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = eval_env.reset()

        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }


def evaluate(policy, env, trainer, episodes=500, step_per_epoch = 1000):  
    policy.eval()
    obs = env.reset()
    
    eval_ep_info_buffer = []
    num_episodes = 0
    episode_reward, episode_length = 0, 0
    acc_total = 0
    acc_1_off_total = 0
    terminal_counter = 0
    while num_episodes <= episodes:
        act = env.get_pl()
        if num_episodes % 2 == 0:
            next_state_gt = env.get_next_obs() #next state gt
        else:
            next_state_gt = env.get_obs()
        action = policy.sample_action(obs, deterministic=True)
        action = action.repeat(90) #repeat the action for 90 steps

        full_pl = env.get_full_pl()
        next_obs, reward, terminal, _ = env.step(action) #next state predictions
        # print(f'reward func rew:{reward}')
        #obs: (0,90) next_state_gt:(90,180) next_obs: (90,180), action: (90,180) act: (90,180)
        
        episode_reward += reward
        episode_length += 1

        terminal_counter += 1
        acc, acc_1_off = trainer.eval_bcq(action, full_pl)
        acc_total += acc
        acc_1_off_total += acc_1_off

        obs = next_obs.reshape(1,-1)

        if terminal_counter == step_per_epoch:

            trainer.plot_predictions_rl(obs.reshape(1,90,12), next_state_gt.reshape(1,90,12), next_obs.reshape(1,90,12), action.reshape(1,90), full_pl.reshape(1,90), num_episodes)
            eval_ep_info_buffer.append(
                {"episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "episode_accurcy": acc_total/step_per_epoch, 
                    "episode_1_off_accuracy": acc_1_off_total/step_per_epoch}
            )
            terminal_counter = 0
            episode_reward, episode_length = 0, 0
            acc_total, acc_1_off_total = 0, 0
            obs = env.reset()
            num_episodes +=1

            print("episode_reward", episode_reward, 
                  "episode_length", episode_length,
                  "episode_accuracy", acc_total/step_per_epoch, 
                  "episode_1_off_accuracy", acc_1_off_total/step_per_epoch)

    return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval/episode_accuracy": [ep_info["episode_accurcy"] for ep_info in eval_ep_info_buffer],
            "eval/episode_1_off_accuracy": [ep_info["episode_1_off_accuracy"] for ep_info in eval_ep_info_buffer],
        },


def get_eval(policy, env, logger, trainer, args,):
    reward_l, acc_l, off_acc = [], [], []
    reward_std_l, acc_std_l, off_acc_std = [], [], []
    if args.task == 'Abiomed-v0':
        eval_info = evaluate(policy, env, trainer, episodes=args.eval_episodes, step_per_epoch = args.step_per_epoch)
    else:
        eval_info = _evaluate(policy, env, args.eval_episodes)
    ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
    ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
    if args.task == 'Abiomed-v0':
        ep_accuracy_mean, ep_accuracy_std = np.mean(eval_info[0]["eval/episode_accuracy"]), np.std(eval_info[0]["eval/episode_accuracy"])
        ep_1_off_accuracy_mean, ep_1_off_accuracy_std = np.mean(eval_info[0]["eval/episode_1_off_accuracy"]), np.std(eval_info[0]["eval/episode_1_off_accuracy"])

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
        # normalized_score_std = d4rl.get_normalized_score(env, ep_reward_std/ep_length_std)
        logger.record("normalized_episode_reward", normalized_score_mean, ep_length_mean, printed=False)
    else:
        logger.record("episode_reward", ep_reward_mean/ep_length_mean, args.eval_episodes, printed=False)

    if args.task == 'Abiomed-v0':
        logger.record("avg episode_accuracy", np.array(acc_l).mean(), args.eval_episodes, printed=False)
        logger.record("avg episode_1_off_accuracy", np.array(off_acc).mean(), args.eval_episodes, printed=False)


def test(args=get_args()):

    run = wandb.init(
                project=args.task,
                group=f"mopo",
                config=vars(args),
                # name=f"hpo_trial_{trial.number}", reinit=True
                )
    
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'test_seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
    log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)

    Devid = 0 if args.device == 'cuda' else -1
    set_device_and_logger(Devid,logger)

    # create env and dataset
    if args.task == "Abiomed-v0":
        gym.envs.registration.register(
        id='Abiomed-v0',
        entry_point='abiomed_env:AbiomedEnv',  
        max_episode_steps = 1000,
        )
        env = gym.make(args.task, args = args, logger = logger, data_name = "train", pretrained = True)
        dataset = d4rl.qlearning_dataset(env)
    else:
        env = gym.make(args.task)
        dataset = d4rl.qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    

    env.seed(args.seed)


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

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

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


    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    offline_buffer.load_dataset(dataset)
    model_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    # create dynamics model
    dynamics_model = TransitionModel(obs_space=env.observation_space,
                                     action_space=env.action_space,
                                     static_fns=static_fns,
                                     lr=args.dynamics_lr,
                                     **config["transition_params"]
                                     )    
    
    algo = MOPO(
        sac_policy,
        dynamics_model,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        logger=logger,
        **config["mopo_params"]
    )
   
    # create trainer
    trainer = Trainer(
        algo,
        # world_model,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=logger,
        log_freq=args.log_freq,
        run_id = run.id,
        eval_episodes=args.eval_episodes,
        
    )
    
    policy_state_dict = torch.load(args.policy_path)
    sac_policy.load_state_dict(policy_state_dict)

    get_eval(sac_policy, env, logger, trainer, args)
    
if __name__ == "__main__":

    
    test()
