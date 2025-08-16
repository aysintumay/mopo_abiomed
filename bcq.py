import argparse
import gym
import numpy as np
import os
import torch
import random
import tqdm
# import d4rl
import datetime 
import pandas as pd
import pickle
import algo.continuous_bcq.BCQ as BCQ
import algo.continuous_bcq.DDPG as DDPG
import algo.continuous_bcq.utils as utils
import sys
from torch.utils.tensorboard import SummaryWriter
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.buffer import ReplayBuffer
from common.logger import Logger
from common.util import set_device_and_logger
from helpers.plotter import plot_policy
import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from noisy_mujoco.abiomed_env.rl_env import AbiomedRLEnvFactory
from noisy_mujoco.wrappers import (RandomNormalNoisyActions,
                                      RandomNormalNoisyTransitions,
                                        RandomNormalNoisyTransitionsActions
                                    )
from noisy_mujoco.abiomed_env.cost_func import (compute_acp_cost,
                                                overall_acp_cost,
                                                compute_map_model_air,
                                                compute_hr_model_air,
                                                compute_pulsatility_model_air,
                                                aggregate_air_model,
                                                weaning_score_model,
                                                unstable_percentage_model,
                                                    super_metric
                                                    )
# python algo/bcq.py --task halfcheetah-random-v0 --seeds 1 2 3 --model-dir saved_models/BCQ  --device_id 5
# we can do --max_timesteps 1 --eval_episodes 1 for testing


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="halfcheetah-random-v0")               # OpenAI gym environment name
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument("--algo_name", type=str, default="bcq")                  # Algorithm name
    parser.add_argument("--model_dir", type=str, default="/abiomed/models/policy_models/bcq")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devid", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="results")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="")
        
    parser.add_argument("--eval_freq", default=2e4, type=float)     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ

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
    parser.add_argument("--fs", action="store_true", help = "doing feature selection")
    return parser.parse_args()

# Trains BCQ offline
def train_BCQ(env, state_dim, action_dim, max_action, device, output_dir, seed, args):
    # For saving files
    setting = f"{args.task}_{seed}"

    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)


    if args.data_path != "":
        with open(args.data_path, 'rb') as f:
            dataset = pickle.load(f)
        # dataset = {k: v[:5] for k, v in dataset.items()}
    else:
        if args.task == "abiomed":
            dataset1 = env.world_model.data_train
            dataset2 = env.world_model.data_val
            dataset3 = env.world_model.data_test
            dataset = (dataset1, dataset2, dataset3)
            buffer_len  = len(dataset1.data) + len(dataset2.data) + len(dataset3.data)
            # dataset.data = dataset.data[:5]
            # dataset.pl = dataset.pl[:5]
            # dataset.labels = dataset.labels[:5]

        else:
            dataset = env.get_dataset() 

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load_dataset(dataset, env) 

    print('Loaded buffer')

    evaluations = []
    episode_num = 0
    done = True 
    training_iters = 0
    with tqdm.tqdm(total=args.max_timesteps, desc="Training Progress", unit="step") as pbar:
        while training_iters < args.max_timesteps: 
                print('Train step:', training_iters)
                pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
                if training_iters % args.eval_freq == 0:
                    ev = eval_policy(policy, env, args.task, seed, args.eval_episodes)
                    evaluations.append(ev)
                
                print(f'Iteration {training_iters} Actor loss: {pol_vals:.2f}')

                training_iters += args.eval_freq
    #save model
    if not os.path.exists(os.path.join(output_dir, setting)):
        os.makedirs(os.path.join(output_dir, setting))
    save_dir = os.path.join(output_dir, setting, f"bcq_{args.max_timesteps}")
    policy.save(save_dir)
    print(f"Training completed. Model saved to {save_dir}")
    eval_final = eval_policy(policy, env, args.task, seed, eval_episodes=100, plot=True)
    return eval_final


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, env_name, seed, eval_episodes=10, mean=0, std=1,
                plot=False, writer=None):
    """
    Unified evaluator:
      - Creates eval_env from env_name and seeds it.
      - If env_name == 'abiomed': computes ACP, AIR metrics, unstable %, weaning, super metric.
      - Else: reports average return.
    Returns a dict of averages (and prints them).
    """
    # eval_env = gym.make(env_name)
    # try:
    #     eval_env.reset(seed=seed)
    # except TypeError:
    #     # older gym
    #     eval_env.seed(seed)

    if env_name == 'abiomed':
        avg_reward = 0.0
        avg_acp = 0.0

        total_map_air = 0.0
        total_hr_air = 0.0
        total_puls_air = 0.0
        total_agg_air = 0.0
        total_unstable_pct = 0.0
        total_wean_score = 0.0
        total_super = 0.0

        for k in range(eval_episodes):
            ep_states = []
            (state, info), done = eval_env.reset(), False     # normalized state (env-specific)
            all_states = info['all_states']                   # normalized
            all_states = np.concatenate([state.reshape(1, -1), all_states], axis=0)
            truncated = False

            while not (done or truncated):
                s_norm = (np.array(state).reshape(1, -1) - mean) / std
                action = policy.select_action(s_norm)

                next_state, reward, done, truncated, _ = eval_env.step(action)
                avg_reward += reward

                ep_states.append(state)  # store current obs
                state = next_state

            # per-episode metrics
            avg_acp += overall_acp_cost([eval_env.episode_actions])

            ep_states_np = np.asarray(ep_states, dtype=np.float32)

            wm = getattr(eval_env, 'world_model', None)
            if wm is None:
                wm = env.world_model  # fallback if world_model is global

            total_map_air  += compute_map_model_air(wm, ep_states_np, eval_env.episode_actions)
            total_hr_air   += compute_hr_model_air(wm, ep_states_np, eval_env.episode_actions)
            total_puls_air += compute_pulsatility_model_air(wm, ep_states_np, eval_env.episode_actions)
            total_agg_air  += aggregate_air_model(wm, ep_states_np, eval_env.episode_actions)
            total_super    += super_metric(wm, ep_states_np, eval_env.episode_actions)

            total_wean_score   += weaning_score_model(wm, ep_states_np, eval_env.episode_actions)
            total_unstable_pct += unstable_percentage_model(wm, ep_states_np)

            if (k == 2) and plot:
                # Plot needs next-state aligned sequence; append final state
                next_state_l = ep_states.copy()
                next_state_l.append(state)
                plot_policy(eval_env, next_state_l[1:], all_states)

        # Averages
        avg_reward /= eval_episodes
        acp_mean   = avg_acp / eval_episodes
        map_air    = total_map_air  / eval_episodes
        hr_air     = total_hr_air   / eval_episodes
        puls_air   = total_puls_air / eval_episodes
        agg_air    = total_agg_air  / eval_episodes
        unstable   = total_unstable_pct / eval_episodes
        weaning    = total_wean_score / eval_episodes
        super_mean = total_super / eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: Return {avg_reward:.3f}")
        print(f"ACP {acp_mean:.4f}")
        print(f"MAP AIR/ep: {map_air:.5f} | HR AIR/ep: {hr_air:.5f} | Pulsatility AIR/ep: {puls_air:.5f}")
        print(f"Aggregate AIR/ep: {agg_air:.5f}")
        print(f"Unstable hours (%): {unstable}")
        print(f"Weaning score: {weaning}")
        print(f"Super metric: {super_mean:.5f}")
        print("---------------------------------------")
        return {
                "avg_reward": avg_reward,
                "acp": acp_mean,
                "map_air": map_air,
                "hr": hr_air,
                "puls": puls_air,
                "agg_air": agg_air,
                "unstable_pct": unstable,
                "weaning_score": weaning,
                "super_metric": super_mean,
        }
    else:

        avg_reward = 0.
        for _ in range(eval_episodes):
            (state, _), done = eval_env.reset(), False
            truncated = False
            
            while not (done or truncated):
                s_norm = (np.array(state).reshape(1, -1) - mean) / std
                action = policy.select_action(s_norm)

                next_state, reward, done, truncated, _ = eval_env.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {avg_reward:.3f}")
        print("---------------------------------------")
        return {"avg_reward": avg_reward}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = get_args()

    print("---------------------------------------")	
    print(f"Setting: Training BCQ, Env: {args.task}, Seed: {args.seeds}")
    print("---------------------------------------")

    os.makedirs(args.model_dir, exist_ok=True)
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

        model_path = os.path.join(args.model_dir, args.task, args.algo_name, log_file)
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger = Logger(writer=writer,log_path=log_path)
        model_logger = Logger(writer=writer,log_path=model_path)

        Devid = args.devid if args.device == 'cuda' else -1
        set_device_and_logger(Devid, logger, model_logger)
        args.device = f"cuda:{Devid}" if torch.cuda.is_available() else "cpu"

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
                                        action_space_type="continuous",
                                        reward_type="smooth",
                                        normalize_rewards=True,
                                        seed=42,
                                        device= args.device,
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

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] 
        max_action = float(env.action_space.high[0])
        
        args.obs_shape = env.observation_space.shape
        args.action_dim = np.prod(env.action_space.shape)

        device = torch.device(args.devid if torch.cuda.is_available() else "cpu")

        evals = train_BCQ(env, state_dim, action_dim, max_action, device, args.model_dir, seed, args)
        # np.save(os.path.join(args.model_dir, f"{args.task}_{seed}"), evals)

        # eval_results = evals[-1]
        # Evaluate
        # mean_return = np.mean(eval_results["eval/episode_reward"])
        # std_return = np.std(eval_results["eval/episode_reward"])
        # mean_length = np.mean(eval_results["eval/episode_length"])
        # std_length = np.std(eval_results["eval/episode_length"])
        # results.append({
        #     'seed': seed,
        #     'mean_return': mean_return,
        #     'std_return': std_return,
        #     'mean_length': mean_length,
        #     'std_length': std_length
        # })
        evals['seed']= seed
        results.append(evals)
        # print(f"Seed {seed} - Mean Return: {mean_return:.2f} ± {std_return:.2f}")

    results_df = pd.DataFrame(results)
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")

    os.makedirs(os.path.join(args.logdir, args.task, "bcq"), exist_ok=True)
    results_path = os.path.join(args.logdir, args.task, "bcq", f"bcq_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
