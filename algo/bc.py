import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import gym
import d4rl

# add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.policy_models import MLP, DiagGaussian, BasicActor
from common.logger import Logger
from common.util import set_device_and_logger

# run command in the mopo_abiomed directory: 
# python algo/bc.py --task halfcheetah-random-v0 --seeds 1 2 3 --model-dir saved_models/BC --epochs 25 --device_id 5

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Abiomed-v0")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument("--model-dir", type=str, default="bc_models")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--resultsdir", type=str, default="results")
    parser.add_argument("--data-name", type=str, default="train")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--algo-name", type=str, default="bc")
    return parser.parse_args()

def eval_acc(eval_env, y_pred_test, y_test):

    pred_unreg =  eval_env.unnormalize(np.array(y_pred_test), idx=12)
    real_unreg = eval_env.unnormalize(y_test, idx=12) 


    pl_pred_fl = np.round(pred_unreg.flatten())
    pl_true_fl = np.round(real_unreg.flatten())
    n = len(pl_pred_fl)


    accuracy = sum(pl_pred_fl == pl_true_fl)/n
    accuracy_1_off = (sum(pl_pred_fl == pl_true_fl) + sum(pl_pred_fl+1 == pl_true_fl)+sum(pl_pred_fl-1 == pl_true_fl))/n

    return accuracy, accuracy_1_off

class BehaviorCloning:
    def __init__(self, args, seed=0):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}" if args.device == "cuda" else "cpu")
        
        # Create environment and get dataset
        scaler_info = {'rwd_stds': None, 'rwd_means':None, 'scaler': None}
        if args.task == "Abiomed-v0":
            gym.envs.registration.register(
                id='Abiomed-v0',
                entry_point='abiomed_env:AbiomedEnv',
                max_episode_steps=1000,
            )
            kwargs = {"args": args, 'scaler_info': scaler_info}
            self.env = gym.make(args.task, **kwargs)
        else:
            self.env = gym.make(args.task)
        
        self.env.seed(seed)
        # self.dataset = d4rl.qlearning_dataset(self.env)
        dataset = self.env.data
        
        # Initialize model
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = np.prod(self.env.action_space.shape)

        
        actor_backbone = MLP(input_dim=self.obs_dim, hidden_dims=[256, 256])
        dist = DiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=self.action_dim,
            unbounded=True,
            conditioned_sigma=True
        )
        self.model = BasicActor(actor_backbone, dist, self.env.action_space, self.device)
        self.optimizer = optim.Adam(self.model.actor.parameters(), lr=args.lr)
        
        # Create data tensors
        self.obs = torch.FloatTensor(self.dataset["observations"]).to(self.device)
        self.actions = torch.FloatTensor(self.dataset["actions"]).to(self.device)
        
    def train(self):
        n_samples = len(self.obs)
        print(f"Training on {n_samples} samples")
        n_batches = n_samples // self.args.batch_size
        prev_loss = np.inf
        mse = nn.MSELoss()

        for epoch in range(self.args.epochs):
            total_loss = 0
            indices = np.random.permutation(n_samples)
            
            for i in range(n_batches):
                batch_idx = indices[i * self.args.batch_size:(i + 1) * self.args.batch_size]
                obs_batch = self.obs[batch_idx]
                actions_batch = self.actions[batch_idx]

                if self.args.task == "Abiomed-v0":
                    obs_batch = obs_batch[:, :90, :]  # trim to 90 timesteps
                    obs_batch = obs_batch.reshape(obs_batch.size(0), -1)  # flatten per sample
                
                self.optimizer.zero_grad()
                action, _ = self.model(obs_batch)
                loss = mse(action, actions_batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch + 1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
            # stop training if loss difference is less than 0.01
            if i > 0 and abs(avg_loss - prev_loss) < 0.0005:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            prev_loss = avg_loss

    def evaluate(self, eval_episodes=1000):
        self.model.eval()
        raw_obs = self.env.reset()
        if self.args.task == "Abiomed-v0":
            # raw_obs is (90,12), flatten to (1080,)
            obs = raw_obs.reshape(-1)
        else:
            obs = raw_obs

        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < eval_episodes:
            action = self.model.sample_action(obs, deterministic=True)

            next_raw_obs, reward, terminal, _ = self.env.step(action)
            episode_reward += reward
            episode_length += 1

            # if self.args.task == "Abiomed-v0":
            #     obs = next_raw_obs.reshape(-1)
            # else:
            #     obs = next_raw_obs

            if episode_length == 1:
                eval_ep_info_buffer.append({
                    "episode_reward": episode_reward,
                    "episode_length": episode_length
                })
                num_episodes += 1
                episode_reward, episode_length = 0, 0

                # reset and re‑flatten
                # raw_obs = self.env.reset()
                obs = self.env.get_obs()
                if self.args.task == "Abiomed-v0":
                    obs = raw_obs.reshape(-1)
                else:
                    obs = raw_obs

        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        # self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path, map_location='cuda:0'))


def main():
    args = get_args()

    t0 = datetime.now().strftime("%m%d_%H%M%S")

    # Set device
    if args.device_id < 0 or torch.cuda.is_available() == False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.device_id))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    print("setting device:", device)
    
    # Train and evaluate        
    results = []
    for seed in args.seeds:
        #     random.seed(seed)
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)

            # log_file = f'seed_{seed}_{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
        log_file = f'{t0}-{args.task.replace("-", "_")}_{args.algo_name}'
        log_path = os.path.join(args.logdir, args.task, args.algo_name, log_file)
        model_path = os.path.join(args.model_dir, args.task, args.algo_name, log_file)
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger = Logger(writer=writer,log_path=log_path)
        model_logger = Logger(writer=writer,log_path=model_path)
        
        Devid = args.device_id if args.device == 'cuda' else -1
        set_device_and_logger(Devid, logger, model_logger)

        bc = BehaviorCloning(args)
        # bc.train()        
        # # Save model
        # model_path = os.path.join(args.model_dir, f"bc_{args.task}_{t0}_seed_{seed}.pt")
        # bc.save_model(model_path)


        # # Evaluate
        # eval_results = bc.evaluate()
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
        
        # print(f"Seed {seed} - Mean Return: {mean_return:.2f} ± {std_return:.2f}")

        model_path = os.path.join(args.model_dir, f"bc_Abiomed-v0_0424_185145_seed_{seed}.pt")
        bc.load_model(model_path)

        # Evaluate
        eval_results = bc.evaluate()
        mean_return = np.mean(eval_results["eval/episode_reward"])
        std_return = np.std(eval_results["eval/episode_reward"])
        mean_length = np.mean(eval_results["eval/episode_length"])
        std_length = np.std(eval_results["eval/episode_length"])
        results.append({
            # 'seed': seed,
            'mean_return': mean_return,
            'std_return': std_return,
            'mean_length': mean_length,
            'std_length': std_length
        })

        print(f"Mean Return: {mean_return:.2f} ± {std_return:.2f}")

    # Save results to CSV
    os.makedirs(os.path.join(args.resultsdir, args.task, "bc"), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.resultsdir, args.task, "bc", f"bc_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")





if __name__ == "__main__":
    main() 