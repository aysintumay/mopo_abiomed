import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime
import gym
import d4rl
from torch.utils.tensorboard import SummaryWriter
# add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.policy_models import MLP, ActorProb, DiagGaussian
from common.logger import Logger
from common.util import set_device_and_logger

# run command: 
# python experiment_scripts/bc.py --task halfcheetah-random-v0 --seeds 1 2 3 --model-dir saved_models/BC --epochs 100 --device_id 5

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
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--device_id", type=int, default=0)
    return parser.parse_args()

class BehaviorCloning:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.device = torch.device(f"cuda:{args.device_id}" if args.device == "cuda" else "cpu")
        
        # Create environment and get dataset
        if args.task == "Abiomed-v0":
            gym.envs.registration.register(
                id='Abiomed-v0',
                entry_point='abiomed_env:AbiomedEnv',
                max_episode_steps=1000,
            )
            self.env = gym.make(args.task, args=args, logger=logger, data_name="train", pretrained=args.pretrained)
        else:
            self.env = gym.make(args.task)
            
        self.dataset = d4rl.qlearning_dataset(self.env)
        
        # Initialize model
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = np.prod(self.env.action_space.shape)
        
        # Compute action statistics for normalization
        self.action_mean = torch.FloatTensor(self.dataset["actions"].mean(axis=0)).to(self.device)
        self.action_std = torch.FloatTensor(self.dataset["actions"].std(axis=0)).to(self.device)
        self.action_std = torch.clamp(self.action_std, min=1e-3)  # Avoid division by zero
        
        # Scale actions to [-1, 1] range
        action_min = torch.FloatTensor(self.env.action_space.low).to(self.device)
        action_max = torch.FloatTensor(self.env.action_space.high).to(self.device)
        self.action_scale = (action_max - action_min) / 2.0
        self.action_bias = (action_max + action_min) / 2.0
        
        actor_backbone = MLP(input_dim=self.obs_dim, hidden_dims=[256, 256])
        dist = DiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=self.action_dim,
            unbounded=True,
            conditioned_sigma=True
        )
        self.model = ActorProb(actor_backbone, dist, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        
        # Create data tensors and normalize actions
        self.obs = torch.FloatTensor(self.dataset["observations"]).to(self.device)
        self.actions = torch.FloatTensor(self.dataset["actions"]).to(self.device)
        # Scale actions to [-1, 1] range
        self.actions = (self.actions - self.action_bias) / self.action_scale
        
    def train(self):
        n_samples = len(self.obs)
        n_batches = n_samples // self.args.batch_size
        prev_loss = np.inf

        for epoch in range(self.args.epochs):
            total_loss = 0
            indices = np.random.permutation(n_samples)
            
            for i in range(n_batches):
                batch_idx = indices[i * self.args.batch_size:(i + 1) * self.args.batch_size]
                obs_batch = self.obs[batch_idx]
                actions_batch = self.actions[batch_idx]
                
                self.optimizer.zero_grad()
                dist = self.model.get_dist(obs_batch)
                loss = -dist.log_prob(actions_batch).mean()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch + 1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
            # stop training if loss difference is less than 0.01
            if i > 0 and abs(avg_loss - prev_loss) < 0.001:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            prev_loss = avg_loss

    def evaluate(self, n_episodes=10):
        returns = []
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    dist = self.model.get_dist(obs_tensor)
                    action = dist.sample()
                    # Scale action back to environment's range
                    action = action * self.action_scale + self.action_bias
                    action = action.cpu().numpy()[0]
                    # Clip action to ensure it's within bounds
                    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                
            returns.append(total_reward)
            
        return np.mean(returns), np.std(returns)
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

def main():
    args = get_args()
    results = []
    
    for seed in args.seeds:
        print(f"Running experiment with seed {seed}")
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.device != "cpu":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        # Setup logging
        t0 = datetime.now().strftime("%m%d_%H%M%S")
        log_file = f'bc_seed_{seed}_{t0}-{args.task.replace("-", "_")}'
        log_path = os.path.join(args.logdir, args.task, "bc", log_file)
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger = Logger(writer=writer, log_path=log_path)
        
        # Set device
        Devid = args.device_id if args.device == 'cuda' else -1
        set_device_and_logger(Devid, logger)
        
        # Train and evaluate
        bc = BehaviorCloning(args, logger)
        bc.train()
        
        # Save model
        model_path = os.path.join(args.model_dir, f"bc_{args.task}_seed_{seed}.pt")
        bc.save_model(model_path)
        
        # Evaluate
        mean_return, std_return = bc.evaluate()
        results.append({
            'seed': seed,
            'mean_return': mean_return,
            'std_return': std_return
        })
        
        print(f"Seed {seed} - Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.logdir, args.task, "bc", f"bc_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main() 