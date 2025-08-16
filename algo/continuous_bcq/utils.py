import numpy as np
import torch
import tqdm

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.timesteps = 6
		self.feature_dim = 12
		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
	
	def load_dataset(self, dataset, env=None):
		if not isinstance(dataset, dict): #check if the dta is d4rl
			if isinstance(dataset, tuple): #merge train val and test sets 
			
				all_x = torch.cat([dataset[0].data, dataset[1].data, dataset[2].data], axis=0)
				all_pl = torch.cat([dataset[0].pl, dataset[1].pl, dataset[2].pl], axis=0)
				all_labels = torch.cat([dataset[0].labels, dataset[1].labels, dataset[2].labels], axis=0)
			else:
				all_x = dataset.data
				all_pl = dataset.pl
				all_labels = dataset.labels
				
			# if fs:
			#     #select columns of 0 (MAP), 7 (PULSAT), 9 (HR) in all_x and all_labels
			#     all_x = all_x[:,:,[0, 7, 9]]
			#     all_labels = all_labels[:,:, [0, 7, 9]]
			#     print("FEATURE SELECTION APPLIED FOR 0, 7, 9")
			reward_l = []
			done_l = []
			observation = all_x.reshape(-1,self.timesteps*(self.feature_dim))
			next_observation = torch.cat([all_labels.reshape(-1, self.timesteps, self.feature_dim-1), all_pl.reshape(-1, self.timesteps, 1)], axis = 2)
			next_observation = next_observation.reshape(-1,self.timesteps*(self.feature_dim))
			
			action = all_pl
			#take one number with majority voting among 6 numbers
			action_unnorm = np.array(env.world_model.unnorm_pl(action))
			action = np.array([np.bincount(a.astype(int)).argmax() for a in action_unnorm]).reshape(-1,1)
			#normalize back
			action = env.world_model.normalize_pl(torch.Tensor(action))
			for i in tqdm.tqdm(range(action.shape[0])):
			
				reward = env._compute_reward(next_observation[i].reshape(-1,self.timesteps, self.feature_dim))
				reward_l.append(reward)
				done_l.append(np.array([0]))
			self.state = np.array(observation)
			self.action =  np.array(action)
			self.next_state =  np.array(next_observation)
			self.reward =  np.array(reward_l).reshape(-1,1)
			self.not_done = 1. -  np.array(done_l).reshape(-1,1) 
			self.ptr = len(self.state)
			self.size = len(self.state)
			self.max_size = self.state.shape[0]
			
		else:
			observations = np.array(dataset["observations"], dtype=self.obs_dtype)
			next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
			actions = np.array(dataset["actions"], dtype=self.action_dtype)
			rewards = np.array(dataset["rewards"]).reshape(-1, 1)
			terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

			self.state = observations
			self.next_state = next_observations
			self.action = actions
			self.reward = rewards
			self.not_done = terminals

			self.ptr = len(observations)
			self.size = len(observations)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def save(self, save_folder):
		np.save(f"{save_folder}_state.npy", self.state[:self.size])
		np.save(f"{save_folder}_action.npy", self.action[:self.size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer.shape[0], size)

		self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
		self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer[:self.size]
		self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]