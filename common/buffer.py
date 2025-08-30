import numpy as np
import torch
import tqdm


class ReplayBuffer:
    def __init__(
            self,
            buffer_size,
            obs_shape,
            obs_dtype,
            action_dim,
            action_dtype,
    ):
        self.max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self.ptr = 0
        self.size = 0
        self.timesteps = 6
        self.feature_dim = 12

        self.observations = np.zeros((self.max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self.max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self.max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self.max_size, 1), dtype=np.float32)


    def add(self, obs, next_obs, action, reward, terminal):
        # Copy to avoid modification by reference
        self.observations[self.ptr] = np.array(obs).copy()
        self.next_observations[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.terminals[self.ptr] = np.array(terminal).copy()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def load_dataset(self, dataset, env=None):
        if not isinstance(dataset, dict): #check if the dta is d4rl
            if isinstance(dataset, list):
                
                all_x = torch.cat([dataset[0].data, dataset[1].data, dataset[2].data], axis=0)
                all_pl = torch.cat([dataset[0].pl, dataset[1].pl, dataset[2].pl], axis=0)
                all_labels = torch.cat([dataset[0].labels, dataset[1].labels, dataset[2].labels], axis=0)
               
                print(all_x.shape, all_pl.shape, all_labels.shape)
            else:
                all_x = dataset.data
                all_pl = dataset.pl
                all_labels = dataset.labels

            reward_l = []
            done_l = []
            observation = all_x.reshape(-1,self.timesteps*(self.feature_dim))
            next_observation = torch.cat([all_labels.reshape(-1, self.timesteps, self.feature_dim-1), all_pl.reshape(-1, self.timesteps, 1)], axis = 2)
            next_observation = next_observation.reshape(-1,self.timesteps*(self.feature_dim))
            
            action = all_pl
            #take one number with majority voting among 6 numbers
            action_unnorm = np.array(env.world_model.unnorm_pl(action))
            action_1 = np.array([np.bincount(a.astype(int)).argmax() for a in action_unnorm]).reshape(-1,1)
            #normalize back
            action = env.world_model.normalize_pl(torch.Tensor(action_1))
            obs_reshaped = (observation.reshape(-1, self.timesteps, self.feature_dim)).clone() #shape (1,6,12)
            for i in tqdm.tqdm(range(action.shape[0])):

                if (env.gamma1 != 0.0) or (env.gamma2 != 0.0) or (env.gamma3 != 0.0):
                    
                    #change the last column of obs_reshaped with all_pl[i-1] after i==0
                    if i>0:
                        obs_reshaped[i,:,-1] = all_pl[i-1] 
                    reward = env._compute_reward(next_observation[i].reshape(-1,self.timesteps, self.feature_dim), obs_reshaped[i], action_1[i])
                    # action2 = [action2[1], action_1[i+1]] if (i+1)< action.shape[0] else None
                    
                else:	
                    reward = env._compute_reward(next_observation[i].reshape(-1,self.timesteps, self.feature_dim))

                reward_l.append(reward)
                done_l.append(np.array([0]))
            
            self.observations = np.array(observation)
            self.actions =  np.array(action)
            self.next_observations =  np.array(next_observation)
            self.rewards =  np.array(reward_l).reshape(-1,1)
            self.terminals = 1. -  np.array(done_l).reshape(-1,1)
            self.size = self.observations.shape[0]

            if np.all((self.observations==0)):
                raise ValueError("All observations are zero!")

           
            
        else:
            observations = np.array(dataset["observations"], dtype=self.obs_dtype)
            next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
            actions = np.array(dataset["actions"], dtype=self.action_dtype)
            rewards = np.array(dataset["rewards"]).reshape(-1, 1)
            terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

            self.observations = observations
            self.next_observations = next_observations
            self.actions = actions
            self.rewards = rewards
            self.terminals = terminals

            self.ptr = len(observations)
            self.size = len(observations)

    def add_batch(self, obs, next_obs, actions, rewards, terminals):
        batch_size = len(obs)
        if self.ptr + batch_size > self.max_size:
            begin = self.ptr
            end = self.max_size
            first_add_size = end - begin
            self.observations[begin:end] = np.array(obs[:first_add_size]).copy()
            self.next_observations[begin:end] = np.array(next_obs[:first_add_size]).copy()
            self.actions[begin:end] = np.array(actions[:first_add_size]).copy()
            self.rewards[begin:end] = np.array(rewards[:first_add_size]).copy()
            self.terminals[begin:end] = np.array(terminals[:first_add_size]).copy()

            begin = 0
            end = batch_size - first_add_size
            self.observations[begin:end] = np.array(obs[first_add_size:]).copy()
            self.next_observations[begin:end] = np.array(next_obs[first_add_size:]).copy()
            self.actions[begin:end] = np.array(actions[first_add_size:]).copy()
            self.rewards[begin:end] = np.array(rewards[first_add_size:]).copy()
            self.terminals[begin:end] = np.array(terminals[first_add_size:]).copy()

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

        else:
            begin = self.ptr
            end = self.ptr + batch_size
            self.observations[begin:end] = np.array(obs).copy()
            self.next_observations[begin:end] = np.array(next_obs).copy()
            self.actions[begin:end] = np.array(actions).copy()
            self.rewards[begin:end] = np.array(rewards).copy()
            self.terminals[begin:end] = np.array(terminals).copy()

            self.ptr = end
            self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        batch_indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": self.observations[batch_indices].copy(),
            "actions": self.actions[batch_indices].copy(),
            "next_observations": self.next_observations[batch_indices].copy(),
            "terminals": self.terminals[batch_indices].copy(),
            "rewards": self.rewards[batch_indices].copy()
        }

    def sample_all(self):
        return {
            "observations": self.observations[:self.size].copy(),
            "actions": self.actions[:self.size].copy(),
            "next_observations": self.next_observations[:self.size].copy(),
            "terminals": self.terminals[:self.size].copy(),
            "rewards": self.rewards[:self.size].copy()
        }
        

    @property
    def get_size(self):
        return self.size
