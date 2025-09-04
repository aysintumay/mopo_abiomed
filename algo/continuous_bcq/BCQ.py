import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a))
		return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
		


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		latent_dim = action_dim * 2

		self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.device = device


	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100):

		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

			# Variational Auto-Encoder Training
			# print(state)
			# print(action)
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()


			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				target_Q = reward + not_done * self.discount * target_Q

			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
			self.critic_optimizer.step()


			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()
		 	 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
			self.actor_optimizer.step()


			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			
			# return actor loss
		return actor_loss.item()

	def _move_optimizer_state_(optimizer: torch.optim.Optimizer, device: torch.device):
		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(device)

	# --------- NEW: save method ----------
	def save(self, path: str, include_targets: bool = True, include_optim: bool = True,
				extra: Optional[Dict[str, Any]] = None):
		"""
		Save BCQ (networks, optimizers, and hyperparams).
		"""
		chkpt = {
			"version": "bcq.v1",
			"actor": self.actor.state_dict(),
			"critic": self.critic.state_dict(),
			"vae": self.vae.state_dict(),
			"hyperparams": {
				"max_action": self.max_action,
				"action_dim": self.action_dim,
				"discount": self.discount,
				"tau": self.tau,
				"lmbda": self.lmbda,
			},
		}

		if include_targets:
			chkpt.update({
				"actor_target": self.actor_target.state_dict(),
				"critic_target": self.critic_target.state_dict(),
			})

		if include_optim:
			chkpt.update({
				"actor_optimizer": self.actor_optimizer.state_dict(),
				"critic_optimizer": self.critic_optimizer.state_dict(),
				"vae_optimizer": self.vae_optimizer.state_dict(),
			})

		if extra is not None:
			chkpt["extra"] = extra

		torch.save(chkpt, path)

	# --------- NEW: classmethod load ----------
	@classmethod
	def load(cls, path: str, state_dim: int, action_dim: int, device: torch.device,
			strict: bool = True):
		"""
		Load BCQ from a checkpoint created by `save`.
		Rebuilds the object, restores weights, targets, and optimizers if present.
		"""
		chkpt = torch.load(path, map_location=device)

		# Recreate model with saved hyperparams
		hp = chkpt["hyperparams"]
		model = cls(
			state_dim=state_dim,
			action_dim=action_dim,
			max_action=hp["max_action"],
			device=device,
			discount=hp["discount"],
			tau=hp["tau"],
			lmbda=hp["lmbda"],
		)

		# Load primary networks
		model.actor.load_state_dict(chkpt["actor"], strict=strict)
		model.critic.load_state_dict(chkpt["critic"], strict=strict)
		model.vae.load_state_dict(chkpt["vae"], strict=strict)

		# Load target networks if available; otherwise sync from primaries
		if "actor_target" in chkpt and "critic_target" in chkpt:
			model.actor_target.load_state_dict(chkpt["actor_target"], strict=strict)
			model.critic_target.load_state_dict(chkpt["critic_target"], strict=strict)
		else:
			model.actor_target.load_state_dict(model.actor.state_dict())
			model.critic_target.load_state_dict(model.critic.state_dict())

		# Load optimizers if available and move their states to device
		if "actor_optimizer" in chkpt:
			model.actor_optimizer.load_state_dict(chkpt["actor_optimizer"])
			cls._move_optimizer_state_(model.actor_optimizer, device)

		if "critic_optimizer" in chkpt:
			model.critic_optimizer.load_state_dict(chkpt["critic_optimizer"])
			cls._move_optimizer_state_(model.critic_optimizer, device)

		if "vae_optimizer" in chkpt:
			model.vae_optimizer.load_state_dict(chkpt["vae_optimizer"])
			cls._move_optimizer_state_(model.vae_optimizer, device)

		return model