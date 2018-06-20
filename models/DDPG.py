import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

import utils
import sys
import multiprocessing

# torch.set_num_threads(multiprocessing.cpu_count())

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


def var(tensor, volatile=False):
	# use cuda if is_available
	if torch.cuda.is_available():
		return Variable(tensor, volatile=volatile).cuda()
	else:
		return Variable(tensor, volatile=volatile)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.b1 = nn.BatchNorm1d(state_dim)
		self.l1 = nn.Linear(state_dim, 200)
		self.b2 = nn.BatchNorm1d(200)
		self.l2 = nn.Linear(200, 100)
		self.b3 = nn.BatchNorm1d(100)
		# self.l3 = nn.Linear(150, 100)
		self.l4 = nn.Linear(100, action_dim)

	def forward(self, x):
		x = self.b1(x)
		x = F.relu(self.l1(x))
		x = self.b2(x)
		x = F.relu(self.l2(x))
		x = self.b3(x)
		# x = F.relu(self.l3(x))
		x =  F.tanh(self.l4(x))
		return (x + 1)/2.0


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.b1 = nn.BatchNorm1d(state_dim)
		self.l1 = nn.Linear(state_dim, 250)
		# self.b2 = nn.BatchNorm1d(250)
		self.l2 = nn.Linear(250 + action_dim, 200)
		# self.l3 = nn.Linear(200, 250)
		# self.b3 = nn.BatchNorm1d(200)
		self.l4= nn.Linear(200, 1)

	def forward(self, x, u):
		x = self.b1(x)
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(torch.cat([x, u], 1)))
		# x = self.l3(x)
		x = self.l4(x)
		return x


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action)
		self.actor_target = Actor(state_dim, action_dim, max_action)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=1e-7)

		self.critic = Critic(state_dim, action_dim)
		self.critic_target = Critic(state_dim, action_dim)
		self.critic_target.load_state_dict(self.critic.state_dict())

		# use cuda if available
		if torch.cuda.is_available():
			self.actor = self.actor.cuda()
			self.actor_target = self.actor_target.cuda()
			self.critic = self.critic.cuda()
			self.critic_target = self.critic_target.cuda()

		self.criterion = nn.MSELoss()
		self.state_dim = state_dim
		self.episode = 0


	def select_action(self, state):
		state = var(torch.FloatTensor(state.reshape(-1, self.state_dim)), volatile=True)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.0001):
		actor_loss_list = []
		critic_loss_list = []
		for it in range(iterations):

			# Sample replay buffer
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = var(torch.FloatTensor(x))
			action = var(torch.FloatTensor(u))
			next_state = var(torch.FloatTensor(y), volatile=True)
			done = var(torch.FloatTensor(1 - d))
			reward = var(torch.FloatTensor(r))

			if torch.cuda.is_available():
				state = state.cuda()
				action = action.cuda()
				next_state = next_state.cuda()
				done = done.cuda()
				reward = reward.cuda()

			# Q target = reward + discount * Q(next_state, pi(next_state))
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (done * discount * target_Q)
			target_Q.volatile = False

			# Get current Q estimate
			current_Q = self.critic(state, action)


			# Compute critic loss
			critic_loss = self.criterion(current_Q, target_Q)

			learning_rate = 1e-7/float(self.episode + 1)
			# Optimize the critic
			self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=learning_rate , weight_decay=1e-2)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
			critic_loss_list.append(float(critic_loss))

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state)).mean()
			actor_loss_list.append(float(actor_loss))

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
		self.episode += 1
		return np.mean(actor_loss_list), np.mean(critic_loss_list)



	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor_%i.pt' % (directory, filename, self.episode))
		torch.save(self.critic.state_dict(), '%s/%s_critic_%i.pt' % (directory, filename, self.episode))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pt' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pt' % (directory, filename)))
