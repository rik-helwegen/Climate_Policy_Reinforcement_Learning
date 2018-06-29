from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import utils
import model
import sys


class Trainer:

	def __init__(self, state_dim, action_dim, ram, LR_actor, LR_critic, gamma, tau, batchsize, expl_rate, version):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:param ram: replay memory buffer object
		:return:
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.LR_actor = LR_actor
		self.LR_critic = LR_critic
		self.gamma = gamma
		self.tau = tau
		self.ram = ram
		self.batchsize = batchsize
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim, 0, 0.15, expl_rate)
		self.action_lim = 1.0

		self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.LR_actor)

		self.critic = model.Critic(self.state_dim, self.action_dim)
		self.target_critic = model.Critic(self.state_dim, self.action_dim)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.LR_critic)

		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)

	def get_exploitation_action(self, state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		self.actor.eval()
		state = Variable(torch.from_numpy(state.reshape(1, 7)))
		action = self.actor.forward(state).detach()
		self.actor.train()
		return action.data.numpy().reshape(1)

	def get_exploration_action(self, state, eps):
		"""
		gets the action from actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		self.actor.eval()

		state = Variable(torch.from_numpy(state.reshape(1, 7)))
		action = self.actor.forward(state).detach()
		new_action = (action.data.numpy() + self.noise.sample(eps)).clip(0, 1)
		self.actor.train()
		return new_action.reshape(1)

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		s1,a1,r1,s2 = self.ram.sample(self.batchsize)

		s1 = Variable(torch.from_numpy(s1))
		a1 = Variable(torch.from_numpy(a1))
		r1 = Variable(torch.from_numpy(r1))
		s2 = Variable(torch.from_numpy(s2))


		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
		a2 = self.target_actor.forward(s2).detach()
		next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
		# y_exp = r + gamma*Q'( s2, pi'(s2))
		y_expected = r1 + self.gamma*next_val
		# y_pred = Q( s1, a1)
		y_predicted = torch.squeeze(self.critic.forward(s1, a1))
		# compute critic loss, and update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		# loss_critic = F.mse_loss(y_predicted, y_expected)

		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		# ---------------------- optimize actor ----------------------
		pred_a1 = self.actor.forward(s1)
		loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		utils.soft_update(self.target_actor, self.actor, self.tau)
		utils.soft_update(self.target_critic, self.critic, self.tau)

		return float(loss_actor.data), float(loss_critic.data)


	def save_models(self, episode_count, version):
		"""
		saves the actor and critic SOURCE models
		:param episode_count: the count of episodes iterated
		:return:
		"""
		torch.save(self.actor.state_dict(), './Models/' 'actor_' + str(version)+'.pt')
		torch.save(self.critic.state_dict(), './Models/' 'critic_' + str(version)+'.pt')
		print 'Models saved successfully'

	def load_models(self, episode):
		"""
		loads the target actor and critic models, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		## NOTE: SOURCE MODELS ARE SAVED NOW, NOT TARGET
		"""
		self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor_' + str(version)+'.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic_' + str(version)+'.pt'))
		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
		print 'Models loaded succesfully'
