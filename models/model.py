import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic, self).__init__()

		self.hiddensize1 = 256
		self.hiddensize2 = 128
		self.hiddensize3 = 128
		self.actionsize = 128


		self.state_dim = state_dim
		self.action_dim = action_dim
		self.b0 = nn.BatchNorm1d(state_dim)

		self.f1 = nn.Linear(state_dim,self.hiddensize1)
		self.f1.weight.data = fanin_init(self.f1.weight.data.size())
		self.b1 = nn.BatchNorm1d(self.hiddensize1)

		self.f2 = nn.Linear(self.hiddensize1,self.hiddensize2)
		self.f2.weight.data = fanin_init(self.f2.weight.data.size())
		self.b2 = nn.BatchNorm1d(self.hiddensize2)

		self.fa = nn.Linear(action_dim,self.actionsize)
		self.fa.weight.data = fanin_init(self.fa.weight.data.size())

		self.f3 = nn.Linear(self.hiddensize2 + self.actionsize, self.hiddensize3)
		self.f3.weight.data = fanin_init(self.f3.weight.data.size())

		self.f4 = nn.Linear(self.hiddensize3,1)
		self.f4.weight.data.uniform_(-EPS,EPS)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		state = self.b0(state)
		hidden1 = self.b1(F.relu(self.f1(state)))
		hidden2 = self.b2(F.relu(self.f2(hidden1)))
		a1 = F.relu(self.fa(action))
		hidden3 = torch.cat((hidden2,a1),dim=1)
		hidden4 = F.relu(self.f3(hidden3))
		output = self.f4(hidden4)

		return output


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:return:
		"""
		super(Actor, self).__init__()

		self.hiddensize1 = 256
		self.hiddensize2 = 128
		self.hiddensize3 = 64

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim

		self.b0 = nn.BatchNorm1d(state_dim)
		self.b1 = nn.BatchNorm1d(self.hiddensize1)
		self.b2 = nn.BatchNorm1d(self.hiddensize2)
		self.b3 = nn.BatchNorm1d(self.hiddensize3)

		self.fc1 = nn.Linear(state_dim,self.hiddensize1)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(self.hiddensize1,self.hiddensize2)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(self.hiddensize2,self.hiddensize3)
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

		self.fc4 = nn.Linear(self.hiddensize3,action_dim)
		self.fc4.weight.data.uniform_(-EPS,EPS)

	def forward(self, state):
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""
		state = self.b0(state)
		x = self.b1(F.relu(self.fc1(state)))
		x = self.b2(F.relu(self.fc2(x)))
		x = self.b3(F.relu(self.fc3(x)))
		action = F.sigmoid(self.fc4(x))
		return action
