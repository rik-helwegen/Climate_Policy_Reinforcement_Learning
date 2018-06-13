import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
# project imports
from dice2007cjl import Dice2007cjl as Dice
import numpy as np
import torch
import argparse
import os

import utils
import TD3
import OurDDPG
from DDPG import Actor
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
from demo import *

MAXSTEPS = 600


def var(tensor, volatile=False):
	if torch.cuda.is_available():
		return Variable(tensor, volatile=volatile).cuda()
	else:
		return Variable(tensor, volatile=volatile)

def test_policy(model):
	env = Dice()
	state = env.reset()
	rewards = []
	actions = []

	for t in range(MAXSTEPS):
		action = model(var(torch.FloatTensor(np.array(state))))
		action = np.clip(float(action), 0, 1)
		new_state, reward, done  = env.step(action)
		actions.append(action)
		rewards.append(reward)
	return rewards, actions

def test_policy_DP(DP_actions):
	env = Dice()
	state = env.reset()
	rewards = []
	counter = 0
	actions = []
	for step in range(len(coen_actions)):
		# 10 is the stepsize
		for t in range(10):
			if t > 0:
				if step == 59:
					action = DP[step] + t * ((DP_actions[step] - DP_actions[step - 1])/10)
				else: 
					action = DP_actions[step] + t * ((DP_actions[step + 1] - DP_actions[step])/10)
			else:
				action = DP_actions[step] 
			new_state, reward, done  = env.step(action)
			actions.append(action)
			rewards.append(reward)
			counter += 1
	return rewards, actions

def random_policy():
	env = Dice()
	state = env.reset()
	rewards = []
	actions = []
	for t in range(MAXSTEPS):
		action = np.random.uniform()
		new_state, reward, done  = env.step(action)
		rewards.append(reward)
		actions.append(action)
	return rewards, actions

def make_stats(policy_rewards, policy_actions, random_rewards, random_actions, DP_rewards, DP_actions):
	print("The sum of rewards of the TRAINED policy is: ", sum(policy_rewards))
	print("The sum of rewards of the RANDOM policy is: ", sum(random_rewards))
	print("The sum of rewards of the DP policy is: ", sum(DP_rewards))

	timevec = list(range(2005, 2005+MAXSTEPS))
	plt.figure()

	plt.subplot(3,2,1)
	plt.plot(timevec, random_rewards)
	plt.title("Rewards for random policy")

	plt.subplot(3,2,2)
	plt.plot(timevec, random_actions)
	plt.title('Mu (action) for random policy')


	plt.subplot(3,2,3)
	plt.plot(timevec, policy_rewards)
	plt.title("Rewards for trained policy")

	plt.subplot(3,2,4)
	plt.plot(timevec, policy_actions)
	plt.title('Mu (action) for trained policy')


	plt.subplot(3,2,5)
	plt.plot(timevec, DP_rewards)
	plt.title("Rewards for DP policy")


	plt.subplot(3,2,6)
	plt.plot(timevec, DP_actions)
	plt.title('Mu (action) for DP policy')

	plt.show()
	

if __name__ == '__main__':
	model = Actor(6, 1, 1)
	# model.load_state_dict(torch.load('mytraining.pt'))
	policy_rewards, policy_actions = test_policy(model)
	random_rewards, random_actions = random_policy()
	f = open('results.pckl', 'rb')
	DP_actions = pickle.load(f)
	f.close()
	DP_rewards, DP_actions = test_policy_DP(DP_actions.x)

	make_stats(policy_rewards, policy_actions, random_rewards, random_actions, DP_rewards, DP_actions)


