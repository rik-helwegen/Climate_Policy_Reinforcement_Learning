import matplotlib
# matplotlib.use('agg')
import pickle
import matplotlib.pyplot as plt
# from scipy.optimize import minimize
import numpy as np
# project imports
from dice_eval import Dice2007cjl as Dice
import numpy as np
import torch
import argparse
import os

import utils
from model import Actor, Critic
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
# from demo import *

# make sure this is the same as t_max in dice2007cjl
MAXSTEPS = 600


def var(tensor, volatile=False):
	return Variable(tensor, volatile=volatile)


def test_policy(model, critic):
	env = Dice()
	state = env.reset()
	rewards = []
	actions = []
	Q_values = []
	for t in range(MAXSTEPS):
		state = np.float32(state)
		state = Variable(torch.from_numpy(state.reshape(1, 7)))
		action_temp = model.forward(state).detach()
		action = action_temp.data.numpy().reshape(1)
		Q_value = critic.forward(state, action_temp)
		Q_values.append(float(Q_value))
		new_state, reward, done  = env.step(action[0])
		state = new_state
		actions.append(action[0])
		rewards.append(reward)

	# plot Q values
	# plt.figure()
	# x = list(range(0, len(Q_values)))
	# plt.plot(x, Q_values)
	# # plt.savefig("Q_values")
	# # plt.close()
	# plt.show()
	return rewards, actions

def test_policy_DP(DP_actions):
	env = Dice()
	state = env.reset()
	rewards = []
	counter = 0
	actions = []
	for step in range(len(DP_actions)):
		# 10 is the stepsize
		for t in range(10):
			if t > 0:
				if step == len(DP_actions)-1:
					action = DP_actions[step] + t * ((DP_actions[step] - DP_actions[step - 1])/10)
				else:
					action = DP_actions[step] + t * ((DP_actions[step + 1] - DP_actions[step])/10)
			else:
				action = DP_actions[step]
			new_state, reward, done  = env.step(action)
			state = new_state
			(K, M_AT, M_UP, M_LO, T_AT, T_LO, time) = new_state
			counter += 1
			# time_list.append(time)
			actions.append(action)
			rewards.append(reward)
	return rewards, actions

def random_policy():
	env = Dice()
	state = env.reset()
	rewards = []
	actions = []
	for t in range(MAXSTEPS):
		action = np.random.uniform()
		new_state, reward, done  = env.step(action)
		state = new_state
		rewards.append(reward)
		actions.append(action)
	return rewards, actions

def make_stats(policy_rewards, policy_actions, random_rewards, random_actions, DP_rewards, DP_actions, dp=False):
	print("The sum of rewards of the TRAINED policy is: ", sum(policy_rewards))
	print("The sum of rewards of the RANDOM policy is: ", sum(random_rewards))
	print("The sum of rewards of the OPTIMAL policy is: ", sum(DP_rewards))

	timevec = list(range(2005, 2005+MAXSTEPS))

	print(len(DP_rewards))
	print(len(random_rewards))
	print(len(policy_rewards))

	random_difference = [random_rewards[i] - DP_rewards[i] for i in range(len(policy_rewards))]
	policy_difference = [policy_rewards[i] - DP_rewards[i] for i in range(len(policy_rewards))]
	DP_difference = [ DP_rewards[i] - DP_rewards[i] for i in range(len(policy_rewards))]


	plt.figure()
	plt.subplot(3,2,1)
	plt.plot(timevec, random_actions)
	plt.title("Random policy")

	plt.subplot(3,2,3)
	plt.plot(timevec, policy_actions)
	plt.title('DDPG policy')

	plt.subplot(3,2,5)
	plt.plot(timevec, DP_actions)
	plt.title("Optimal policy")

	plt.subplot(3,2,2)
	plt.plot(timevec, random_difference)
	plt.title("Non-utilized reward")

	plt.subplot(3,2,4)
	plt.plot(timevec, policy_difference)
	plt.title('Non-utilized reward')

	plt.subplot(3,2,6)
	plt.plot(timevec, DP_difference)
	plt.title("Non-utilized reward")

	plt.tight_layout()
	# plt.savefig('result1')
	# plt.close()
	plt.show()

if __name__ == '__main__':
	# evaluate_episode_number = sys.argv[1]
	# show results of dp, works only for t_max=600
	dp = True
	model = Actor(7, 1, 1)
	critic = Critic(7, 1)

	# Adjust model to load:

	file = 'actor_RUN-3_user-mse10_MEps-100_Bsize-128_LRac-1e-05_LRcr-0.006_Tau-0.001_maxBuf-20000_explRt-0.2.pt'
	model.load_state_dict(torch.load('Models/' + file ))
	model.eval()

	critic.load_state_dict(torch.load('Models/' + file.replace('actor','critic')))
	critic.eval()
	# random model, does not need actor model since policy is random
	random_rewards, random_actions = random_policy()

	# test policy with trained actor model
	policy_rewards, policy_actions = test_policy(model, critic)
	DP_rewards = []
	DP_actions = []
	if dp == True:
		f = open('results.pckl', 'rb')
		DP_actions = pickle.load(f)
		f.close()
		DP_rewards, DP_actions = test_policy_DP(DP_actions.x)
		# print("The mean and variance for normalizing the rewards")
		# print(np.mean(DP_rewards), np.std(DP_rewards))
	make_stats(policy_rewards, policy_actions, random_rewards, random_actions, DP_rewards, DP_actions, dp)
