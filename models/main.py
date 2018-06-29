from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import os
# import psutil
import gc
import sys
import train
import buffer
from dice2007cjl import Dice2007cjl as Dice
from dice_eval import Dice2007cjl as Dice_eval

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
from start_storage import *

# user
user = sys.argv[1]

# Number of episodes
MAX_EPISODES = 100

# Max number of steps until terminate
MAX_STEPS = 1000

# HYPER PARAMETERS
# The batch size for training
BATCH_SIZE = 128

# Learning rate
LR_actor = 0.0001
LR_critic = 0.01

# LR_actor_list = [0.000001,0.000001,0.000001,0.000001,0.00001,0.00001,0.00001,0.0001,0.0001,0.0001,]
# LR_critic_list = [0.0001,0.0006,0.001,0.00001,0.0001,0.0006,0.001,0.0001,0.0006,0.001]

# Discount factor
GAMMA = 1.0/1.015

# Update Target network
TAU = 0.0005

# Size of the Replay Memory
MAX_BUFFER = 20000

# exploration rate
expl_rate = 0.2

# Dimensions
S_DIM = 7
A_DIM = 1


# np.random.seed(1234)
# torch.manual_seed(1234)


def evaluate():

	state = env_eval.reset()
	rewards = 0.
	done = False
	actions = []
	while not done:
		action = trainer.get_exploitation_action(np.float32(state))
		state, reward, done = env_eval.step(action[0])
		rewards += reward
		actions.append(action[0])
	print("The sum of rewards is: ", rewards)
	return rewards, actions


# reliability, test n times
n = 10


for n_version in range(n):

	# LR_actor = LR_actor_list[n_version]
	# LR_critic = LR_critic_list[n_version]

	version = 'RUN-%i_user-%s_MEps-%s_Bsize-%s_LRac-%s_LRcr-%s_Tau-%s_maxBuf-%s_explRt-%s' %(n_version, str(user),str(MAX_EPISODES), str(BATCH_SIZE),str(LR_actor), str(LR_critic), str(TAU), str(MAX_BUFFER), str(expl_rate))


	# Statistics
	reward_plot = []

	# Initializations
	ram = buffer.MemoryBuffer(MAX_BUFFER)

	# Initialize buffer:
	init_buffer = create_init_for_R(MAX_BUFFER)
	# print(np.float32(list(init_buffer[0][0])))
	print 'Initialize ram...'
	for obsv in init_buffer:
		ram.add(np.float32(list(obsv[0])),list([obsv[2]]),obsv[3],np.float32(list(obsv[1])))
	print 'Ram initialized'

	trainer = train.Trainer(S_DIM, A_DIM, ram, LR_actor, LR_critic, GAMMA, TAU, BATCH_SIZE, expl_rate, version)




	env = Dice()
	env_eval = Dice_eval()

	highest_reward = -np.inf
	best_policy = []

	actor_loss_dev = []
	critic_loss_dev = []


	for _ep in range(MAX_EPISODES):
		observation = env.reset()
		print 'EPISODE :- ', _ep
		for r in range(MAX_STEPS):
			state = np.float32(observation)

			action = trainer.get_exploration_action(state,  _ep)
			new_observation, reward, done = env.step(action[0])

			if done:
				new_state = None
			else:
				new_state = np.float32(new_observation)
				# push this exp in ram
				ram.add(state, action, reward, new_state)

			observation = new_observation

			# perform optimization
			if ram.len > 300:
				loss_actor, loss_critic = trainer.optimize()
				actor_loss_dev.append(loss_actor)
				critic_loss_dev.append(loss_critic)
			if done:
				sum_rewards, policy = evaluate()

				if sum_rewards > highest_reward :
					highest_reward = sum_rewards
					best_policy = policy
					trainer.save_models(_ep, version)


				reward_plot.append(sum_rewards)

				plt.figure()
				plt.subplot(3, 1, 1)
				x = list(range(0, _ep + 1))
				plt.plot(x, reward_plot)
				plt.title('sum of reward over episodes')

				plt.subplot(3, 1, 2)
				x = list(range(0, len(policy)))
				plt.plot(x, policy)
				plt.title('Policy behaviour')

				plt.subplot(3,1,3)
				x = list(range(len(best_policy)))
				plt.plot(x, best_policy)
				plt.title('Best policy behaviour so far')

				figname = './reward_figs/Stats' + str(version) + '.jpg'
				plt.savefig(figname)
				plt.close()

				plt.figure()
				plt.subplot(2,1,1)
				plt.plot(list(range(len(actor_loss_dev))),actor_loss_dev)
				plt.title('loss actor develop over episodes')
				plt.subplot(2,1,2)
				plt.plot(list(range(len(critic_loss_dev))),critic_loss_dev)
				plt.title('loss critic develop over episodes')
				figname = './loss_figs/losses_' + str(version) + '.jpg'
				plt.savefig(figname)
				plt.close()
				break

		# check memory consumption and clear memory
		gc.collect()
		# process = psutil.Process(os.getpid())
		# print(process.memory_info().rss)
