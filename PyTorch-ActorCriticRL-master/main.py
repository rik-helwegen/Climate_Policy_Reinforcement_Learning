from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc
from dice2007cjl import Dice2007cjl as Dice
import sys


import train
import buffer

env = Dice()# env = gym.make('Pendulum-v0')

MAX_EPISODES = 100
MAX_STEPS = 100
MAX_BUFFER = 100000
# S_DIM = env.observation_space.shape[0]
# A_DIM = env.action_space.shape[0]
# A_MAX = env.action_space.high[0]
S_DIM = 7
A_DIM = 1
A_MAX = 1

print(' State Dimensions :- ', S_DIM)
print( ' Action Dimensions :- ', A_DIM)
print( ' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
average_reward = 0
for _ep in range(MAX_EPISODES):
	observation = env.reset()
	print('EPISODE :- ', _ep)
	if _ep%5 == 0:
		print("Doing an explotation episode")
	reward_per_episode = 0
	for r in range(MAX_STEPS):
		#env.render()

		state = np.float32(observation)

		# action = trainer.get_exploration_action(state)
		if _ep%5 == 0:
			# validate every 5th episode
			action = trainer.get_exploitation_action(state)
			# print(action[0])
		else:
			# get action based on observation, use exploration policy here
			action = trainer.get_exploration_action(state)

		new_observation, reward, done = env.step(action)
		average_reward += reward
		reward_per_episode += reward

		# # dont update if this is validation
		# if _ep%50 == 0 or _ep>450:
		# 	continue

		if done:
			new_state = None

		else:
			new_state = np.float32(new_observation)

			# push this exp in ram
			ram.add(state, action, reward, new_state)

		observation = new_observation

		# perform optimization
		trainer.optimize()
		if done:
			break
	print("The average reward of last episode is: ", reward_per_episode/(r+1))


	# check memory consumption and clear memory
	gc.collect()
	# process = psutil.Process(os.getpid())
	# print(process.memory_info().rss)

	if _ep%100 == 0:
		trainer.save_models(_ep)


print('Completed episodes')
