import matplotlib
matplotlib.use('agg')
import numpy as np
import torch
# import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
from dice2007cjl import Dice2007cjl as Dice
import matplotlib.pyplot as plt
import sys


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
	avg_reward = 0.
	for _ in xrange(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(obs))
			obs, reward, done = env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print "---------------------------------------"
	print "Evaluation over %d episodes: %f" % (eval_episodes, avg_reward)
	print "---------------------------------------"
	return avg_reward

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy_name", default="DDPG")					# Policy name
	parser.add_argument("--env_name", default="DICE")					# Dynamic integrated climate economic model
	parser.add_argument("--das", default=0)								# Wheter using Das4 or not(0=not 1==Using das4)
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1000, type=int)	# How many time steps purely random policy is run for
	parser.add_argument("--eval_freq", default=1e3, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	parser.add_argument("--save_models", action="store_true")			# Whether or not models are saved
	parser.add_argument("--expl_noise", default=0.4, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=128, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.001, type=float)				# Target network update rate
	# following arguments are for TD2 learning, not needed for DDPG
	parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	parser.add_argument("--user", default='', type=str)
	args = parser.parse_args()

	filename = "%s_%s_%s_%s" % (args.user, args.policy_name, args.env_name, str(args.seed))
	print "---------------------------------------"
	print "Settings: %s" % (filename)
	print "---------------------------------------"

	if not os.path.exists("./results"):
		os.makedirs("./results")
	if True and not os.path.exists("./pytorch_models"):
		os.makedirs("./pytorch_models")

	# env = gym.make(args.env_name)
	env = Dice()

	# Set seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = 7 # The six parameters  + The current time
	action_dim = 1 # Predicting mu
	max_action = 1 # Action space lies between 0 and 1

	# Initialize policy
	if args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action)
	elif args.policy_name == "OurDDPG": policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
	elif args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)

	replay_buffer = utils.ReplayBuffer()

	# Evaluate untrained policy
	evaluations = []

	total_timesteps = 0
	timesteps_since_eval = 0
	episode_num = 0
	average_reward = []
	all_rewards = []
	actor_loss_dev = []
	critic_loss_dev = []
	# first timestep done=True such that initialization runs
	done = True

	ouNoise = utils.OrnsteinUhlenbeckActionNoise(action_dim, 0 , 0.15, args.expl_noise)

	while total_timesteps < args.max_timesteps:

		# Done = true for first and last timestep
		if done:

			if total_timesteps != 0:
				print("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward)
				if args.policy_name == "TD3":
					policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
				else:
					actor_loss, critic_loss = policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
				actor_loss_dev.append(actor_loss)
				critic_loss_dev.append(critic_loss)
			# Evaluate episode
			if timesteps_since_eval >= args.eval_freq:
				timesteps_since_eval %= args.eval_freq
			evaluations.append(float(evaluate_policy(policy)))
			if True and episode_num % 10 == 0:
				policy.save(filename, directory="./pytorch_models")
				# np.save("./results/%s" % (file_name), evaluations)




			# Reset environment
			obs = env.reset()
			done = False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			if episode_num > 0:
				plt.figure()
				x = list(range(0, episode_num))
				plt.plot(x, evaluations)
				plt.title("Average reward")
				filename_reward = "results/figures/reward/" + filename + "_average_reward" 
				plt.savefig(filename_reward)
				plt.close()


		# Select action randomly or according to policy
		if total_timesteps < args.start_timesteps:
			action = np.random.uniform()
		else:
			action = policy.select_action(np.array(obs))
			if args.expl_noise != 0:
				action = (action + ouNoise.sample()).clip(0, 1)
				action = action[0]

				# previous noise
				# action = (action + np.random.normal(0, args.expl_noise, size=1)).clip(0, 1)
				# action = action[0]

		# Perform action
		new_obs, reward, done = env.step(action)
		done_bool = 0 if episode_timesteps + 1 == env.t_max else float(done)
		episode_reward += reward
		all_rewards.append(reward)
		average_reward.append(np.mean(all_rewards))

		# Store data in replay buffer
		replay_buffer.add((obs, new_obs, action, reward, done_bool))
		obs = new_obs

		episode_timesteps += 1
		total_timesteps += 1
		timesteps_since_eval += 1

		# plot results of loss development
		# save loss development as image.
		if total_timesteps%20==0 and args.das == 0:
			plt.figure()
			x = list(range(0, len(actor_loss_dev)))
			plt.subplot(1,2,1)
			plt.plot(x, actor_loss_dev)
			plt.title("avg. Actor loss dev over episodes")

			plt.subplot(1,2,2)
			plt.plot(x, critic_loss_dev)
			plt.title("avg. Cricic loss dev over episodes")
			filename_loss = "results/figures/loss/" + filename + "_loss"
			plt.savefig(filename_loss)
			plt.close()

	# Final evaluation
	evaluations.append(evaluate_policy(policy))
	if args.save_models: policy.save("%s" % (filename), directory="./pytorch_models")
np.save("./results/%s" % (filename), evaluations)
