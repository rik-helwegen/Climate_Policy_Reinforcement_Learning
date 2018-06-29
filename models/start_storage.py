# create initial storage
import numpy as np
import torch
# import OurDDPG
import sys
import argparse
import cPickle as pickle
import os.path
from dice2007cjl import Dice2007cjl as Dice

def rollout_samples():
    env = Dice()
    state = env.reset()
    samples = []
    # Adjust for differnt models
    max_steps = 600
    for t in range(max_steps):
        action = np.random.uniform()
        new_state, reward, done  = env.step(action)
        samples.append((state, new_state, action, reward, done))
        state = new_state

    return samples

def create_init_for_R(total_samples=100000):

    # set filename
    filename = 'R'+str(total_samples)+'.pkl'

    # read if exists
    if os.path.isfile(filename):
        with open(filename, "rb") as f:   # Unpickling
            storage = pickle.load(f)
            # print 'Loaded storage from memory'
    # write otherwise
    else:
        storage =[]
         # number of rollouts:
        rollouts = (total_samples-total_samples%600)/600+1
        for n in range(rollouts):
            single_rollout = rollout_samples()
            storage += single_rollout
        print 'Created initial sample pool of size %i' %len(storage)

    storage = storage[-total_samples:]
    rewards = [x[3] for x in storage]
    print('reward statistics:')
    print('- std:')
    print np.std(rewards)
    print('- mean:')
    print np.mean(rewards)
    # store file
    with open(filename, 'wb') as f:
        pickle.dump(storage, f)
    # print 'Saved file as %s, with %i entries' %(filename, len(storage))
    return storage

if __name__ == "__main__":
    create_init_for_R()
