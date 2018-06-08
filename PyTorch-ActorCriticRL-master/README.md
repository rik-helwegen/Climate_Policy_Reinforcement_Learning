# Using PyTorch-ActorCriticRL to train the DSICE economic climate model

PyTorch implementation of continuous action actor-critic algorithm. The algorithm uses DeepMind's Deep Deterministic Policy Gradient [DDPG](https://arxiv.org/abs/1509.02971) method for updating the actor and critic networks along with [Ornstein–Uhlenbeck](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) process for exploring in continuous action space while using a Deterministic policy.

## DDPG

[DDPG](https://arxiv.org/abs/1509.02971) is a policy gradient alogrithm, that uses stochastic behaviour policy for exploration (Ornstein-Uhlenbeck in this case) and outputs a deterministic target policy, which is easier to learn.
