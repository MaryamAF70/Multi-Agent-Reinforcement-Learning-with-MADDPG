import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pettingzoo.mpe as mpe
from collections import deque
import random
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim*n_agents + action_dim*n_agents, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
env = mpe.simple_spread_v3.parallel_env(N=2, local_ratio=0.5, max_cycles=25)
state_dim = env.observation_spaces['agent_0'].shape[0]
action_dim = env.action_spaces['agent_0'].shape[0]
n_agents = 2
actors = [Actor(state_dim, action_dim) for _ in range(n_agents)]
critic = Critic(state_dim, action_dim, n_agents)
print("Environment initialized with 2 agents!")