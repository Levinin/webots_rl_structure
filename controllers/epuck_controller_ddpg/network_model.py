# Author:   Andy Edmondson
# Email:    andrew.edmondson@gmail.com
# Date:     3 Mar 2023
#
# Purpose:  Network model in pytorch for DDPG within webots
#
# References
# ----------
# DDPG paper:
#       Lillicrap, Timothy P., Jonathan J. Hunt, Alexander Pritzel, Nicolas Manfred Otto Heess, Tom Erez,
#       Yuval Tassa, David Silver, and Daan Wierstra. ‘Continuous Control with Deep Reinforcement Learning’.
#       CoRR abs/1509.02971 (2016).
#
# This implementation based on:
#       Sanghi, Nimish. Deep Reinforcement Learning with Python: With PyTorch, TensorFlow and OpenAI Gym.
#       New York: Apress, 2021. https://doi.org/10.1007/978-1-4842-6809-4.
#
#
# Changes from reference implementation
# -------------------------------------
# Resolved the continual swapping between GPU and CPU, keeping everything in GPU unless absolutely necessary.
# ==========


import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPActorCritic(nn.Module):
    """Combines the actor and critic into a single nn.Module."""
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.state_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]

        # build Q and policy functions
        self.q = MLPQFunction(self.state_dim, self.act_dim).cuda()
        self.policy = MLPActor(self.state_dim, self.act_dim, self.act_limit).cuda()

    def act(self, state):
        with torch.no_grad():
            # return self.policy(state).cpu().numpy()
            return self.policy(state)

    def get_action(self, s, noise_scale):
        a = self.act(torch.as_tensor(s, dtype=torch.float32).cuda())
        a = a + torch.FloatTensor(a.shape).normal_(mean=0.0, std=noise_scale).to("cuda")
        return a.clamp(-self.act_limit, self.act_limit)


class MLPQFunction(nn.Module):
    """Q-Function model for value estimation."""
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.Q = nn.Linear(256, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        q = self.Q(x)
        return torch.squeeze(q, -1)


class MLPActor(nn.Module):
    """Policy network for the actor critic."""
    def __init__(self, state_dim, act_dim, act_limit):
        super().__init__()
        self.act_limit = act_limit
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, act_dim)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.actor(x)
        x = torch.tanh(x)                       # to output in range(-1,1)
        x = self.act_limit * x
        return x



