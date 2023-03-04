# Author:   Andy Edmondson
# Email:    andrew.edmondson@gmail.com
# Date:     3 Mar 2023
# Purpose:  Replay buffer for DDPG. This is a basic buffer, quite inefficient and not prioritised.
#
# Includes: ReplayBuffer class
#
# This implementation based on:
#       Sanghi, Nimish. Deep Reinforcement Learning with Python: With PyTorch, TensorFlow and OpenAI Gym.
#       New York: Apress, 2021. https://doi.org/10.1007/978-1-4842-6809-4.
#
# ----------


import numpy as np


class ReplayBuffer:
    def __init__(self, size=1e6):
        self.size = size
        self.buffer = []
        self.next_id = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size

    def sample(self, batch_size=32):
        idxs = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, done_flags = list(zip(*samples))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done_flags)