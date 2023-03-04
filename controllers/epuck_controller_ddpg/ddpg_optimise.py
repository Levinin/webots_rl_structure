# Author:   Andy Edmondson
# Email:    andrew.edmondson@gmail.com
# Date:     3 March 2023
# Purpose:  Optimisation functions for DDPG
#
# Includes: compute_q_loss
#           compute_policy_loss
#           one_step_update
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
# ----------

import torch


def compute_q_loss(agent, target_network, batch, gamma=0.99):

    states, actions, rewards, next_states, done_flags = zip(*batch)

    # convert numpy array to torch tensors
    states = torch.tensor(states, dtype=torch.float).to("cuda")
    actions = torch.tensor(actions, dtype=torch.float).to("cuda")
    rewards = torch.tensor(rewards, dtype=torch.float).to("cuda")
    next_states = torch.tensor(next_states, dtype=torch.float).to("cuda")
    done_flags = torch.tensor(done_flags, dtype=torch.float).to("cuda")

    # get q-values for all actions in current states
    # use agent network
    predicted_qvalues = agent.q(states, actions)

    # Bellman backup for Q function
    with torch.no_grad():
        q__next_state_values = target_network.q(next_states, target_network.policy(next_states))
        target = rewards + gamma * (1 - done_flags) * q__next_state_values

    # MSE loss against Bellman backup
    loss_q = ((predicted_qvalues - target)**2).mean()

    return loss_q


def compute_policy_loss(agent, batch):

    states, _, _, _, _ = zip(*batch)

    # convert numpy array to torch tensors
    states = torch.tensor(states, dtype=torch.float).to("cuda")

    predicted_qvalues = agent.q(states, agent.policy(states))

    loss_policy = - predicted_qvalues.mean()

    return loss_policy


def one_step_update(agent, target_network, q_optimizer, policy_optimizer,
                    batch, gamma=0.99, polyak=0.995):
    # one step gradient for q-values
    q_optimizer.zero_grad()
    loss_q = compute_q_loss(agent, target_network, batch, gamma)
    loss_q.backward()
    q_optimizer.step()

    # Freeze Q-network
    for params in agent.q.parameters():
        params.requires_grad = False

    # one setep gradient for policy network
    policy_optimizer.zero_grad()
    loss_policy = compute_policy_loss(agent, batch)
    loss_policy.backward()
    policy_optimizer.step()

    # UnFreeze Q-network
    for params in agent.q.parameters():
        params.requires_grad = True

    # update target networks with polyak averaging
    with torch.no_grad():
        for params, params_target in zip(agent.parameters(), target_network.parameters()):
            params_target.data.mul_(1 - polyak)
            params_target.data.add_((polyak) * params.data)


