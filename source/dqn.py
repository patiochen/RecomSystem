# File 2: bqn.py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from copy import deepcopy
from env import RecommenderEnv

# Step 2: Build the DQN Model
class QFunction(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes):
        super(QFunction, self).__init__()
        sizes = [state_size] + hidden_sizes + [action_size]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)

# Step 3: Train the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, options):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=options['replay_memory_size'])
        self.gamma = options['gamma']  # discount rate
        self.epsilon = options['epsilon']  # exploration rate
        self.epsilon_min = options['epsilon_min']
        self.epsilon_decay = options['epsilon_decay']
        self.model = QFunction(state_size, action_size, options['hidden_sizes'])
        self.target_model = deepcopy(self.model)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=options['alpha'], amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()

        # Freeze target network parameters
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Number of training steps so far
        self.n_steps = 0

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def epsilon_greedy(self, state):
        epsilon = self.epsilon
        nA = self.action_size

        # Transfer state to torch tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        # get Q value
        q_values = self.model(state_tensor).squeeze(0)

        # initialize the action probability
        action_probabilities = np.ones(nA) * (epsilon / nA)

        # find the best action then
        best_action = torch.argmax(q_values).item()

        # set action pro as (1 - epsilon)
        action_probabilities[best_action] += (1.0 - epsilon)

        return action_probabilities

    def compute_target_values(self, next_states, rewards, dones):
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        target_q = self.target_model(next_states)

        # get max q value for each state
        max_q = torch.max(target_q, dim=1)[0]

        # compute target values
        targets = rewards + (1 - dones) * self.gamma * max_q

        return targets

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        # Convert numpy arrays to torch tensors
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)

        # Current Q-values
        current_q = self.model(states)
        current_q = torch.gather(current_q, dim=1, index=actions.unsqueeze(1)).squeeze(-1)

        with torch.no_grad():
            target_q = self.compute_target_values(next_states, rewards, dones)

        # Calculate loss
        loss_q = self.loss_fn(current_q, target_q)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_episode(self, env, options):
        if not hasattr(self, 'epsilon'):
            self.epsilon = 1.0

        state = env.env_reset()

        for _ in range(options['steps']):
            # use epsilon-greedy to select actions
            action_probabilities = self.epsilon_greedy(state)
            action = np.random.choice(self.action_size, p=action_probabilities)

            # get next state and rewards
            next_state, reward, done = env.env_step(action)

            # store transition
            self.memorize(state, action, reward, next_state, done)

            # sample random minibatch of transitions
            if len(self.memory) >= options['batch_size']:
                self.replay(options['batch_size'])

            # update state
            state = next_state

            if done:
                break

            # update target after certain steps
            self.n_steps += 1
            if self.n_steps % options['update_target_estimator_every'] == 0:
                self.update_target_model()

        # set epsilon final->0.05
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def __str__(self):
        return "DQN"
