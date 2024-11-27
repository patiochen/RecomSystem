import numpy as np
from data_preprocessing import state_action_rewards

# Define the Reinforcement Learning Environment
class RecommenderEnv:
    def __init__(self, data, num_items, window_size):
        self.data = data
        self.num_items = num_items
        self.window_size = window_size
        self.env_reset()

    def env_reset(self):
        self.current_step = 0
        self.state = [0] * self.window_size
        return self.state

    def env_step(self, action):
        reward = 0
        done = False

        if self.current_step < len(self.data) - 1:
            _, next_action, reward = self.data[self.current_step]
            self.state = self.state[1:] + [action]
            self.current_step += 1
        else:
            done = True

        return self.state, reward, done