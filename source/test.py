# File 5: test.py
import torch
import numpy as np
from env import RecommenderEnv
from dqn import DQNAgent
from data_preprocessing import state_action_rewards

# Initialize environment and agent
num_items = 8885
window_size = 2
env = RecommenderEnv(state_action_rewards, num_items=num_items, window_size=window_size)

# Initialize agent with the same parameters as during training
agent = DQNAgent(state_size=window_size, action_size=num_items, options={
    'replay_memory_size': 10000,
    'gamma': 0.99,
    'epsilon': 0.0,  # Set epsilon to 0 for purely greedy policy during testing
    'epsilon_min': 0.0,  # Ensure no exploration during testing,
    'epsilon_decay': 1.0,  # Disable epsilon decay during testing
    'hidden_sizes': [128, 64],
    'alpha': 0.001,
    'batch_size': 32
})

# Load the trained model
agent.model.load_state_dict(torch.load("recommender_model.pth"))
agent.model.eval()  # Set to evaluation mode

# Test the agent
num_test_episodes = 5
max_steps_per_episode = 100  # Limit maximum steps per episode
for e in range(num_test_episodes):
    state = env.env_reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < max_steps_per_episode:
        # Choose the best action based on current policy
        with torch.no_grad():  # Disable gradient calculation for testing
            action = agent.model(torch.tensor(state, dtype=torch.float32)).argmax().item()
        next_state, reward, done = env.env_step(action)
        total_reward += reward
        step_count += 1
        state = next_state

        # Print each step's details
        print(f"Test Episode {e + 1}, Step {step_count}, Action: {action}, Reward: {reward}")

    # Print the total reward after each episode
    print(f"Test Episode {e + 1} completed - Total Reward: {total_reward:.2f}")

print("Testing completed.")
