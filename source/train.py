# File: train.py
import numpy as np
import torch
from env import RecommenderEnv
from dqn import DQNAgent
from data_preprocessing import state_action_rewards

# Initialize environment and agent
num_items = 8885
window_size = 2
env = RecommenderEnv(state_action_rewards, num_items=num_items, window_size=window_size)
agent = DQNAgent(state_size=window_size, action_size=num_items, options={
    'replay_memory_size': 10000,
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.995,
    'hidden_sizes': [128, 64],
    'alpha': 0.001,
    'batch_size': 32,
    'steps': 20,
    'update_target_estimator_every': 1000
})

# Train the agent
num_episodes = 10
batch_size = 32
max_steps_per_episode = 100  # Limit maximum steps per episode

for e in range(num_episodes):
    state = env.env_reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < max_steps_per_episode:
        action_probabilities = agent.epsilon_greedy(state)
        action = np.random.choice(agent.action_size, p=action_probabilities)
        next_state, reward, done = env.env_step(action)
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        step_count += 1
        agent.replay(batch_size)

        # Print each step's details
        if step_count % 10 == 0:
            print(f"Episode {e + 1}, Step {step_count}, Action: {action}, Reward: {reward}")

    # Print episode completion details
    print(f"Episode {e + 1}/{num_episodes} completed - Total Reward: {total_reward:.2f}")

# Save the trained model
torch.save(agent.model.state_dict(), "recommender_model.pth")
print("Model saved as recommender_model.pth")

print("Training completed.")
