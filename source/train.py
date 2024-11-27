from env import RecommenderEnv, state_action_rewards
from dqn import DQNAgent

# Initialize environment and agent
env = RecommenderEnv(state_action_rewards, num_items=8885, window_size=window_size)
agent = DQNAgent(state_size=window_size, action_size=8885)

# Train the agent
num_episodes = 1000
batch_size = 32
for e in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        agent.replay(batch_size)

print("Training completed.")
