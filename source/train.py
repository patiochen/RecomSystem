import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from env import RecEnvironment
from dqn import DQNAgent
from data_preprocessing import state_action_rewards


def train_dqn():
    """
    训练DQN推荐系统
    """
    # 初始化环境和智能体
    env = RecEnvironment(state_action_rewards)
    state_size = 5
    action_size = 8885

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=64,
        learning_rate=1e-4,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update=200
    )

    # 训练参数设置
    n_episodes = 100
    max_steps = 50
    print_interval = 10
    rewards_history = []
    losses_history = []

    best_reward = float('-inf')

    # 训练循环
    for episode in tqdm(range(n_episodes)):
        state = np.array(env.reset(), dtype=np.float32)
        total_reward = 0
        episode_losses = []
        step_count = 0
        done = False

        while not done and step_count < max_steps:
            # 获取有效动作集合并转换为numpy数组
            valid_actions = np.array(env.get_valid_actions(), dtype=np.int64)

            # 选择动作
            if len(valid_actions) == 0:
                action = agent.choose_action(state)
            else:
                if np.random.random() < agent.epsilon:
                    action = np.random.choice(valid_actions)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        q_values = agent.q_network(state_tensor)
                        valid_q_values = q_values[0][valid_actions]
                        action_idx = valid_q_values.argmax().item()
                        action = valid_actions[action_idx]

            # 环境交互
            reward, done = env.step(action)

            # 获取下一个状态
            next_state = np.array(env.get_current_state(), dtype=np.float32)

            # 存储经验
            agent.remember(state, action, reward, next_state, done)

            # 训练网络
            if len(agent.memory) > agent.batch_size:
                loss = agent.train()
                if loss is not None:
                    episode_losses.append(loss)

            total_reward += reward
            state = next_state
            step_count += 1

        # 记录训练数据
        rewards_history.append(total_reward)
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            losses_history.append(avg_loss)

            # 检查loss是否异常
            if avg_loss > 100:
                print(f"\nWarning: High loss detected: {avg_loss}")

        # 更新最佳奖励
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.q_network.state_dict(), 'best_model.pth')

        # 打印训练进度
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(rewards_history[-print_interval:])
            avg_loss = np.mean(losses_history[-print_interval:]) if losses_history else 0
            print(f"\nEpisode {episode + 1}")
            print(f"Steps: {step_count}")
            print(f"Average Reward: {avg_reward:.3f}")
            print(f"Average Loss: {avg_loss:.3f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Memory Size: {len(agent.memory)}")
            print(f"Valid Actions: {len(valid_actions)}")
            print(f"Best Reward: {best_reward:.3f}")
            print("------------------------")

    # 绘制训练历史
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Rewards History')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(losses_history)
    plt.title('Loss History')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    return agent, rewards_history, losses_history


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 开始训练
    print("Starting DQN training...")
    agent, rewards, losses = train_dqn()
    print("Training completed!")