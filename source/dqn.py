import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def load_training_data():
    train_data = []
    with open('dataset/train_data.txt', 'r') as f:
        for line in f:
            try:
                state_part, rest = line.split('],')
                state_str = state_part.strip('[')
                state = [int(x.strip().strip("'")) for x in state_str.split(',')]
                action, reward = rest.strip().split(',')
                action = int(action)
                reward = float(reward)
                train_data.append((state, action, reward))
            except Exception as e:
                continue
    return train_data


def plot_training_metrics(loss_history, reward_history, q_value_history, recommend_history):
    plt.figure(figsize=(20, 5))

    # 绘制loss曲线
    plt.subplot(141)
    plt.plot(loss_history)
    plt.title('Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    # 绘制平均reward曲线
    plt.subplot(142)
    plt.plot(reward_history)
    plt.title('Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

    # 绘制平均Q值曲线
    plt.subplot(143)
    plt.plot(q_value_history)
    plt.title('Average Q-Value per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-Value')

    # 绘制推荐频率TOP 10
    plt.subplot(144)
    items = sorted(recommend_history.items(), key=lambda x: x[1], reverse=True)[:10]
    items_id = [str(x[0]) for x in items]
    items_freq = [x[1] for x in items]
    plt.bar(items_id, items_freq)
    plt.title('Top 10 Recommended Items')
    plt.xlabel('Item ID')
    plt.ylabel('Recommendation Frequency')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()


def train_dqn(epochs=200, batch_size=32):
    state_size = 5
    action_size = 8885

    # 历史记录
    loss_history = []
    reward_history = []
    q_value_history = []
    recommend_history = {}

    try:
        train_data = load_training_data()
        print(f"Loaded {len(train_data)} training samples")

        if len(train_data) < batch_size:
            raise ValueError(f"Not enough training data")

        policy_net = DQN(state_size, action_size)
        target_net = DQN(state_size, action_size)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            total_reward = 0
            total_q = 0
            n_samples = 0
            np.random.shuffle(train_data)

            n_batches = 0
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                if len(batch) < batch_size:
                    continue

                states = torch.FloatTensor([x[0] for x in batch])
                actions = torch.LongTensor([x[1] for x in batch])
                rewards = torch.FloatTensor([x[2] for x in batch])

                q_values = policy_net(states)
                q_value = q_values.gather(1, actions.unsqueeze(1))

                loss = criterion(q_value.squeeze(), rewards)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_reward += rewards.sum().item()
                total_q += q_values.mean().item()
                n_samples += len(batch)

                # 记录推荐的商品
                predictions = q_values.argmax(dim=1)
                for pred in predictions:
                    item_id = pred.item()
                    recommend_history[item_id] = recommend_history.get(item_id, 0) + 1

                n_batches += 1

            # 更新目标网络
            if epoch % 5 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # 记录每个epoch的平均指标
            if n_batches > 0:
                avg_loss = total_loss / n_batches
                avg_reward = total_reward / n_samples  # 使用平均reward
                avg_q = total_q / n_batches

                loss_history.append(avg_loss)
                reward_history.append(avg_reward)
                q_value_history.append(avg_q)

                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}, Avg Q: {avg_q:.4f}")

        # 绘制训练指标
        plot_training_metrics(loss_history, reward_history, q_value_history, recommend_history)
        return policy_net

    except Exception as e:
        print(f"Training error: {str(e)}")
        return None

if __name__ == "__main__":
    model = train_dqn()
    if model is not None:
        torch.save(model.state_dict(), 'recommender_model.pth')
        print("Training completed and model saved!")
    else:
        print("Training failed!")