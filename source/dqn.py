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
    """load training set"""
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


def load_test_data():
    """加载测试数据"""
    test_data = []
    with open('dataset/test_data.txt', 'r') as f:
        for line in f:
            try:
                state_part, rest = line.split('],')
                state_str = state_part.strip('[')
                state = [int(x.strip().strip("'")) for x in state_str.split(',')]
                action, reward = rest.strip().split(',')
                action = int(action)
                reward = float(reward)
                test_data.append((state, action, reward))
            except Exception as e:
                continue
    return test_data


def plot_training_metrics(loss_history, reward_history, q_value_history, recommend_history):
    """ploy figures"""
    plt.figure(figsize=(20, 5))

    # loss function
    plt.subplot(141)
    plt.plot(loss_history)
    plt.title('Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    # reward - fig2
    plt.subplot(142)
    plt.plot(reward_history)
    plt.title('Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

    # Q-value - fig3
    plt.subplot(143)
    plt.plot(q_value_history)
    plt.title('Average Q-Value per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-Value')

    # top10 -fig4
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


def plot_episode_recommendation(model, test_data, episode_length=200):
    """plot rec seq"""
    recommended_items = []
    actual_items = []
    rewards = []

    with torch.no_grad():
        for state, true_action, reward in test_data[:episode_length]:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            recommended_item = q_values.argmax().item()

            recommended_items.append(recommended_item)
            actual_items.append(true_action)
            rewards.append(reward if recommended_item == true_action else 0)

    plt.figure(figsize=(15, 8))

    # rec seq plot
    plt.subplot(2, 1, 1)
    plt.plot(range(len(recommended_items)), recommended_items, 'b-', label='Recommended Items', marker='o')
    plt.plot(range(len(actual_items)), actual_items, 'r--', label='Actual Items', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Item ID')
    plt.title('Recommendations vs Actual Items over Time')
    plt.legend()
    plt.grid(True)

    # reward seq plot
    plt.subplot(2, 1, 2)
    plt.plot(range(len(rewards)), rewards, 'g-', label='Reward', marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title('Rewards over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('episode_recommendation.png')
    plt.close()


def train_dqn(epochs=1000, batch_size=32):
    state_size = 5
    action_size = 8885

    try:
        # load traning set
        train_data = load_training_data()
        print(f"Loaded {len(train_data)} training samples")

        # create nn
        policy_net = DQN(state_size, action_size)
        target_net = DQN(state_size, action_size)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # tranining
        loss_history = []
        reward_history = []
        q_value_history = []
        recommend_history = {}

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

                predictions = q_values.argmax(dim=1)
                for pred in predictions:
                    item_id = pred.item()
                    recommend_history[item_id] = recommend_history.get(item_id, 0) + 1

                n_batches += 1

            # update target network
            if epoch % 5 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # record tranning label
            if n_batches > 0:
                avg_loss = total_loss / n_batches
                avg_reward = total_reward / n_samples
                avg_q = total_q / n_batches

                loss_history.append(avg_loss)
                reward_history.append(avg_reward)
                q_value_history.append(avg_q)

                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, "
                      f"Avg Reward: {avg_reward:.4f}, Avg Q: {avg_q:.4f}")

        # plot figures
        plot_training_metrics(loss_history, reward_history, q_value_history, recommend_history)

        test_data = load_test_data()
        plot_episode_recommendation(policy_net, test_data)

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