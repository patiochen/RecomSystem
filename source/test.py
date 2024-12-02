import torch
import matplotlib.pyplot as plt
from dqn import DQN


def visualize_single_episode():
    # load trained model
    state_size = 5
    action_size = 8885
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load('recommender_model.pth'))
    model.eval()

    # load test dataset
    test_data = load_test_data()
    print(f"Loaded {len(test_data)} test samples")

    # show 200 recomendation results in one eps
    episode_length = 200  # show first 200 points

    # record rec products and its reward at each time
    recommended_items = []
    actual_items = []
    rewards = []

    # recommend products for test set
    with torch.no_grad():
        for state, true_action, reward in test_data[:episode_length]:
            # get recommendation
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            recommended_item = q_values.argmax().item()

            # record rec products, real products and reward
            recommended_items.append(recommended_item)
            actual_items.append(true_action)
            rewards.append(reward if recommended_item == true_action else 0)

    # plot figures
    plt.figure(figsize=(15, 8))

    # figures for time seq of rec products
    plt.subplot(2, 1, 1)
    plt.plot(range(len(recommended_items)), recommended_items, 'b-', label='Recommended Items', marker='o')
    plt.plot(range(len(actual_items)), actual_items, 'r--', label='Actual Items', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Item ID')
    plt.title('Recommendations vs Actual Items over Time')
    plt.legend()
    plt.grid(True)

    # reward plots
    plt.subplot(2, 1, 2)
    plt.plot(range(len(rewards)), rewards, 'g-', label='Reward', marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title('Rewards over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('episode_timeline.png')
    print("Episode timeline saved as 'episode_timeline.png'")
    plt.close()


def load_test_data():
    """load test dataset"""
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


if __name__ == "__main__":
    visualize_single_episode()