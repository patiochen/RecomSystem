import torch
from dqn import DQN

def load_test_data():
    """加载测试数据"""
    test_data = []
    with open('dataset/test_data.txt', 'r') as f:
        for line in f:
            try:
                # 解析状态和动作、奖励
                state_part, rest = line.split('],')
                state_str = state_part.strip('[')
                state = [int(x.strip().strip("'")) for x in state_str.split(',')]
                action, reward = rest.strip().split(',')
                action = int(action)
                reward = float(reward)
                test_data.append((state, action, reward))
            except Exception as e:
                print(f"Error parsing test data line: {line.strip()}")
                continue
    return test_data

def test_model():
    # 加载模型
    state_size = 5
    action_size = 8885
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load('recommender_model.pth'))
    model.eval()  # 设置为评估模式

    # 加载测试数据
    test_data = load_test_data()
    print(f"加载了 {len(test_data)} 条测试数据")

    # 评估指标
    correct = 0
    total_reward = 0

    # 对每个测试样本进行预测
    for state, true_action, true_reward in test_data:
        with torch.no_grad():
            # 获取模型预测
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            predicted_action = q_values.argmax().item()

            # 检查预测是否正确
            if predicted_action == true_action:
                correct += 1
                total_reward += true_reward

    # 计算准确率
    accuracy = correct / len(test_data)
    print(f"\n测试结果:")
    print(f"总测试样本数: {len(test_data)}")
    print(f"预测正确数: {correct}")
    print(f"准确率: {accuracy:.4f}")
    print(f"总奖励: {total_reward}")

if __name__ == "__main__":
    test_model()