import numpy as np


class RecommendEnv:
    def __init__(self):
        # 初始化环境
        self.state_size = 5
        self.state = [0] * self.state_size
        self.data = []
        self.current_step = 0
        self._load_data()

    def _load_data(self):
        # 只加载训练数据
        with open('dataset/train_data.txt', 'r') as f:
            self.data = f.readlines()

    def reset(self):
        # 重置环境
        self.state = [0] * self.state_size
        self.current_step = 0
        return self.state

    def step(self, action):
        if self.current_step >= len(self.data):
            return self.state, 0, True, {}

        # 获取真实的动作和奖励
        line = self.data[self.current_step].strip()
        true_action = int(line.split(',')[-2])
        reward = float(line.split(',')[-1])

        # 更新状态和奖励
        if action == true_action:
            obtained_reward = reward
            self.state = self.state[1:] + [action]
        else:
            obtained_reward = 0

        self.current_step += 1
        done = (self.current_step >= len(self.data))

        return self.state, obtained_reward, done, {}


if __name__ == "__main__":
    env = RecommendEnv()
    state = env.reset()
    print(f"Initial state: {state}")

    action = 1  # 测试动作
    next_state, reward, done, _ = env.step(action)

    print(f"Action: {action}")
    print(f"Next state: {next_state}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")