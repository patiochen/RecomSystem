import random
import torch
import numpy as np

class RecommenderEnvironment:
    def __init__(self, training_data, state_length=10):
        self.training_data = training_data
        self.state_length = state_length
        self.current_index = 0
        self.current_user_data = None
        self.reset()

    def reset(self):
        """
        Reset the environment for a new user session.
        """
        # 随机选择一个用户的数据
        self.current_user_data = random.choice(self.training_data)
        self.current_index = 0
        # 返回初始状态
        initial_state, _, _ = self.current_user_data
        return initial_state

    def step(self, action):
        """
        Take an action in the environment.
        :param action: The recommended item.
        :return: next_state, reward, done
        """
        _, actual_action, reward = self.current_user_data
        # 奖励是根据实际数据判断推荐的商品是否匹配
        reward = reward if action == actual_action else 0

        self.current_index += 1
        done = self.current_index >= len(self.training_data)
        if done:
            next_state = None
        else:
            next_state, _, _ = self.current_user_data

        return next_state, reward, done

# 加载预处理后的训练数据
training_data = torch.load('dataset/preprocessed_training_data_train.pt')

# 创建环境实例
env = RecommenderEnvironment(training_data)
