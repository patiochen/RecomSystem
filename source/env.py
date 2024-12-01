import numpy as np
from typing import List, Tuple, Dict


class RecEnvironment:
    def __init__(self, state_action_rewards: List[Tuple], window_size: int = 5):
        self.window_size = window_size
        self.state_action_rewards = state_action_rewards
        self.actions = list(set([action for _, action, _ in self.state_action_rewards]))
        self.n_actions = len(self.actions)
        self.reset()

    def step(self, action: int) -> Tuple[float, bool]:
        if self.current_step >= len(self.state_action_rewards) - 1:
            return 0, True

        _, true_action, reward = self.state_action_rewards[self.current_step]

        # 修改reward机制
        if action == true_action:
            if reward == 1.0:  # 购买行为
                actual_reward = 5.0  # 大幅提高购买的奖励
            elif reward == 0.5:  # 加购行为
                actual_reward = 2.0  # 提高加购的奖励
            else:
                actual_reward = 0.0
        else:
            actual_reward = -0.1  # 添加小的负奖励

        self.current_step += 1
        done = self.current_step >= len(self.state_action_rewards) - 1

        return actual_reward, done

    def reset(self) -> List[int]:
        self.current_step = 0
        initial_state, _, _ = self.state_action_rewards[0]
        return initial_state

    def get_valid_actions(self) -> List[int]:
        history_items = set()
        start = max(0, self.current_step - 100)
        for i in range(start, self.current_step):
            _, action, _ = self.state_action_rewards[i]
            history_items.add(action)
        return list(history_items)

    def get_current_state(self) -> List[int]:
        state, _, _ = self.state_action_rewards[self.current_step]
        return state