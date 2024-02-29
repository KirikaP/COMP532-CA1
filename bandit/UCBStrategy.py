# UCBSrategy.py

import math
import numpy as np


class UCBStrategy:
    def __init__(self, bandit_machine):
        self.bandit = bandit_machine
        self.counts = [0] * self.bandit.n_arms  # 每个臂被拉动的次数
        self.values = [0.0] * self.bandit.n_arms  # 每个臂的估计值
        self.total_counts = 0  # 总的尝试次数
        self.reward_step = []

    def select_arm(self):
        # 如果有臂还未被尝试过，则先尝试这些臂
        for arm in range(self.bandit.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0] * self.bandit.n_arms
        for arm in range(self.bandit.n_arms):
            bonus = math.sqrt((2 * math.log(self.total_counts)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus

        return ucb_values.index(max(ucb_values))

    def update(self, chosen_arm, reward):
        # 更新选中臂的信息
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value
        self.reward_step.append(reward)

    def run(self, num_plays):
        for _ in range(num_plays):
            chosen_arm = self.select_arm()
            reward = self.bandit.play(chosen_arm)
            self.update(chosen_arm, reward)

    def get_reward_step(self):
        return self.reward_step