# bandit_machine.py

import numpy as np

class BanditMachine:

    # initialize a bandit machine with n_arms arms
    def __init__(self, n_arms):
        self.n_arms = n_arms  # 赌博机的臂数
        # 初始化每个臂的期望奖励为不同的值，均值为0
        self.miu_list = np.random.normal(0, 1, n_arms)  
        self.sigma = 1  # 所有臂的奖励标准差为1
        self.counts = np.zeros(n_arms, dtype=int)  # 记录每个臂被拉动的次数
        self.values = np.zeros(n_arms)  # 记录每个臂的估计价值

    def play(self, arm):
        """拉动指定的臂"""
        if arm < 0 or arm >= self.n_arms:
            raise ValueError("Arm index is out of range.")
        
        # 生成指定臂的奖励
        reward = np.random.normal(self.miu_list[arm], self.sigma)
        # 更新被拉动臂的信息
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward
        return reward