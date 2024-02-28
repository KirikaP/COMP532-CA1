import numpy as np

class BanditArm:
    def __init__(self, miu, sigma):
        self.miu = miu  # 该臂的期望奖励
        self.sigma = sigma  # 该臂的奖励分布的标准差
        self.count = 0  # 被选择的次数
        self.value = 0  # 估计的价值

    def update(self, reward):
        """更新臂的价值估计"""
        self.count += 1
        # 使用增量公式更新平均值
        self.value = self.value + (1 / self.count) * (reward - self.value)

    def generate_reward(self):
        """生成该臂的奖励"""
        return np.random.normal(self.miu, self.sigma)

    def play(self):
        """拉动该臂"""
        reward = self.generate_reward()
        self.update(reward)
        return reward
