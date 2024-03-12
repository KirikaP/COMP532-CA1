import numpy as np

class BanditMachine:
    def __init__(self, n_arms):
        self.n_arms = n_arms  # 赌博机的臂数
        
        self.sigma = 2  # 所有臂的奖励标准差为1
        self.miu_list = np.random.normal(0, self.sigma, n_arms)  # 初始化每个臂的期望奖励为不同的值，均值为0
        self.counts = np.zeros(n_arms, dtype=int)  # 记录每个臂被拉动的次数
        self.values = np.zeros(n_arms)  # 记录每个臂的估计价值
        
    def update(self, arm, reward):
        """
        更新被拉动臂的信息
        :param arm: int, 臂的索引
        :param reward: float, 奖励
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward # 增量更新估计价值

    def play(self, arm):
        """
        拉动指定的臂
        :param arm: int, 臂的索引
        :return: float, 奖励
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError("Arm index is out of range.")
        
        # 生成指定臂的奖励
        reward = np.random.normal(self.miu_list[arm], self.sigma)
        # 更新被拉动臂的信息
        self.update(arm, reward)
        return reward
