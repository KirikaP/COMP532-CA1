import numpy as np

class EpsilonGreedy:
    def __init__(self, machines, epsilon):
        """
        :param machines: BanditMachine, 多臂赌博机的实例
        :param epsilon: float, 探索的概率
        """
        self.machines = machines
        self.epsilon = epsilon
        self.arms = machines.miu_list
        self.n_arms = len(self.arms)
        self.counts = np.zeros(self.n_arms, dtype=int)  # 记录每个动作被选择的次数
        self.values = np.zeros(self.n_arms, dtype=float)  # 记录每个动作的估计价值
        self.total_counts = 0  # 记录总共选择的次数

    def select_arm(self):
        # 先遍历一遍所有臂，如果有臂尚未被选择过，优先选择
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # 以概率 epsilon 随机选择一个臂，以概率 1 - epsilon 选择当前估计价值最高的臂
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        # 更新被选择动作的估计价值和被选择次数
        self.total_counts += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

    def play(self):
        # 选择动作，获得奖励，更新估计
        chosen_arm = self.select_arm()
        reward = self.machines.play(chosen_arm)
        self.update(chosen_arm, reward)
        return chosen_arm, reward
