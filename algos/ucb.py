import numpy as np



class UCB:
    def __init__(self, machines, c):
        """
        :param arms: list, 每个元素是一个BanditArm对象
        :param c: float, 探索因子
        """
        self.machines = machines
        self.c = c 
        self.arms = machines.miu_list # 赌博机的臂
        self.n_arms = len(self.arms) # 臂数
        self.counts = np.zeros(self.n_arms, dtype=int)  # 记录每个动作被选择的次数
        self.values = np.zeros(self.n_arms, dtype=float)  # 记录每个动作的估计价值
        self.total_counts = 0 # 记录总共选择的次数

    def select_arm(self):
        # 如果有臂尚未被选择过，优先选择
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        
        
        # 选择ucb值最大的动作
        ucb_values = self.values + self.c * np.sqrt(np.log(self.total_counts) / (self.counts + 1))
        return np.argmax(ucb_values)

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
