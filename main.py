import numpy as np
import matplotlib.pyplot as plt

from bandit_algos.k_arm_bandit import BanditMachine
from bandit_algos.ucb_bandit import UCBBandit

bandits = BanditMachine(10)
for i in range(10):
    print("arm: ", i, "miu: ", bandits.miu_list[i])


# # 初始化bandit的参数
# miu_list = [0.2, -0.85, 1.55, 0.3, 1.2, -1.5, -0.2, -1.0, 0.9, -0.6] # 每个臂的奖励期望值
# sigma = 1  # 正态分布的标准差

# # 实例化bandit_arm
# bandits = [BanditArm(miu, sigma) for miu in miu_list]

bandits = BanditMachine(10)
for i in range(10):
    print("arm: ", i, "miu: ", bandits.miu_list[i])


# 初始化 UCB 算法
c = 1.5
ucb_bandit = UCBBandit(bandits, c)

# 运行UCB算法
for _ in range(1000):
    chosen_arm, reward = ucb_bandit.play()
    print("chosen_arm: ", chosen_arm, "reward: ", reward)


