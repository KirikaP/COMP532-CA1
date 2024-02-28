import numpy as np
import matplotlib.pyplot as plt

from bandit_algos.k_arm_bandit import BanditMachine
from bandit_algos.ucb_bandit import UCBBandit

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


