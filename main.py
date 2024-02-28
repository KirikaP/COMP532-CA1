import numpy as np
import matplotlib.pyplot as plt

from k_arm_bandit import BanditArm
from bandit_algos.ucb_bandit import UCBBandit

# 初始化bandit的参数
miu_list = [0.2, -0.85, 1.55, 0.3, 1.2, -1.5, -0.2, -1.0, 0.9, -0.6] # 每个臂的奖励期望值
sigma = 1  # 正态分布的标准差

# 实例化bandit_arm
bandits = [BanditArm(miu, sigma) for miu in miu_list]
    
# 初始化 UCB 算法
c = 1.5
ucb_bandit = UCBBandit(bandits, c)

# 运行UCB算法
for _ in range(1000):
    chosen_arm, reward = ucb_bandit.play()
    print("chosen_arm: ", chosen_arm, "reward: ", reward)



# # 10-armed bandit
# def action_reward(action):
#     reward = miu_list[action] + np.random.normal(0, scale=sigma)
#     # print("action: ", action, "reward: ", reward)
#     return reward
