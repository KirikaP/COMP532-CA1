import numpy as np
import matplotlib.pyplot as plt


sigma = 1  # 10-armed bandit normal distribution std deviation
# 10-armed bandit normal distribution miu list
miu_list = [0.2, -0.85, 1.55, 0.3, 1.2, -1.5, -0.2, -1.0, 0.9, -0.6]

# 10-armed bandit
def action_reward(action):
    reward = miu_list[action] + np.random.normal(0, scale=sigma)
    return reward
