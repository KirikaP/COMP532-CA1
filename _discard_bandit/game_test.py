# game_test.py is the main file to test the bandit machine

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bandit_machine import BanditMachine
from UCBStrategy import UCBStrategy


# implement the main function
if __name__ == "__main__":
    n_arms = 10  # the arms of the bandit machine
    bandit_machine = BanditMachine(n_arms)
    ucb = UCBStrategy(bandit_machine)

    ucb.run(2000)
    l1 = ucb.get_reward_step()
    s1 = pd.Series(l1)
    dict_rewards = {"Steps" : s1.index, "Reward" : s1.values}
    reward_df = pd.DataFrame(dict_rewards)
    reward_df["Average_Reward"] = reward_df["Reward"].expanding().mean()
    print(reward_df.head(10))
    reward_df.plot(x="Steps", y="Average_Reward")
    plt.show()
    
