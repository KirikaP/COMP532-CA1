# game_test.py is the main file to test the bandit machine

from bandit_machine import BanditMachine

# implement the main function
if __name__ == "__main__":
    n_arms = 5
    bandit = BanditMachine(n_arms)
    reward = 0
    for i in range(2000):
        arm = i % n_arms
        reward += bandit.play(arm)
    print(reward)
