import numpy as np
import matplotlib.pyplot as plt
from ucb import UCB
from epsilon_greedy import EpsilonGreedy
from k_arm_bandit import BanditMachine

bandit = BanditMachine(10)
'''
bandit.miu_list=[ 0.49730841 , 1.81020755 , 0.34373525 -0.85815164 , 2.74389601 -0.31067408,
  3.5822515 ,  1.48002517 , 0.39002063 , 0.70150586]
  '''
eps_01 = EpsilonGreedy(bandit,0.1)
eps_00 = EpsilonGreedy(bandit,0)
eps_001 = EpsilonGreedy(bandit,0.01)
ucb_2 = UCB(bandit,2)
ucb_1 = UCB(bandit,1)
ucb_05 = UCB(bandit,0.5)
optimal = np.argmax(bandit.miu_list)
step = 1000
runs = 2000

def test_algo(algo,runs,step):
    Q_runs_lst = [[] for m in range(runs)]
    Q_runs_lst_average = [0 for n in range(step)]
    action_runs_lst = [[] for k in range(runs)]  # 二维列表，存放所有runs的所有step的最佳动作判断变量，1是正确动作，0不是最佳动作
    action_runs_lst_average = [0 for j in range(step)]  # 一维列表，将 action_runs_lst 中的所有runs的对应step的action求平均
    for run_times in range(0, runs):
        k=1
        algo.counts = np.zeros(algo.n_arms, dtype=int)
        algo.values = np.zeros(algo.n_arms, dtype=float)
        algo.total_counts = 0
        while k<=step:
            reward=algo.play()[1]
            action=algo.play()[0]
            Q_runs_lst[run_times].append(reward)
            sum_best_action = 0
            if action==optimal:
                sum_best_action +=1
            action_runs_lst[run_times].append(sum_best_action)
            k += 1
    for i in range(0, step):
        for j in range(0, runs):
            Q_runs_lst_average[i] = Q_runs_lst_average[i] + Q_runs_lst[j][i]
            action_runs_lst_average[i] = action_runs_lst_average[i] + action_runs_lst[j][i]
        Q_runs_lst_average[i] = Q_runs_lst_average[i] / runs
        action_runs_lst_average[i] = action_runs_lst_average[i] / runs
    return Q_runs_lst_average,action_runs_lst_average

x = np.arange(step)
eps_01_reward,eps_01_rate = test_algo(eps_01,runs,step)
eps_00_reward,eps_00_rate = test_algo(eps_00,runs,step)
eps_001_reward,eps_001_rate = test_algo(eps_001,runs,step)
ucb_1_reward,ucb_1_rate = test_algo(ucb_1,runs,step)
ucb_05_reward,ucb_05_rate = test_algo(ucb_05,runs,step)
ucb_2_reward,ucb_2_rate = test_algo(ucb_2,runs,step)
plt.figure(1,figsize=(15,6))
plt.plot(x,ucb_2_reward,linewidth=0.5,color='r',label="ucb,c=2")
plt.plot(x,eps_01_reward,linewidth=0.5,label="epsilon_greedy,epsilon=0.1")
plt.plot(x,eps_001_reward,linewidth=0.5,color='g',label="epsilon_greedy,epsilon=0.01")
plt.plot(x,eps_00_reward,linewidth=0.5,color='orange',label="epsilon_greedy,epsilon=0")
plt.xlabel('steps')
plt.ylabel('average reward')
plt.legend(loc="best")
plt.figure(2,figsize=(15,6))
plt.plot(x,ucb_2_rate,linewidth=0.5,color='r',label="ucb,c=2")
plt.plot(x,eps_01_rate,linewidth=0.5,label="epsilon_greedy,epsilon=0.1")
plt.plot(x,eps_001_rate,linewidth=0.5,color='g',label="epsilon_greedy,epsilon=0.01")
plt.plot(x,eps_00_rate,linewidth=0.5,color='orange',label="epsilon_greedy,epsilon=0")
plt.xlabel('steps')
plt.ylabel('average optimal rate')
plt.legend(loc="best")
plt.figure(3,figsize=(15,6))
plt.plot(x,ucb_1_reward,linewidth=0.5,label="ucb,c=1")
plt.plot(x,ucb_2_reward,linewidth=0.5,color='r',label="ucb,c=2")
plt.plot(x,ucb_05_reward,linewidth=0.5,color='g',label="ucb,c=0.5")
plt.xlabel('steps')
plt.ylabel('average reward')
plt.legend(loc="best")
plt.figure(4,figsize=(15,6))
plt.plot(x,ucb_1_rate,linewidth=0.5,label="ucb,c=1")
plt.plot(x,ucb_2_rate,linewidth=0.5,color='r',label="ucb,c=2")
plt.plot(x,ucb_05_rate,linewidth=0.5,color='g',label="ucb,c=0.5")
plt.xlabel('steps')
plt.ylabel('average optimal rate')
plt.legend(loc="best")
plt.show()
