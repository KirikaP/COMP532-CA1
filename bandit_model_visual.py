import numpy as np
import matplotlib.pyplot as plt


def draw_one_arm(reward_dist):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes[0]
    ax.grid()
    ax.hist(reward_dist, bins=21)

    ax = axes[1]
    ax.grid()
    ax.violinplot(
        reward_dist,
        showmeans=True,
        quantiles=[0, 0.025, 0.25, 0.75, 0.925]
    )

    return fig, axes


def draw_mu(reward_mu):
    fig, ax = plt.subplots()
    ax.plot(reward_mu, 'ro--')
    ax.set_xlabel(u"10-arm")
    ax.set_ylabel(u"expected mean")
    return fig, ax


def draw_k_arm(k_reward_dist_mu, k_reward_dist_mu_sort):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax = axes[0]
    ax.grid()
    ax.violinplot(k_reward_dist_mu, showmeans=True)
    mean = np.round(np.mean(k_reward_dist_mu, axis=0), 3)
    for i in range(10):
        ax.text(i + 1 + 0.2, mean[i] - 0.1, str(mean[i]))

    ax = axes[1]
    ax.grid()
    ax.violinplot(k_reward_dist_mu_sort, showmeans=True)
    mean = np.round(np.mean(k_reward_dist_mu_sort, axis=0), 3)
    for i in range(10):
        ax.text(i + 1 + 0.2, mean[i] - 0.1, str(mean[i]))

    ax.set_title('sorted')

    return fig, axes


if __name__ == "__main__":

    num_arm = 10
    num_data = 2000
    # random seed, type a int value to fix results, or it'll be random
    # np.random.seed(5)
    # random arm's distribution
    k_reward_dist = np.random.randn(num_data, num_arm)
    print("raw mean=", np.round(np.mean(k_reward_dist, axis=0), 3))
    fig1, _ = draw_one_arm(k_reward_dist[:, 0])
    # 10 arms' mu
    reward_mu = np.random.randn(num_arm)
    print("expected average return=", np.round(reward_mu, 3))
    fig2, _ = draw_mu(reward_mu)
    # 10 arms' distribution in violin
    k_reward_dist_mu = reward_mu + k_reward_dist
    print("actual mean=", np.round(np.mean(k_reward_dist_mu, axis=0), 3))
    # sort fig
    reward_mu_sort_arg = np.argsort(reward_mu)
    k_reward_dist_mu_sort = np.zeros_like(k_reward_dist_mu)
    for i in range(10):
        idx = reward_mu_sort_arg[i]  # i-th arm for idx
        k_reward_dist_mu_sort[:, i] = k_reward_dist_mu[:, idx]

    fig3, _ = draw_k_arm(k_reward_dist_mu, k_reward_dist_mu_sort)

    plt.show()