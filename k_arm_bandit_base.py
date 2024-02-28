import numpy as np

class BanditAlgorithm:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.arm_values = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)

    def select_arm(self):
        raise NotImplementedError("select_arm method must be implemented")

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.arm_values[arm] += (reward - self.arm_values[arm]) / self.arm_counts[arm]

class MyAlgorithm(BanditAlgorithm):
    def select_arm(self):
        # For demonstration, let's select a random arm
        return np.random.choice(self.num_arms)

# Example usage
num_arms = 5
num_iterations = 1000

bandit = MyAlgorithm(num_arms)

for _ in range(num_iterations):
    arm = bandit.select_arm()
    # In a real scenario, the reward would be obtained from some environment
    reward = np.random.normal(loc=0.0, scale=1.0)  # Placeholder reward function
    bandit.update(arm, reward)

# Print the estimated values of each arm
print("Estimated values of each arm:", bandit.arm_values)
