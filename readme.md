# COMP532-CA1 Multi-Armed Bandit Problem
## UCB - Upper Confidence Bound
The UCB (Upper Confidence Bound) algorithm is a method that balances **exploration** and **exploitation** in the multi-armed bandit problem. The formula is as follows
$$A_t \doteq \underset{a}{\text{argmax}} \left[ Q_t(a)+c \sqrt{\frac{\log t}{N_t(a)}} \right]$$
where
- $\doteq$ means equality relationship that is true by definition
- $𝐴_𝑡$ is the action selected in time step $𝑡$
- $𝑄_𝑡(𝑎)$ is the estimated value at action $𝑎$
- $𝑁_𝑡(𝑎)$ is the number of times action $𝑎$ is selected up to time step $𝑡$
- $c$ is the exploration parameter, which determines the trade-off between exploration and exploitation

Core idea: Estimate an upper confidence bound for each action and select the action with the highest upper bound at each step. 
The higher the upper bound, the greater the potential of the action:
- its estimated value is high (exploit)
- relatively high uncertainty (explore)
