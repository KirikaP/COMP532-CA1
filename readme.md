# COMP532-CA1 Multi-Armed Bandit Problem
## UCB - Upper Confidence Bound
$$A_t \approx \underset{a}{\text{argmax}}
\Bigg[ Q_t(a) + c\sqrt{\frac{\ln{t}}{N_t(a)}} \Bigg]$$