# COMP532-CA1 Multi-Armed Bandit Problem
## UCB - Upper Confidence Bound
$$A_t \approx \underset{a}{\text{argmax}}
\Bigg[ Q_t(a) + c\sqrt{\frac{\ln{t}}{N_t(a)}} \Bigg]$$

UCB（Upper Confidence Bound）算法是一种在多臂赌博机问题中平衡探索和利用的方法。

核心思想: 为每个动作估计一个置信上界，并在每一步中选择具有最高上界的动作。

上界越高，意味着该动作的潜力越大，要么是因为它的估计价值高（利用），要么是因为相对不确定性高（探索）。

UCB算法的选择规则如上述公式所示，其中

$$
\begin{align*}
& 𝐴_𝑡 \text{ 是在时间步 }𝑡\text{ 选择的动作} \\
& 𝑄_𝑡(𝑎) \text{ 是动作 }𝑎\text{ 的估计价值} \\
& 𝑁_𝑡(𝑎) \text{ 是动作 }𝑎\text{ 至时间步 }𝑡\text{ 被选择的次数} \\
& 𝑐 \text{ 是探索参数，决定了探索和利用之间的权衡}
\end{align*}
$$