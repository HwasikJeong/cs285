#  🍎 Q-Learning

See analysis of the Q-Learning [🔥here🔥](https://github.com/JeongHwaSik/cs285/blob/main/hw3/hw3.pdf).

## Experiment 1: Deep Q-Learning (DQN)
```
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1

python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 2

python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 3
```

**Q1. Run DQN on `CartPole-v1` using different learning rates (0.05 and 0.001). How do the predicted Q-values and critic error change?**

<p align="center">
    <img src="https://github.com/user-attachments/assets/a3484f80-30fd-425c-bfd4-d9757a42a3ef" width="49%"/>
    <img src="https://github.com/user-attachments/assets/392da8ef-f230-4518-baf9-fff299c62e76" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

With a larger learning rate (0.05), the Q-values converge more quickly but the TD-error tends to overreact, making the function approximation unstable. On the other hand, with a smaller learning rate (0.001), the Q-values converge more slowly but the target and predicted Q-values stay closer together, resulting in better stability.

**Q2. Compare DQN Q-Values with True Q-Values.**

<p align="center">
    <img src="https://github.com/user-attachments/assets/ca038561-8cab-4adb-ba40-a9bdb45702fc" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

In the graph above, the dark blue horizontal line represents the true Q-value at the final step, while the light blue line shows the Q-values learned by DQN. As shown in the graph, the Q-values estimated by DQN are significantly higher than the true Q-value. This is because DQN estimates the Q-value using the equation $Q(s, a) \approx r(s, a) + \max\limits_{a'}Q(s', a')$, which includes a $\max$ operation. This $\max$ operation can lead to overestimation of Q-values when using function approximation, introducing significant bias. (This can be shown using Jensen’s Inequality — see [here](https://github.com/JeongHwaSik/cs285/blob/main/hw3/hw3.pdf) for proof.) To mitigate this overestimation bias, Double DQN can be used, which updates the critic using two separate networks.

</br>

## Experiment 2: Double DQN
```
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --
seed 1

python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --
seed 2

python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --
seed 3
```

**Q1. Compare "vanilla" DQN with Double DQN.**

<p align="center">
    <img src="https://github.com/user-attachments/assets/0824fd24-ed8c-4250-aacd-c5e3dec376e6" width="49%"/>
    <img src="https://github.com/user-attachments/assets/afb70f64-7dd4-4416-9526-48b48177c581" width="49%"/>
</p>

In Q-learning, bootstrapping involves computing $\max\limits_{a'}Q_{\phi'}(s',a')$ However, since $\arg\max\limits_{a'}Q_{\phi'}(s',a')$ contains a lot of noise, taking the maximum over noisy estimates tends to introduce an overestimation bias. This happens because the max operation selects the highest (and possibly overestimated) value. To address this, Double DQN uses two separate networks: one to select the next action and another to evaluate its Q-value. This helps decorrelate the noise between selection and evaluation, reducing overestimation. As shown in the graph above, the Q-values from Double DQN are noticeably lower compared to those from standard DQN, indicating reduced overestimation.

**Q2. Compare DQN with Policy Gradient.**

<p align="center">
    <img src="https://github.com/user-attachments/assets/afb70f64-7dd4-4416-9526-48b48177c581" width="49%"/>
    <img src="https://github.com/user-attachments/assets/594672cf-f1e9-4993-9d11-adaca3f30ce0" width="49%"/>
</p>

The graph on the left shows training results on `LunarLander-v2` using DQN (and Double DQN), while the graph on the right shows results from training on the same environment using a Policy Gradient method (specifically GAE). Comparing the two reinforcement learning algorithms, I can see that training with DQN is significantly more stable.

</br>

## Experiment 3: [Soft Actor-Critic (SAC)]((https://arxiv.org/pdf/1801.01290))
DQN performs well in environments with discrete action spaces because computing $\max\limits_{a}Q(s,a)$ is straightforward. However, in continuous action spaces, finding this maximum becomes a challenging optimization problem — often non-linear and non-convex. To address this, Actor-Critic methods use two separate networks: one to approximate the Q-function (similar to DQN) and another to represent the policy $\pi$ that is explicitly trained to maximize the expected Q-value, $E_{a\sim\pi(a|s)}Q(s,a)$. There are two main approaches to updating the policy $\pi$: the **REINFORCE gradient estimator** and the **REPARAMETRIZATION trick**.

**1️⃣ REINFORCE Gradient Estimator**

$$
\nabla_{\theta}E_{s\sim D, a\sim\pi(a|s)}[Q(s,a)] 
$$

$$
= \nabla_{\theta}\sum\limits_{s}p(s)\sum\limits_{a}\pi_{\theta}(a|s)Q(s,a) 
$$

$$
= \sum\limits_{s}p(s)\sum\limits_{a}\pi_{\theta}(a|s)\nabla_{\theta}log[\pi_{\theta}(a|s)]Q(s,a)
$$

$$
= E_{s\sim D, a\sim\pi(a|s)}[\nabla_{\theta}log[\pi_{\theta}(a|s)]Q(s,a)]
$$

```
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reinforce1.yaml

python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reinforce10.yaml
```

**Q1. Compare the performance of single-sample REINFORCE and 10-sample REINFORCE.**

REINFORCE works reasonably well when many samples are drawn from the policy $\pi$ to estimate the gradient. However, in high-dimensional action spaces, the variance of this estimator can become quite large, requiring significantly more samples to obtain stable and accurate updates. 

For example, single-sample REINFORCE estimates the policy gradient using just one action sampled from $\pi(a|s)$:

$$
\nabla_{\theta}J(\pi_{\theta})\approx\nabla_{\theta}log[⁡\pi_{\theta}(a|s)]Q(s,a)
$$

This estimator is unbiased but often suffers from high variance, especially in complex environments.

In contrast, 10-sample REINFORCE averages the gradient over 10 independently sampled actions:

$$
\nabla_{\theta}J(\pi_{\theta})\approx\frac{1}{10}\sum\limits_{i=1}^{10}\nabla_{\theta}log[⁡\pi_{\theta}(a_i|s)]Q(s,a_i)\;\;\;\;\; a_i\sim\pi(a|s)
$$

This reduces the variance of the gradient estimate and generally leads to more stable learning. However, it comes at the cost of increased computation per update. In high-dimensional spaces, such multi-sample averaging becomes more critical as single-sample estimates may not provide reliable signals for improving the policy.

<p align="center">
    <img src="https://github.com/user-attachments/assets/ef588457-e199-4ea0-8081-f7fe53f9f52b" width="49%"/>
</p>

In this experiment on `HalfCheetah-v4`, the action space is 6-dimensional. As a result, the single-sample REINFORCE actor update struggles to learn effectively. In contrast, using 10 samples for the REINFORCE update significantly improves performance indicating better gradient estimation in high-dimensional action spaces.

**2️⃣ REPARAMETRIZATION Trick**

$$
\nabla_{\theta}E_{s\sim D, a\sim\pi(a|s)}[Q(s,a)]
$$

$$
= \nabla_{\theta}\sum\limits_{s}p(s)\sum\limits_{a}\pi_{\theta}(a|s)Q(s,a)
$$

$$
 = \nabla_{\theta}\sum\limits_{s}p(s)\sum\limits_{\epsilon}N(\epsilon)Q(s,\mu+\sigma\epsilon) 
$$

$$
 = \nabla_{\theta}E_{s\sim D, \epsilon\sim N}[\nabla_{\theta}Q(s,\mu+\sigma\epsilon)]
$$

```
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reparametrize.yaml
```

**Q1. Compare single-sample REINFORCE and 10-sample REINFORCE with the REPARAMETERIZATION trick.**

<p align="center">
    <img src="https://github.com/user-attachments/assets/8a80d96f-4ae8-4a60-9d01-501a6146983c" width="49%"/>
</p>

The above graph shows that REPARAMETRIZATION-based actor update in SAC significantly outperforms both versions of the REINFORCE-based updates on `HalfCheetah-v4`. The reparameterized approach yields stable and consistent performance improvements, reaching much higher evaluation returns over time. In contrast, the REINFORCE method with 10 samples shows moderate learning but remains well below the reparameterized variant, while the single-sample REINFORCE fails to learn meaningful behavior. This highlights the advantage of reparameterization in providing lower-variance gradient estimates for high-dimensional continuous control tasks (in this case 6-dimension).

**Q2. Similar to DQN, using a single critic in Q-learning can lead to overestimation bias in target Q-values. To address this issue, several widely used strategies have been proposed: Double-Q, Clipped Double-Q, and [Randomized Ensembled Double-Q](https://arxiv.org/pdf/2101.05982). Discuss how these results relate to overestimation bias.**

<p align="center">
    <img src="https://github.com/user-attachments/assets/1df88f1d-057b-487e-933e-ed1804555264" width="49%"/>
    <img src="https://github.com/user-attachments/assets/1686baf4-6147-4d1e-ae1d-50d947d6bd67" width="49%"/>
</p>

The results show that all the tested methods help reduce overestimation bias to some extent. On the right graph, Q-values final point stays around 150–250, while the actual evaluation returns in the first graph reach 400–700, indicating that none of the methods significantly overestimate future returns. Among them, 🔴 Randomized Ensembled Double-Q and 🟢 Clipped Double-Q produce more conservative Q-value estimates, likely due to their use of `torch.min` when computing target values. Despite these lower estimates, both methods achieve competitive or even superior evaluation returns, showing that more cautious value estimates can still lead to effective policy learning. This underscores the importance of stabilizing target values rather than simply maximizing Q-values.