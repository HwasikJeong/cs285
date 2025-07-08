#  👮🏼‍♂️ Q-Learning

## Experiment 1: Deep Q-Learning (DQN)
```
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1

python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 2

python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 3
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/a3484f80-30fd-425c-bfd4-d9757a42a3ef" width="49%"/>
    <img src="https://github.com/user-attachments/assets/392da8ef-f230-4518-baf9-fff299c62e76" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

**Q1. Run DQN on `CartPole-v1` using different learning rates (0.05 and 0.001). How do the predicted Q-values and critic error change?**

With a larger learning rate (0.05), the Q-values converge more quickly but the TD-error tends to overreact, making the function approximation unstable. On the other hand, with a smaller learning rate (0.001), the Q-values converge more slowly but the target and predicted Q-values stay closer together, resulting in better stability.

**Q2. DQN over-estimation bias**

<p align="center">
    <img src="https://github.com/user-attachments/assets/ca038561-8cab-4adb-ba40-a9bdb45702fc" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

To estimate the Q-value, DQN use the max operation in the equation $Q(s, a) \approx r(s, a) + \max\limits_{a'}Q(s', a')$. However, this max operation can lead to over-estimation of Q-values when using function approximation, introducing significant bias. (This can be shown using Jensen’s Inequality — see [here](https://github.com/JeongHwaSik/cs285/blob/main/hw3/hw3.pdf) for proof.) To mitigate this over-estimation bias, Double DQN can be used, which updates the critic using two separate networks.

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

In Q-learning, bootstrapping involves computing $\max\limits_{a'}Q_{\phi'}(s',a')$ However, since $\arg\max\limits_{a'}Q_{\phi'}(s',a')$ contains a lot of noise, taking the maximum over noisy estimates tends to introduce an over-estimation bias. This happens because the max operation selects the highest (and possibly overestimated) value. To address this, Double DQN uses two separate networks: one to select the next action and another to evaluate its Q-value. This helps decorrelate the noise between selection and evaluation, reducing over-estimation. As shown in the graph above, the Q-values from Double DQN are noticeably lower compared to those from standard DQN, indicating reduced over-estimation.

**Q2. Compare DQN with Policy Gradient.**

<p align="center">
    <img src="https://github.com/user-attachments/assets/594672cf-f1e9-4993-9d11-adaca3f30ce0" width="49%"/>
    <img src="https://github.com/user-attachments/assets/afb70f64-7dd4-4416-9526-48b48177c581" width="49%"/>
</p>

The graph on the left shows training results on `LunarLander-v2` using DQN (and Double DQN), while the graph on the right shows results from training on the same environment using a Policy Gradient method (specifically GAE). Comparing the two reinforcement learning algorithms, I can see that training with DQN is significantly more stable.

</br>

## Experiment 3:
```

```

**Q1. **

<p align="center">
    <img src="" width="49%"/>
</p>

answers