# 🔎 Exploration & 💼 Offline Reinforcement Learning

See analysis of the Exploration & Offline RL [🔥here🔥](https://github.com/JeongHwaSik/cs285/blob/main/hw5/hw5.pdf).

## Experiment 1: 🔎 Exploration

<!-- ![Tag](https://img.shields.io/badge/Model_Based-skyblue)
![Tag](https://img.shields.io/badge/Off_Policy-red)
![Tag](https://img.shields.io/badge/Discrete_Action_Space-green) -->

A common strategy for exploration in reinforcement learning is to encourage the agent to visit states that are considered unlikely or novel. Various algorithms attempt to quantify this 'unlikeliness' to guide the agent's behavior. One such method is [Random Network Distillation (RND)](https://arxiv.org/pdf/1810.12894), which employs two neural networks: a fixed, randomly-initialized target network $f^*_\theta(s)$ and a learning network $\hat{f}_\phi(s)$. The idea is that the prediction error between these two networks serves as an 'unlikeliness', states that produce a high prediction error are deemed unfamiliar or unlikely. By prioritizing states with higher error, the agent is incentivized to explore less-visited regions of the state space thereby improving its overall exploration efficiency.


**❓Q1. Compare reinforcement learning with exploration versus without exploration. (I assumed that the random exploration is not an exploration algorithm.)**

<p align="center">
    <img src="https://github.com/user-attachments/assets/2dea7240-8a01-4075-b513-9acdeabcd619" width="33%"/>
    <img src="https://github.com/user-attachments/assets/8c2c2487-3872-452d-b605-4cbe0f777e61" width="33%"/>
    <img src="https://github.com/user-attachments/assets/94e92cc6-f868-4e1c-86b1-c21ba1470dfd" width="33%"/>
    <!-- <figcaption align="center">RL without exploration (random) in three environments of increasing difficulty: easy (left), medium (center), and hard (right).</figcaption> -->
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/1b86b40d-0ba2-4792-8061-fec7aea7a4e0" width="33%"/>
    <img src="https://github.com/user-attachments/assets/ee85874e-4445-47bc-bdd8-cde9812f0a41" width="33%"/>
    <img src="https://github.com/user-attachments/assets/3c5e0b26-e402-4945-8c3f-47ce9541efb4" width="33%"/>
    <!-- <figcaption align="center">RL with exploration (RND) in three environments of increasing difficulty: easy (left), medium (center), and hard (right).</figcaption> -->
</p>

The top row shows the results of reinforcement learning without exploration (i.e., using random action selection) across three environments of increasing difficulty: easy (left), medium (center), and hard (right). The bottom row displays the corresponding results when using an exploration-driven algorithm (RND) with an enhanced colormap to visualize the explored state space more clearly.

As shown, the agent is able to cover a much larger and more diverse region of the state space when guided by an exploration strategy. Moreover, even in the hard environment, the agent is able to approach the goal state more closely. This highlights the necessity of exploration in promoting deeper and more directed search, particularly in complex environments where random actions fail to provide sufficient coverage.

</br>

## Experiment 2: 💼 Offline RL

Offline RL is ~


**❓Q1. **

<p align="center">
    <img src="" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

blabla

