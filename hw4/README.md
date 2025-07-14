#  Model-Based Reinforcement Learning

See analysis of the MBRL [🔥here🔥](https://github.com/JeongHwaSik/cs285/blob/main/hw4/hw4.pdf).

## Experiment 1: MBRL

![Tag](https://img.shields.io/badge/Model_Based-skyblue)
![Tag](https://img.shields.io/badge/Off_Policy-red)
![Tag](https://img.shields.io/badge/Discrete_Action_Space-green)
![Tag](https://img.shields.io/badge/Continuous_Action_Space-darkgreen)

[Policy Gradient](https://github.com/JeongHwaSik/cs285/blob/main/hw2/README.md) and [Q-Learning](https://github.com/JeongHwaSik/cs285/blob/main/hw3/README.md) (including DQN) algorithms aim to directly learn a policy or value function (such as a Q-function) that maximizes the expected cumulative reward in order to predict future actions. However, these methods require extensive interaction with the environment to gather large amounts of training data, which may not be practical in many real-world scenarios.

Model-Based Reinforcement Learning (MBRL) addresses this limitation by learning a transition model $f(s'|s, a)$ that approximates the true environment dynamics $p(s'|s, a)$ using only a limited number of interactions. Once learned, this model can be used to generate additional data reducing the need for direct environment interaction.



**❓Q1. Compare two action selection methods in MBRL: [CEM(Cross-Entropy Method)](https://arxiv.org/pdf/1909.11652) & [Random-Shooting](https://arxiv.org/pdf/1909.11652).**

<p align="center">
    <img src="" width="49%"/>
    <img src="" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

Since CEM is significantly slower than random shooting, I limited it to 5 iterations whereas random-shooting was run for 15 iterations.

</br>

## Experiment 2: [MBPO (Model-Based Policy Optimization)](https://arxiv.org/pdf/1906.08253)

![](https://img.shields.io/badge/Model_Based-skyblue)
![Tag](https://img.shields.io/badge/Off_Policy-red)
![Tag](https://img.shields.io/badge/Discrete_Action_Space-green)
![Tag](https://img.shields.io/badge/Continuous_Action_Space-darkgreen)

Instead of using action selection methods (e.g., CEM or random shooting) as in experiment 1, we can use an explicit policy such as Soft Actor-Critic (SAC) and train it using both real-world transitions $p(s'|s,a)$ and transitions from a learned model $f(s'|s,a)$.

**❓Q1. Compare Model-Free SAC and Model-Based SAC. (Model-Based SAC leverages additional training data generated from a learned dynamics model.)**

<p align="center">
    <img src="" width="49%"/>
    <img src="" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

Since CEM is significantly slower than random shooting, I limited it to 5 iterations whereas random-shooting was run for 15 iterations.

**❓Q2. Compare Dyna-style MBPO (single step rollouts from the model) with 10-step MBPO (10-step rollouts from the model).**

<p align="center">
    <img src="" width="49%"/>
    <img src="" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>
