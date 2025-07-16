# 🎩 Model-Based Reinforcement Learning

See analysis of the MBRL [🔥here🔥](https://github.com/JeongHwaSik/cs285/blob/main/hw4/hw4.pdf).

## Experiment 1: MBRL

![Tag](https://img.shields.io/badge/Model_Based-skyblue)
![Tag](https://img.shields.io/badge/Off_Policy-red)
![Tag](https://img.shields.io/badge/Discrete_Action_Space-green)

[Policy Gradient](https://github.com/JeongHwaSik/cs285/blob/main/hw2/README.md) and [Q-Learning](https://github.com/JeongHwaSik/cs285/blob/main/hw3/README.md) (including DQN) algorithms aim to directly learn a policy or value function (such as a Q-function) that maximizes the expected cumulative reward in order to predict future actions. However, these methods require extensive interaction with the environment to gather large amounts of training data, which may not be practical in many real-world scenarios.

Model-Based Reinforcement Learning (MBRL) addresses this limitation by learning a transition model $f(s'|s, a)$ that approximates the true environment dynamics $p(s'|s, a)$ using only a limited number of interactions. Once learned, this model can be used to generate additional data reducing the need for direct environment interaction.


**❓Q1. Compare two action selection methods in MBRL, which are "derivative-free optimization": [CEM(Cross-Entropy Method)](https://arxiv.org/pdf/1909.11652) & [Random-Shooting](https://arxiv.org/pdf/1909.11652).**

<p align="center">
    <img src="https://github.com/user-attachments/assets/9e3534fe-a47b-4e1a-9010-9746e1770d4a" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

Derivative-free optimization refers to optimizing action sequences without relying on gradient computations using methods such as Cross-Entropy Method (CEM) or Random Shooting (which are action selection methods for this experiment). These approaches are widely used due to their robustness against model inaccuracies mentioned in this [paper](https://arxiv.org/pdf/1912.01603)

Since CEM is significantly slower than random shooting, I limited it to 5 iterations whereas random-shooting was run for 15 iterations.

</br>

## Experiment 2: [MBPO (Model-Based Policy Optimization)](https://arxiv.org/pdf/1906.08253)

![Tag](https://img.shields.io/badge/Model_Based-skyblue)
![Tag](https://img.shields.io/badge/Off_Policy-red)
![Tag](https://img.shields.io/badge/Discrete_Action_Space-green)
![Tag](https://img.shields.io/badge/Continuous_Action_Space-darkgreen)

Instead of using action selection methods (e.g., CEM or random shooting) as in experiment 1, we can use an explicit policy such as Soft Actor-Critic (SAC) and train it using both real-world transitions $p(s'|s,a)$ and transitions from a learned model $f(s'|s,a)$ which is sample efficient.

**❓Q1. Compare Model-Free SAC and Model-Based SAC. (Model-Based SAC leverages additional training data generated from a learned dynamics model.)**

<p align="center">
    <img src="https://github.com/user-attachments/assets/91189948-d9b1-4d16-a0da-99d9cde01326" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

As shown in the graph above, Model-Based SAC significantly outperforms Model-Free SAC (even when compared over just 5 iterations). This advantage arises from the Model-Based approach’s ability to generate diverse trajectories using learned transition dynamics $f(s'|s, a)$. These results suggest that Model-Based Reinforcement Learning (MBRL) can achieve comparable performance with significantly fewer real-world environment interactions, highlighting its superior sample efficiency.

**❓Q2. Compare Dyna-style MBPO (single step rollouts from the model) with 10-step MBPO (10-step rollouts from the model).**

<p align="center">
    <img src="https://github.com/user-attachments/assets/b0f85582-3d11-4f12-8af7-b9e4eadb50ee" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

In the Dyna-style or 1-step MBPO approach, the replay buffer is augmented with a single synthetic transition $(s, a, r, s')$ per model rollout to aid policy optimization. In contrast, the 10-step MBPO method adds a sequence of 10 such transitions creating longer synthetic trajectories. As shown in the graph above, 10-step MBPO achieves significantly better performance, nearly double that of the Dyna-style variant. The longer synthetic rollouts enable the agent to learn from richer and more temporally extended trajectories, improving the overall effectiveness of model-based planning. However, this also increases the risk of compounding model errors, which should be managed carefully.
