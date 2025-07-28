# 🎩 Model-Based Reinforcement Learning

[Policy Gradient](https://github.com/JeongHwaSik/cs285/tree/main/hw2#%EF%B8%8F-policy-gradients) and [Q-Learning](https://github.com/JeongHwaSik/cs285/tree/main/hw3#-q-learning) (including DQN) algorithms aim to directly learn a policy or value function (such as a Q-function) that maximizes the expected cumulative reward in order to predict future actions. However, these Model-Free methods require extensive interaction with the environment (e.g., `env.step()`) to gather large amounts of training data for planning, which may not be practical in many real-world scenarios. (But if large amounts of data can be collected, Model-Free methods show promise as a general-purpose tools.)

Model-Based Reinforcement Learning (MBRL) addresses this limitation by learning a transition model $f(s'|s, a)$ that approximates the true environment dynamics $p(s'|s, a)$ using only a limited number of interactions. Once learned, this model can be used to generate additional trajectories for planning, reducing the need for direct environment interaction, which is sample efficient. (You will see this MBRL algorithm in [Experiment 1](https://github.com/JeongHwaSik/cs285/tree/main/hw4#experiment-1-mbrl-with-mpc)) However, model accuracy can act as a bottleneck to policy quality especially when search space is huge. Moreover, **during planning**, relying on the real-world reward function (e.g., `env.get_reward()`) is still necessary, which limits practicality in many real-world applications.

An extended variant of MBRL also learns a reward model, enabling the agent to predict both the next state and the expected reward for any given state-action pair. This approach allows the agent to perform **imagination-based planning**, where it internally simulates potential trajectories and their outcomes entirely within the learned models.

This imagination-based MBRL is particularly effective in many challenging domains, such as environments with high-dimensional action spaces, long-horizon sequential tasks, or sparse rewards. By leveraging learned models to simulate diverse scenarios, the agent can explore and refine its strategy more efficiently than relying solely on real-world interactions. 

But long-horizon tasks still remain a significant challenge in reinforcement learning because errors are keep cumulating. To address this, researchers have developed **skill-based** reinforcement learning, where a skill typically refers to a temporally extended sequence of actions. By planning over these higher-level skills instead of individual low-level actions, agents can make more efficient and coherent decisions. This abstraction allows planning to be both faster and more reliable, especially in complex environments where long-term dependencies are important. For more details on skill-based reinforcement learning, refer to [SkiMo](https://arxiv.org/pdf/2207.07560).

## Experiment 1: MBRL with MPC

![Tag](https://img.shields.io/badge/Model_Based-skyblue)
![Tag](https://img.shields.io/badge/Off_Policy-red)
![Tag](https://img.shields.io/badge/Discrete_Action_Space-green)

In this experiment, I implemented a Model-Based Reinforcement Learning (MBRL) with Model Predictive Control (MPC). The approach involves explicitly learning a dynamics model $f(s'|s, a)$ to approximate environment transitions $p(s'|s, a)$, which is then used for planning actions using MPC strategies such as the [Cross-Entropy Method (CEM)](https://arxiv.org/pdf/1909.11652), [Random Shooting](https://arxiv.org/pdf/1909.11652), or Monte Carlo Tree Search (MCTS). 

However, the problem is that planning requires access to the real-world environment in order to obtain rewards (e.g., `env.get_reward()`). Additionally, since MPC plans over a finite (limited) horizon $H$, the agent may overlook delayed rewards or long-term dependencies, making it potentially shortsighted. 😢

For a more detailed analysis and implementation, see this [page](https://github.com/JeongHwaSik/cs285/blob/main/hw4/hw4.pdf).


**❓Q1. Compare two action selection methods in MBRL, which are "derivative-free optimization": CEM(Cross-Entropy Method) & Random-Shooting.**

<p align="center">
    <img src="https://github.com/user-attachments/assets/9e3534fe-a47b-4e1a-9010-9746e1770d4a" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

Derivative-free optimization refers to optimizing action sequences without relying on gradient computations using methods such as Cross-Entropy Method (CEM) or Random Shooting (which are action selection methods for this experiment). These approaches are widely used due to their robustness against model inaccuracies mentioned in this [paper](https://arxiv.org/pdf/1912.01603)

<!-- Since CEM is significantly slower than random shooting, I limited it to 5 iterations whereas random-shooting was run for 15 iterations. -->

</br>

## Experiment 2: MBPO (Model-Based Policy Optimization)

![Tag](https://img.shields.io/badge/Model_Based-skyblue)
![Tag](https://img.shields.io/badge/Off_Policy-red)
![Tag](https://img.shields.io/badge/Discrete_Action_Space-green)
![Tag](https://img.shields.io/badge/Continuous_Action_Space-darkgreen)

Instead of using action selection methods as in Experiment 1, we can use an explicit policy such as Soft Actor-Critic (SAC) and train it using both real-world transitions $p(s'|s,a)$ and transitions from a learned model $f(s'|s,a)$ which is sample efficient. However, a significant discrepancy can exist between the real environment and the learned model, potentially leading to inaccurate policy learning. To address this issue, the [Model-Based Policy Optimization (MBPO)](https://arxiv.org/pdf/1906.08253) paper proposes using short model-generated rollouts. This approach allows for more reliable and efficient policy optimization.

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