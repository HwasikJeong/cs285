Assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

# Table of Contents

### 1. [🥷 Imitation Learning](https://github.com/JeongHwaSik/cs285/tree/main/hw1#-imitation-learning)
### 2. [👮🏼‍♂️ Policy Gradients](https://github.com/JeongHwaSik/cs285/tree/main/hw2#%EF%B8%8F-policy-gradients)
### 3. [🍎 Q-Learning (DQN)](https://github.com/JeongHwaSik/cs285/tree/main/hw3#-q-learning)
### 4. [🎩 Model-Based Reinforcement Learning](https://github.com/JeongHwaSik/cs285/tree/main/hw4#-model-based-reinforcement-learning)
### 5. [🔎 Exploration](https://github.com/JeongHwaSik/cs285/tree/main/hw5#-exploration)
### 6. [💼 Offline Reinforcement Learning](https://github.com/JeongHwaSik/cs285/tree/main/hw5#-offline-reinforcement-learning)

</br>

# Demo Videos

### • DIAYN

<table>
  <tr>
    <td align="center">
      <figure>
        <img src="https://github.com/user-attachments/assets/0450ca4f-4e26-4ab8-9bec-70eed97c0b6b" width="90%"/> 
        <figcaption>Walking Backward</figcaption>
      </figure>
    </td>
    <td align="center">
      <figure>
        <img src="https://github.com/user-attachments/assets/74f0014d-cceb-4891-b115-d319ef9a44f8" width="92%" />
        <figcaption>Walking Forward</figcaption>
      </figure>
    </td>
    <td align="center">
      <figure>
        <img src="https://github.com/user-attachments/assets/7d9b2545-9927-40c6-ae0d-51b96ea0a6ac" width="99%" />
        <figcaption>Stand Still</figcaption>
      </figure>
    </td>
    <td align="center">
      <figure>
        <img src="https://github.com/user-attachments/assets/1e62e8e9-70f1-4905-8172-a384883435ff" width="90%" />
        <figcaption>Front Leg Standing</figcaption>
      </figure>
    </td>
    <td align="center">
      <figure>
        <img src="https://github.com/user-attachments/assets/f04765b1-3bb0-409c-8021-28c8b6cbb42c" width="93%" />
        <figcaption>Half Flipping</figcaption>
      </figure>
    </td>
  </tr>
</table>

### • Behavior Cloning (BC)

https://github.com/user-attachments/assets/2f44c5fb-f61d-4416-ab5d-ed811b7a23f8

### • DAgger

https://github.com/user-attachments/assets/484bcba9-099e-4f83-be3a-d6427c185cf2

### • Policy Gradients (PG)

https://github.com/user-attachments/assets/7c07762d-5ab9-4def-9cba-a3edd7cd5d4d

### • Generalized Advantage Estimation (GAE)

https://github.com/user-attachments/assets/804d5d89-1d33-4396-ab9b-babf095c8cbe

### • Soft Actor-Critic (SAC)

https://github.com/user-attachments/assets/42ed7cc3-8174-43a0-a366-94b8915f7874

<!-- # 1. 🥷 Imitation Learning
I implemented **Behavior Cloning (BC)**, which directly imitates the expert’s behavior, and **DAGGER**, which improves imitation by incorporating additional data collected through expert queries during the agent’s own rollouts.

https://github.com/user-attachments/assets/2f44c5fb-f61d-4416-ab5d-ed811b7a23f8

https://github.com/user-attachments/assets/484bcba9-099e-4f83-be3a-d6427c185cf2

</br>

# 2. 👮🏼‍♂️ Policy Gradient
I implemented **Policy Gradient with a Baseline**, incorporating both reward-to-go and advantage normalization, as well as an advanced policy gradient method called **Generalized Advantage Estimation (GAE)**. These approaches were used to train policy for controlling the environments. More implementation details and analysis can be found [🔥here🔥](https://github.com/JeongHwaSik/cs285/tree/main/hw2).

https://github.com/user-attachments/assets/7c07762d-5ab9-4def-9cba-a3edd7cd5d4d

https://github.com/user-attachments/assets/804d5d89-1d33-4396-ab9b-babf095c8cbe

</br>

# 3. 🍎 Deep Q-Learning (DQN)
I implemented **basic DQN** and **Double DQN** for discrete action spaces, and **Soft Actor-Critic (SAC)** for continuous action spaces. These Q-learning algorithms involve many implementation details so refer to [🔥this page🔥](https://github.com/JeongHwaSik/cs285/tree/main/hw3) for a more in-depth analysis.

The video below shows the results of training SAC — with the reparameterization trick used for actor updates — for just 1M iterations. Try 5M iterations afterwards.

https://github.com/user-attachments/assets/42ed7cc3-8174-43a0-a366-94b8915f7874

</br>

# 4. 🎩 Model-Based Reinforcement Learning
I implemented a basic **Model-Based Reinforcement Learning (MBRL)** framework using two action selection strategies: the Cross-Entropy Method (CEM) and Random Shooting. While Monte Carlo Tree Search (MCTS) could also be used for action selection, it was not applied in this case. Additionally, I extended the implementation to include Model-Based Soft Actor-Critic (SAC), which is one of **Model-Based Policy Optimization (MBPO)**, that leverages learned transition dynamics unlike the standard Model-Free SAC (in hw3) that relies solely on real-world interactions.


# 5. 🔎 Exploration


# 6. 💼 Offline Reinforcement Learning -->