Assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

# 1. Behavioral Cloning
The behavioral cloning agent achieved approximately 25% of the expert's performance, whereas the DAGGER agent achieved over 99%. The difference is clearly visible in the video.
For a more detailed view, run `tensorboard --logdir data` within the `hw1` directory.

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

https://github.com/user-attachments/assets/42ed7cc3-8174-43a0-a366-94b8915f7874
