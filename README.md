Assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

# 1. Behavioral Cloning & DAGGER
The behavioral cloning agent achieved approximately 25% of the expert's performance, whereas the DAGGER agent achieved over 99%. The difference is clearly visible in the video.
For a more detailed view, run `tensorboard --logdir data` within the `hw1` directory.

https://github.com/user-attachments/assets/2f44c5fb-f61d-4416-ab5d-ed811b7a23f8

https://github.com/user-attachments/assets/484bcba9-099e-4f83-be3a-d6427c185cf2

</br>

# 2. Policy Gradient with Baseline & GAE
I implemented Policy Gradient with a Baseline, incorporating both Reward-to-Go and Advantage Normalization, as well as an advanced policy gradient method called Generalized Advantage Estimation (GAE). These approaches were used to train policies for controlling the HalfCheetah-v4 and LunarLander-v2 environments, respectively. More implementation details can be found [🔥here🔥](https://github.com/JeongHwaSik/cs285/tree/main/hw2).

https://github.com/user-attachments/assets/7c07762d-5ab9-4def-9cba-a3edd7cd5d4d

https://github.com/user-attachments/assets/804d5d89-1d33-4396-ab9b-babf095c8cbe
