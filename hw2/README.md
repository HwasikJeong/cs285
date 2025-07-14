#  👮🏼‍♂️ Policy Gradient

See analysis of the Policy Gradient [🔥here🔥](https://github.com/JeongHwaSik/cs285/blob/main/hw2/hw2.pdf).

## Experiment 1: Reward-To-Go, Advantage Normalization 

![Tag](https://img.shields.io/badge/Model_Free-blue)
![Tag](https://img.shields.io/badge/On_Policy-red)
![Tag](https://img.shields.io/badge/Discrete_Action_Space-green)
![Tag](https://img.shields.io/badge/Continuous_Action_Space-darkgreen)

```
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/69475738-56af-4bc5-a6f7-4ca14f29981f" width="49%"/>
    <img src="https://github.com/user-attachments/assets/1dfae4c2-fc48-4fe3-a94d-4f49e27562a2" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

**Q1. Which value estimator has better performance without advantage normalization: the trajectory-centric one, or the one using reward-to-go?**

When the experiment was conducted without advantage normalization, the reward-to-go method (orange) converged to a reward of 200 faster than the trajectory-centric method (blue). Reward-to-go leverages the property of causality, meaning that the policy at time t’ cannot affect the reward at time t when t < t’. This method excludes rewards that occurred before time t when calculating the policy gradient and only includes rewards that occur afterward. As a result, it can reduce variance by removing noise when learning the policy.

**Q2. Did advantage normalization help?**

If advantage is used without normalization, the scale of the advantage can vary across different trajectory batches, causing large fluctuations during the policy gradient update process and potentially increasing variance. Additionally, due to these scale differences, extremely large policy updates may occur, reducing training stability. Therefore, by normalizing the advantage before applying it to the policy gradient, it is possible to prevent extremely large policy updates, align the scale of advantages across batches, reduce variance, and achieve relatively more stable learning and faster convergence.

**Q3. Did the batch size make an impact?**

Looking at the graph of the naive policy gradient (blue), we can see that the model using the larger batch size (lb) on the lower side converges faster at the same 50,000 environment steps. This is similar to the principle in deep learning where given the same total number of images, increasing the batch size leads to faster optimization. (Larger batch size ⇒ More samples in one iteration ⇒ Lower variance gradient estimates.)

</br>

## Experiment 2: PG with Baseline

![Tag](https://img.shields.io/badge/Model_Free-blue)
![Tag](https://img.shields.io/badge/On_Policy-red)
![Tag](https://img.shields.io/badge/Discrete_Action_Space-green)
![Tag](https://img.shields.io/badge/Continuous_Action_Space-darkgreen)

```
# No baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah

# Baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/e7de5bd3-d548-49a9-b8c5-15ae34e215c6" width="49%"/>
    <img src="https://github.com/user-attachments/assets/f75f0aff-99aa-489f-a2d9-ded463772378" width="49%"/>
</p>

**Q1. How does baseline gradient steps (`-bgs`) and baseline learning rate (`-blr`) affect the baseline learning curve and the performance of the policy?**

<p align="center">
    <img src="https://github.com/user-attachments/assets/227462a3-f10d-44e0-a869-e7c7ac7c02e4" width="49%"/>
    <img src="https://github.com/user-attachments/assets/64593e9f-5c54-4d05-88df-13548f4567f3" width="49%"/>
</p>
<p align="center">
    <img src="https://github.com/user-attachments/assets/57e8c46d-ff19-43a1-b813-5b8ebb09c47d" width="49%"/>
    <img src="https://github.com/user-attachments/assets/b199ca08-0558-4d07-84ec-432938147917" width="49%"/>
</p>

The larger the baseline gradient step, the more baseline updates occur within a single policy update iteration, leading to a better approximation of the baseline estimate. As a result, the policy update proceeds with a higher return. A baseline learning rate of 0.01 yields the optimal return value and this should be carefully tuned as a hyperparameter.

</br>

## Experiment 3: Generalized Advantage Estimation (GAE)

![Tag](https://img.shields.io/badge/Model_Free-blue)
![Tag](https://img.shields.io/badge/On_Policy-red)
![Tag](https://img.shields.io/badge/Discrete_Action_Space-green)
![Tag](https://img.shields.io/badge/Continuous_Action_Space-darkgreen)

```
lambda=(0.0 0.95 0.98 0.99 1.0) \
&& \
for lamb in ${lambda[@]}; do python cs285/scripts/run_hw2.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda ${lamb} --exp_name lunar_lander_lambda${lamb}; done;
```

**Q1. Consider the parameter $\lambda$. What does $\lambda = 0$ correspond to? What about $\lambda = 1$? Relate this to the task performance in `LunarLander-v2`.**

<p align="center">
    <img src="https://github.com/user-attachments/assets/594672cf-f1e9-4993-9d11-adaca3f30ce0" width="49%"/>
</p>

$\lambda$ acts as a bias-variance tradeoff parameter. When $\lambda$ = 0, GAE reduces to the 1-step Temporal Difference (TD) error, resulting in high bias but low variance. On the other hand, when $\lambda$ = 1, it uses the full Monte Carlo advantage, which gives low bias but high variance. 

When $\lambda$ is 0, the learning relies only on the 1-step TD error, leading to too much bias and poor training performance. In contrast, when $\lambda$ is 1, it uses the full Monte Carlo return, achieving the lowest bias but suffering from very high variance, making the learning process noisy shown above learning curve. Therefore, due to this bias-variance tradeoff, a $\lambda$ value around 0.98–0.99 is considered a sweet spot.