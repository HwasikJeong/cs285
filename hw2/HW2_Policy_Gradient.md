#  👮🏼‍♂️ Policy Gradient

## Experiment 1: Reward-To-Go, Advantage Normalization 
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

**📈 Plot the learning curves (average return vs. number of environment steps)
for the experiments with small batch and large batch experiments.**

<img src="https://github.com/user-attachments/assets/69475738-56af-4bc5-a6f7-4ca14f29981f" />

<img src="https://github.com/user-attachments/assets/1dfae4c2-fc48-4fe3-a94d-4f49e27562a2" />

**Q1. Which value estimator has better performance without advantage normalization: the trajectory-centric one, or the one using reward-to-go?**

When the experiment was conducted without advantage normalization, the reward-to-go method (orange) converged to a reward of 200 faster than the trajectory-centric method (blue). Reward-to-go leverages the property of causality, meaning that the policy at time t’ cannot affect the reward at time t when t < t’. This method excludes rewards that occurred before time t when calculating the policy gradient and only includes rewards that occur afterward. As a result, it can reduce variance by removing noise when learning the policy.

**Q2. Did advantage normalization help?**

If advantage is used without normalization, the scale of the advantage can vary across different trajectory batches, causing large fluctuations during the policy gradient update process and potentially increasing variance. Additionally, due to these scale differences, extremely large policy updates may occur, reducing training stability. Therefore, by normalizing the advantage before applying it to the policy gradient, it is possible to prevent extremely large policy updates, align the scale of advantages across batches, reduce variance, and achieve relatively more stable learning and faster convergence.

**Q3. Did the batch size make an impact?**

Looking at the graph of the naive policy gradient (blue), we can see that the model using the larger batch size (lb) on the right converges faster at the same 50,000 environment steps. This is similar to the principle in deep learning where given the same total number of images, increasing the batch size leads to faster optimization. (Larger batch size ⇒ More samples in one iteration ⇒ Lower variance gradient estimates.)

</br>

## Experiment 2: PG with Baseline
```
# No baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah

# Baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
```

<img src="https://github.com/user-attachments/assets/f75f0aff-99aa-489f-a2d9-ded463772378"/>
<img src="https://github.com/user-attachments/assets/e7de5bd3-d548-49a9-b8c5-15ae34e215c6"/>

**Q1. How does baseline gradient steps (`-blg`) and baseline learning rate (`-blr`) affect the baseline learning curve and the performance of the policy?**

The larger the baseline gradient step, the more baseline updates occur within a single policy update iteration, leading to a better approximation of the baseline estimate. As a result, the policy update proceeds with a higher return. A baseline learning rate of 0.01 yields the optimal return value and this should be carefully tuned as a hyperparameter.