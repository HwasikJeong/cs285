# 🥷 Imitation Learning

Imitation Learning tries to **mimic expert behavior** by learning a policy that just replicates the actions observed in a dataset without understanding or optimizing long-term rewards. Since the objective is simply to imitate, there’s no need to define or optimize a reward function. It is just a supervised learning problem where the inputs are states and the labels are expert actions.

It’s common to confuse Offline Reinforcement Learning (Offline RL) with imitation learning because both rely on pre-collected data and do not involve active interaction with the environment (i.e., `env.step()`). However, they differ significantly in both goals and assumptions. Offline RL aims to learn a policy that maximizes cumulative reward even when the dataset contains suboptimal or exploratory behavior, while imitation learning assumes the data reflects expert decisions worth copying. See more details about Offline RL in a later section of [hw5](https://github.com/JeongHwaSik/cs285/tree/main/hw5#-offline-reinforcement-learning).

There are three primary limitations of imitation learning:

<span style="color:red">(1) Compounding errors (covariate shift):</span> Errors accumulate over time due to distribution mismatch between training and deployment. This issue can be mitigated by either collecting a large and diverse dataset or incorporating online corrective supervision, such as DAgger.

<span style="color:red">(2) Multimodal demonstration data:</span> When using an $L_2$ loss, the policy tends to regress toward the mean of multiple valid trajectories, leading to suboptimal behavior. This limitation can be addressed by modeling multimodality explicitly using approaches such as Gaussian mixture models, categorical distributions, variational autoencoders (VAEs), or diffusion-based methods.

<p align="center">
    <img src="[https://github.com/user-attachments/assets/51fc6d5e-594a-46bd-94fe-85a8bfa06b1f](https://github.com/user-attachments/assets/81acf5c5-3383-4ffc-afe4-7cb5e97f2435)" width="99%"/>
</p>

<span style="color:red">(3) Mismatch in observability between expert and agent:</span> The expert may have access to privileged information that is not available to the agent, making exact imitation infeasible.

## Experiment 1: Behavioral Cloning

Behavior Cloning (BC) is a straightforward imitation learning approach that directly maps states to actions by mimicking an expert’s behavior. It learns a policy by regressing over state-action pairs collected from expert demonstrations much like supervised learning.

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1
```

**❓Q1. How closely does the BC agent's behavior match that of the expert policy?**

<div align='center'>

| Environment     | Expert Avg. Return | BC Avg. Return  | closeness (%) |
|-----------------|-----------------------|--------------------------|--------|
| `Ant-v4`          | 4682                  | 1145                     | 24.46% |
| `HalfCheetah-v4`  | 4035                  | 3234                     | 80.15% |
| `Walker2d-v4`     | 5383                  | 356                      | 6.61%. |
| `Hopper-v4`       | 3718                  | 879                      | 23.64% |

</div>

The Behavior Cloning (BC) agent shows limited success in matching the expert policy across most environments. With the exception of `HalfCheetah-v4`, where the BC agent achieves approximately 80% of the expert's performance, all other environments fall below 30% closeness. In particular, performance in `Walker2d-v4` is notably poor, reaching only 6.61% of the expert's return.

This performance gap can be attributed to a key limitation of Behavior Cloning: it ignores the environment’s dynamics (i.e., the state transition distribution $p(s'|s, a)$) and focuses solely on imitating the expert’s action distribution $\pi(a|s)$. As a result, BC is vulnerable to compounding errors, once it deviates from expert-like behavior, it cannot recover effectively due to its lack of environmental awareness. See the the visualization of compounding error below.

<p align="center">
    <img src="https://github.com/user-attachments/assets/51fc6d5e-594a-46bd-94fe-85a8bfa06b1f" width="99%"/>
</p>

</br>

## Experiment 2: DAgger

**DA**gger (short for **D**ataset **A**ggregation) is an imitation learning algorithm that collects training data from the distribution induced by the learned policy denoted as $p_{\pi_{\theta}}(o_t)$ rather than from the distribution of expert demonstrations $p_{data}(o_t)$.

<p align="center">
    <img src="https://github.com/user-attachments/assets/67684690-d67d-481a-aa55-5f5f1ee2e3ee" width="99%"/>
</p>

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1
```

**❓Q1. Compare DAgger with Behavior Cloning (BC).**

<div align='center'>

| Environment     | DAgger Avg. Return | BC Avg. Return  | Improvement |
|-----------------|-----------------------|--------------------------|--------|
| `Ant-v4`          | 4791                  | 1145                     | ×4.18  |
| `HalfCheetah-v4`  | 4078                  | 3234                     | ×1.26  |
| `Walker2d-v4`     | 5368                  | 356                      | ×15.08 |
| `Hopper-v4`       | 3738                  | 879                      | ×4.25  |

</div>

DAgger consistently outperforms Behavior Cloning (BC) across all environments, with improvements ranging from ×1.26 in `HalfCheetah-v4` to over ×15 in `Walker2d-v4`. The most dramatic gain is observed in `Walker2d-v4`, where DAgger achieves an average return more than 15 times higher than BC.

This significant improvement stems from DAgger's core advantage: unlike BC, which passively learns from a fixed set of expert demonstrations, DAgger actively collects new data by querying the expert during the agent’s own rollouts. This allows it to correct for distributional shift and compounding errors by incorporating corrective actions in states that BC might never see during training. As a result, DAgger better adapts to the actual state distribution induced by the learned policy, leading to more robust and effective performance.

