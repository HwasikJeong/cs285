# 🥷 Imitation Learning

See analysis of the Imitation Learning [🔥here🔥](https://github.com/JeongHwaSik/cs285/blob/main/hw1/hw1.pdf).

## Experiment 1: Behavioral Cloning

![Tag](https://img.shields.io/badge/Supervised_Learning-blue)

Behavior Cloning (BC) is a straightforward imitation learning approach that directly maps states to actions by mimicking an expert’s behavior. It learns a policy by regressing over state-action pairs collected from expert demonstrations much like supervised learning.

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1
```

**❓Q1. How closely does the BC agent's behavior match that of the expert policy?**

| Environment     | Expert Avg. Return | BC Agent Avg. Return  | closeness (%) |
|-----------------|-----------------------|--------------------------|--------|
| `Ant-v4`          | 4682                  | 1145                     | 24.46% |
| `HalfCheetah-v4`  | 4035                  | 3234                     | 80.15% |
| `Walker2d-v4`     | 5383                  | 356                      | 6.61%. |
| `Hopper-v4`       | 3718                  | 879                      | 23.64% |

The Behavior Cloning (BC) agent shows limited success in matching the expert policy across most environments. With the exception of `HalfCheetah-v4`, where the BC agent achieves approximately 80% of the expert's performance, all other environments fall below 30% closeness. In particular, performance in `Walker2d-v4` is notably poor, reaching only 6.61% of the expert's return.

This performance gap can be attributed to a key limitation of Behavior Cloning: it ignores the environment’s dynamics, i.e., the state transition distribution $p(s'|s, a)$ and focuses solely on imitating the expert’s action distribution $\pi(a|s)$. As a result, BC is vulnerable to compounding errors, once it deviates from expert-like behavior, it cannot recover effectively due to its lack of environmental awareness.

</br>

## Experiment 2: DAGGER

![Tag](https://img.shields.io/badge/Supervised_Learning-blue)

```
python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl --video_log_freq -1
```

**❓Q1. Compare DAGGER with Behavior Cloning (BC).**

| Environment     | DAGGER Average Return | BC Agent Average Return  | Improvement |
|-----------------|-----------------------|--------------------------|--------|
| `Ant-v4`          | 4791                  | 1145                     | ×4.18  |
| `HalfCheetah-v4`  | 4078                  | 3234                     | ×1.26  |
| `Walker2d-v4`     | 5368                  | 356                      | ×15.08 |
| `Hopper-v4`       | 3738                  | 879                      | ×4.25  |

DAGGER consistently outperforms Behavior Cloning (BC) across all environments, with improvements ranging from ×1.26 in `HalfCheetah-v4` to over ×15 in `Walker2d-v4`. The most dramatic gain is observed in `Walker2d-v4`, where DAGGER achieves an average return more than 15 times higher than BC.

This significant improvement stems from DAGGER's core advantage: unlike BC, which passively learns from a fixed set of expert demonstrations, DAGGER actively collects new data by querying the expert during the agent’s own rollouts. This allows it to correct for distributional shift and compounding errors by incorporating corrective actions in states that BC might never see during training. As a result, DAGGER better adapts to the actual state distribution induced by the learned policy, leading to more robust and effective performance.

</br>

## Experiment 3: Switched DAGGER

![Tag](https://img.shields.io/badge/Supervised_Learning-blue)

```
```
