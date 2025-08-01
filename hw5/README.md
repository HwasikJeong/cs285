# 🔎 Exploration

In reinforcement learning (RL), it is unrealistic to expect an agent to operate solely within previously encountered states. This is where exploration becomes essential. It allows the agent to discover new states and learn to handle diverse situations. As discussed earlier, standard approaches like [policy gradients](https://github.com/JeongHwaSik/cs285/tree/main/hw2#%EF%B8%8F-policy-gradients) and [Q-learning](https://github.com/JeongHwaSik/cs285/tree/main/hw3#-q-learning) often rely on simple exploration strategies such as $\epsilon$-greedy or random action selection. However, these dithering methods typically require an exponentially large amount of data to achieve adequate exploration, making them inefficient in complex environments.

To address the limitations of simple, heuristic exploration strategies, researchers have developed more efficient and novelty-seeking exploration algorithms. These approaches generally fall into three broad categories. **Count-based exploration** methods, such as [Random Network Distillation (RND)](https://arxiv.org/pdf/1810.12894) and [EX2](https://arxiv.org/pdf/1703.01260), reward the agent for visiting less frequent or novel states. **Posterior sampling** approaches, like [Bootstrapped DQN](https://arxiv.org/pdf/1602.04621), encourage exploration by leveraging uncertainty in the agent’s value estimates. Lastly, **information gain-based** methods, such as [VIME](https://arxiv.org/pdf/1605.09674) and [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1801.01290), promote exploration by maximizing the agent’s information about the environment dynamics. In Experiment 1, I will implement RND as a representative of the count-based exploration family.

Another area closely related to exploration is **Unsupervised Skill Discovery**, which focuses on learning diverse and distinguishable behaviors in the latent space without relying on external rewards. These methods typically use information-theoretic objectives as a pseudo-reward. Notable examples include [DIAYN](https://arxiv.org/pdf/1802.06070), [LSD](https://arxiv.org/pdf/2202.00914), and [CSD](https://arxiv.org/pdf/2302.05103). In Experiment 2, I will implement DIAYN, which uses mutual information to encourage the agent to discover a diverse set of skills.

Note: A detailed analysis of exploration can be found [🔥here🔥](https://github.com/JeongHwaSik/cs285/blob/main/hw5/hw5.pdf).

## Experiment 1: Random Network Distillation (RND)

<!-- ![Tag](https://img.shields.io/badge/Model_Based-skyblue)
![Tag](https://img.shields.io/badge/Off_Policy-red)
![Tag](https://img.shields.io/badge/Discrete_Action_Space-green) -->

A common strategy for exploration in reinforcement learning is to encourage the agent to visit states that are considered unlikely or novel. Various algorithms attempt to quantify this 'unlikeliness' to guide the agent's behavior. One such method is [Random Network Distillation (RND)](https://arxiv.org/pdf/1810.12894), which employs two neural networks: a fixed, randomly-initialized target network ![formula](https://latex.codecogs.com/png.image?\dpi{110}f^*_\theta(s)) and a learning network ![f_hat_phi](https://latex.codecogs.com/png.image?\dpi{110}\hat{f}_\phi(s)). The idea is that the prediction error between these two networks serves as an 'unlikeliness', states that produce a high prediction error are deemed unfamiliar or unlikely. By prioritizing states with higher error, the agent is incentivized to explore less-visited regions of the state space thereby improving its overall exploration efficiency.


**❓Q1. Compare reinforcement learning with exploration versus without exploration. (I assumed that the random exploration is not an exploration algorithm.)**

<p align="center">
    <img src="https://github.com/user-attachments/assets/2dea7240-8a01-4075-b513-9acdeabcd619" width="33%"/>
    <img src="https://github.com/user-attachments/assets/8c2c2487-3872-452d-b605-4cbe0f777e61" width="33%"/>
    <img src="https://github.com/user-attachments/assets/94e92cc6-f868-4e1c-86b1-c21ba1470dfd" width="33%"/>
    <!-- <figcaption align="center">RL without exploration (random) in three environments of increasing difficulty: easy (left), medium (center), and hard (right).</figcaption> -->
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/1b86b40d-0ba2-4792-8061-fec7aea7a4e0" width="33%"/>
    <img src="https://github.com/user-attachments/assets/ee85874e-4445-47bc-bdd8-cde9812f0a41" width="33%"/>
    <img src="https://github.com/user-attachments/assets/3c5e0b26-e402-4945-8c3f-47ce9541efb4" width="33%"/>
    <!-- <figcaption align="center">RL with exploration (RND) in three environments of increasing difficulty: easy (left), medium (center), and hard (right).</figcaption> -->
</p>

The top row shows the results of reinforcement learning without exploration (i.e., using random action selection) across three environments of increasing difficulty: easy (left), medium (center), and hard (right). The bottom row displays the corresponding results when using an exploration-driven algorithm (RND) with an enhanced colormap to visualize the explored state space more clearly.

As shown, the agent is able to cover a much larger and more diverse region of the state space when guided by an exploration strategy. Moreover, even in the hard environment, the agent is able to approach the goal state more closely. This highlights the necessity of exploration in promoting deeper and more directed search, particularly in complex environments where random actions fail to provide sufficient coverage.

## Experiment 2: Skill Discovery

The [“Diversity is All You Need” (DIAYN)](https://arxiv.org/pdf/1802.06070) paper introduces a novel paradigm for exploring the state space without relying on external reward signals (unsupervised manner). Instead of learning from environment provided rewards, DIAYN encourages the agent to learn a set of diverse behaviors by conditioning its policy on a latent variable, or "skill" denoted as $z$. The policy becomes $\pi(a|s,z)$ aiming to produce distinguishable behaviors for different values of $z$. This diversity is enforced using an information-theoretic objective:

$$
F(\theta) = H[Z] - H[Z|S] + H[A|S,Z]
$$

$$
\ge H[A|S,Z] + E_{s\sim{p(z)}, s\sim{\pi(z)}}[\log{q_{\phi}(z|s)-\log{p(z)}}]
$$

Here, $H[A|S,Z]$ corresponds to the entropy of the policy, which is already present in algorithms like [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1801.01290). The second term, $\log{q_{\phi}(z|s)-\log{p(z)}}$ acts as a pseudo-reward that encourages the agent to produce behaviors that make the skill $z$ easily inferable from the resulting state. This allows the agent to autonomously discover a diverse repertoire of skills without any external task reward.

```
python cs285/scripts/run_diayn.py -cfg experiments/diayn/halfcheetah_diayn.yaml --n_skills 50
```

Below are some of the skills learned in `HalfCheetah-v4` environment using DIAYN:

<table>
  <tr>
    <td align="center">
      <figure>
        <img src="https://github.com/user-attachments/assets/0450ca4f-4e26-4ab8-9bec-70eed97c0b6b" width="90%"/> 
        <figcaption>Backward Walking</figcaption>
      </figure>
    </td>
    <td align="center">
      <figure>
        <img src="https://github.com/user-attachments/assets/74f0014d-cceb-4891-b115-d319ef9a44f8" width="92%" />
        <figcaption>Forward Walking</figcaption>
      </figure>
    </td>
    <td align="center">
      <figure>
        <img src="https://github.com/user-attachments/assets/7d9b2545-9927-40c6-ae0d-51b96ea0a6ac" width="99%" />
        <figcaption>Stationary Standing</figcaption>
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


</br>

# 💼 Offline Reinforcement Learning

Previous reinforcement learning (RL) algorithms primarily rely on the online learning paradigm where an agent continuously interacts with the real environment to update its policy. However, such interaction can be extremely costly, consider, for example, the case of autonomous driving. This high cost of real-world interaction has been one of the major obstacles to the widespread adoption of RL.

Offline reinforcement learning (Offline RL) offers an alternative by learning solely from previously collected data without requiring any additional online interaction. This approach aligns more closely with data-driven learning methods and is sometimes referred to as data-driven reinforcement learning. By leveraging large offline datasets, Offline RL has the potential to learn more generalizable policies much like how neural networks in computer vision learn generalized patterns from massive amounts of labeled images.

<p align="center">
    <img src="https://github.com/user-attachments/assets/594e4ede-ac04-47d4-af15-34d54c9afc05" width="99%"/>
    <!-- <figcaption align="center">Pictorial illustration of classic online reinforcement learning (a), classic off-policy reinforcement learning (b), and offline reinforcement learning (c) mentioned in this <a href="https://arxiv.org/pdf/2005.01643">paper</a>.</figcaption> -->
</p>

However, one major challenge arises from the mismatch in assumptions between computer vision and offline reinforcement learning (Offline RL). In computer vision, models are typically trained under the assumption that input images are i.i.d., which helps them learn generalized patterns effectively. In contrast, Offline RL often operates in out-of-distribution (OOD) scenarios where the agent must make decisions in states that were not well represented in the training data, specifically $a'$ in $E_{a'\sim\pi_{new}}(s',a')$. To avoid unpredictable or unsafe behavior, the Offline RL agent must at least behave conservatively, closely matching the behavior policy $\pi_\beta$ (the policy used to collect the dataset) especially in regions of the state-action space where data is sparse, or estimate Q-values with less bias (accurately) without using $a'$.

In summary, Offline RL involves balancing two competing objectives: **① learning a policy $\pi_{\theta}$ that outperforms the behavior policy $\pi_{\beta}$**, and **② limiting divergence from the behavior policy to mitigate distributional shift errors.** In this section, I will introduce three Offline RL methods(CQL, AWAC, and IQL), each of which takes a distinct approach to address this trade-off.

<p align="center">
    <img src="https://github.com/user-attachments/assets/cc5aea33-42b2-44d6-b433-685ea72fe11e" width="99%"/>
</p>

(See more detailed Offline RL in these folllowing papers: [paper1](https://arxiv.org/pdf/2005.01643), [paper2](https://arxiv.org/pdf/1906.00949), [paper3](https://arxiv.org/pdf/2204.05618))

## Experiment 1: Conservative Q-Learning (CQL)

Offline RL wants to discourage the agent from assigning high Q-values to out-of-distribution (OOD) actions (not seen in the dataset $D$). At the same time, it encourages the agent to assign higher Q-values for in-distribution actions. To address this, [Conservative Q-Learning (CQL)](https://arxiv.org/pdf/2006.04779) introduces an additional regularization term in its objective. This term works by minimizing the soft maximum of Q-values, specifically $\log\sum_{a}\exp(Q(s,a))$, which penalizes high Q-values across all actions. Simultaneously, it maximizes the Q-values $Q(s, a)$ of actions actually observed in the dataset $D$ ensuring the policy remains grounded in the data. This conservative approach helps improve stability and reliability in offline settings where exploration is not possible.

(Note that the term *overestimation* in CQL differs from its use in [Double-DQN](https://arxiv.org/pdf/1509.06461))

**❓Q1. Compare CQL with naive DQN (where regularization coefficient $\alpha=0$).**

<p align="center">
    <img src="https://github.com/user-attachments/assets/c7d55d1c-8a21-49c4-bd72-eb28a6380925" width="49%"/>
    <img src="https://github.com/user-attachments/assets/4e8078a1-dacf-4daf-8e03-da3b08bc1afc" width="49%"/>
    <!-- <figcaption align="center">lb stands for large batch size</figcaption> -->
</p>

The offline dataset used here is the one collected during the Exploration section (above) using the PointMass-Medium environment. As shown in the graph above, both DQN and CQL achieve similar evaluation returns (i.e., rewards), indicating comparable policy performance. However, CQL consistently produces lower Q-values compared to DQN. This behavior is intentional as CQL is designed to suppress Q-values in order to counteract overestimation, which is a fundamental challenge in offline reinforcement learning. By doing so, CQL promotes safer and more conservative value estimates when learning from fixed datasets without online interaction.

<!-- **❓Q2. Compare CQL with different $\alpha$ values range from [0.1, 1.0, 10].**

<p align="center">
    <img src="" width="49%"/>
    <figcaption align="center">lb stands for large batch size</figcaption>
</p> -->


## Experiment 2: AWAC & IQL

Conservative Q-Learning (CQL) tackles the challenges of offline reinforcement learning by adding a regularization term to the standard temporal difference (TD) loss, discouraging overestimation of Q-values. In contrast, methods like [Implicit Q-Learning (IQL)](https://arxiv.org/pdf/2110.06169) and [Advantage-Weighted Actor-Critic (AWAC)](https://arxiv.org/pdf/2006.09359) take a different route: rather than correcting Q-values later, they aim to learn an in-distribution policy $\pi$ that remains close to the behavior policy $\pi_{\beta}$ found in the offline data. Thus, they replace the traditional target:

$$
Q(s,a)←r(s,a) + \gamma\max\limits_{a'}Q(s', a')
$$

with a policy-weighted expectation: 

$$
Q(s,a)←r(s,a) + \gamma E_{a'\sim\pi_{new}}[Q(s', a')]
$$

which biases learning toward actions $a'$ likely under the learned policy.

✔︎ **AWAC (Advantage Weighted Actor-Critic)**, as the name suggests, aims to learn an explicit actor policy $\pi$ by maximizing a weighted log-likelihood objective:

$$
\theta ⟵ \arg\max\limits_{\theta}E_{s,a\sim B}[\log\pi_{\theta}(a|s)\exp(\frac{1}{\lambda}A^{\pi_k}(s,a))]
$$

This objective encourages the policy to favor actions with high advantage estimates $A^{\pi_k(s,a)}$ while down-weighting actions with low advantages. As a result, the learned policy prioritizes actions that are likely to improve performance based on the current critic. You can also interpret this process as a form of regression toward the behavior policy $\pi_\beta$ with the advantage function guiding which actions to emphasize during learning.

By updating the policy in this way, the agent learns to focus on high-advantage actions and effectively ignores low-advantage or out-of-distribution (OOD) actions, especially if $\pi$ closely approximates the weighted behavior policy. Once the actor is updated, the Q-function can be trained using a standard temporal difference (TD) loss, similar to previous approaches.

✔︎ Unlike AWAC, which explicitly performs policy improvement by weighting actions based on their estimated advantage, **Implicit Q-Learning (IQL)** takes an implicit approach. It estimates the state value function using expectile regression, which shifts focus toward high-performing actions within the dataset—without ever querying out-of-distribution actions.

To do this, IQL introduces a separate value function $V_{\phi}(s)$ which is trained to match the expectile of the Q-values for actions seen in the data:

$$
L_{V}(\phi) = E_{(s,a)\sim{D}}[L_{2}^{\tau}(Q_{\theta}(s,a)-V_{\phi}(s))]
$$

Here, $L_{2}^{\tau}$ is the asymmetric squared loss that places more weight on overestimates when $\tau>0.5$ thus favoring higher Q-values. This helps approximate the maximum in-distribution value without querying actions outside the dataset.

Then, the Q-function is updated using the value function as a bootstrapped target:

$$
L_{Q}(\theta) = E_{(s,a,s')\sim{D}}[(r(s,a)+\gamma V_{\phi}(s')-Q_{\theta}(s,a))^2]
$$

This two-stage update process enables IQL to avoid the risk of overestimation from out-of-distribution actions, a common issue in offline RL.

</br>

<!-- **❓Q1. Compare AWAC with IQL.**

<p align="center">
    <img src="" width="49%"/>
    <img src="" width="49%"/>
</p> -->


## Experiment 3: Offline Pre-Training → Online Fine-Tuning

The offline dataset was collected using RND-based exploration in Experiment 1 as shown below.

<p align="center">
    <img src="https://github.com/user-attachments/assets/f9523fbd-d89e-4829-82ec-c9d65941b380" width="30%"/> 
</p>

In this experiment, I pre-trained the models on the offline dataset and fine-tuned them in an online setting. The results below indicate that offline RL methods do not consistently reach the final destination and may even get stuck despite achieving the goal. In contrast, fine-tuning with online data reliably helps the models converge to the target.

<table>
  <tr>
    <td align="center">
      <figure>
        <img src="https://github.com/user-attachments/assets/deb02a21-a364-4c25-bdbb-6af9e533317a" width="49%" />
        <img src="https://github.com/user-attachments/assets/310f3b49-2d6f-43cf-a91f-5ae6d1c9b401" width="49%" />
        <img src="https://github.com/user-attachments/assets/84f744cd-24c4-42a5-a403-5584ecfff1f6" width="49%" />
        <img src="https://github.com/user-attachments/assets/f5ba5561-1610-4b92-b6e8-f01c48a7b784" width="49%" />
        <figcaption>CQL</figcaption>
      </figure>
    </td>
    <td align="center">
      <figure>
        <img src="" width="49%" />
        <img src="https://github.com/user-attachments/assets/c12f9b20-6e94-4edf-8221-8dbaf2f035af" width="49%" />
        <img src="https://github.com/user-attachments/assets/2e78daec-cde9-4472-bddb-e90197ffce5e" width="49%" />
        <img src="https://github.com/user-attachments/assets/08d3dcee-7e50-4fa2-b40e-86a6d095527b" width="49%" />
        <figcaption>CQL + <b>Fine-Tuning</b></figcaption>
      </figure>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <figure>
        <img src="https://github.com/user-attachments/assets/70afe859-a34a-4d16-ba84-a6427eb4b775" width="49%" />
        <img src="https://github.com/user-attachments/assets/f0c0aa9f-d778-4ec4-92b1-ac7edc67be24" width="49%" />
        <img src="https://github.com/user-attachments/assets/be7de219-13fa-4096-9902-33c23d712a73" width="49%" />
        <img src="https://github.com/user-attachments/assets/2e886083-8c60-4ef1-91c5-17578bb5a673" width="49%" />
        <figcaption>AWAC</figcaption>
      </figure>
    </td>
    <td align="center">
      <figure>
        <img src="https://github.com/user-attachments/assets/d477c31c-40a2-4cd2-8dd6-ae289a20e627" width="49%" />
        <img src="https://github.com/user-attachments/assets/ed3a18c2-9ed4-4701-be9c-b4b41b0d4cab" width="49%" />
        <img src="https://github.com/user-attachments/assets/a9a70c9e-575a-4458-a811-9f3d1adc82c0" width="49%" />
        <img src="https://github.com/user-attachments/assets/4ff8e42d-e8e9-4584-8a0a-dd1afea73a7d" width="49%" />
        <figcaption>AWAC + <b>Fine-Tuning</b></figcaption>
      </figure>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <figure>
        <img src="" width="49%" />
        <img src="" width="49%" />
        <img src="" width="49%" />
        <img src="" width="49%" />
        <figcaption>IQL</figcaption>
      </figure>
    </td>
    <td align="center">
      <figure>
        <img src="" width="49%" />
        <img src="" width="49%" />
        <img src="" width="49%" />
        <img src="" width="49%" />
        <figcaption>IQL + <b>Fine-Tuning</b></figcaption>
      </figure>
    </td>
  </tr>
</table>