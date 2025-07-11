import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = "./data"
EXPERIMENTS = os.listdir(DATA_DIR)

def get_data(log_dir, tag):
    # Load the event file
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # For example, get scalar data for a specific tag
    scalars = event_acc.Scalars(tag)
    return scalars

def dqn_cartpole_return_graph():
    data_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_23-06"): # lr = 1e-3
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'eval_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df

    # plot
    plt.figure(figsize=(8, 5))
    for key, value in data_dict.items():
        if key.endswith("46-10"):
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], color="orange", label="DQN_CartPole-v1_s64_l2_d0.99")
    plt.title("Learning Curves")
    plt.ylabel("Eval. Return")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "dqn_return.png"))

def dqn_cartpole_q_value_graph():
    data_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_07-07"): # lr = 0.05
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'q_values')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df
        if exp.startswith("hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_23-06"): # lr = 1e-3
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'q_values')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df

    # plot
    plt.figure(figsize=(8, 5))
    for key, value in data_dict.items():
        if key.endswith("35-32"):
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], label="lr=0.05")
        if key.endswith("46-10"):
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], label="lr=0.001")
    plt.title("Q-Values for different learning rate (CartPole-v1)")
    plt.ylabel("Q-values")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "dqn_q_value.png"))

def dqn_cartpole_critic_loss_graph():
    data_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_07-07"): # lr = 0.05
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'critic_loss')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df
        if exp.startswith("hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_23-06"): # lr = 1e-3
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'critic_loss')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df

    # plot
    plt.figure(figsize=(8, 5))
    for key, value in data_dict.items():
        if key.endswith("35-32"):
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], label="lr=0.05")
        if key.endswith("46-10"):
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], label="lr=0.001")
    plt.title("Critic Loss for different learning rate (CartPole-v1)")
    plt.ylabel("Critic Loss")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "dqn_critic_loss.png"))


def dqn_lunarlander_q_value_graph():
    data_dict = {}
    true_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_07-07-2025_22-25-06"): # lr = 0.05
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'q_values')
            true_scalars = get_data(log_dir, 'eval_discount_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            true_dict[exp] = true_scalars[-1]
            data_dict[exp] = df
        if exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_07-07-2025_22-40-46"): # lr = 1e-3
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'q_values')
            true_scalars = get_data(log_dir, 'eval_discount_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            true_dict[exp] = true_scalars[-1]
            data_dict[exp] = df
        if exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_07-07-2025_22-55-45"): # lr = 1e-3
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'q_values')
            true_scalars = get_data(log_dir, 'eval_discount_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            true_dict[exp] = true_scalars[-1]
            data_dict[exp] = df

    # plot
    plt.figure(figsize=(8, 5))
    for key, value in data_dict.items():
        # if key.endswith("25-06"):
        #     plt.plot(data_dict[key]['step'], data_dict[key]['value'], label="seed=1", color='red')
        #     plt.axhline(y=true_dict[key].value, linestyle='-', color='red')
        # if key.endswith("40-46"):
        #     plt.plot(data_dict[key]['step'], data_dict[key]['value'], label="seed=2", color='green')
        #     plt.axhline(y=true_dict[key].value, linestyle='-', color='green')
        if key.endswith("55-45"):
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], color='skyblue')
            plt.axhline(y=true_dict[key].value, linestyle='-', color='blue')

    plt.title("Q-Values for different seed (LunarLander-v2)")
    plt.ylabel("Q-Values")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "dqn_lunarlander_q_values.png"))


if __name__ == "__main__":
    # dqn_cartpole_return_graph()
    # dqn_cartpole_q_value_graph()
    # dqn_cartpole_critic_loss_graph()
    dqn_lunarlander_q_value_graph()

