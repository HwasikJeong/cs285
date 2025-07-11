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

def double_dqn_lunarlander_return_graph():
    data_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_23-06-2025_16-01-29")\
            or exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_23-06-2025_16-21-38")\
                or exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_23-06-2025_19-58-02"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'eval_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df
        if exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_23-06-2025_14-51-13")\
            or exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_23-06-2025_15-02-50")\
                or exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_23-06-2025_15-17-06"):
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
        if key.endswith("2025_16-01-29"): # double-dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], label='double-dqn', color='red')
        if key.endswith("2025_16-21-38"): # double-dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], color='red')
        if key.endswith("2025_19-58-02"): # double-dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], color='red')
        if key.endswith("2025_14-51-13"): # dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], label='dqn', color='blue')
        if key.endswith("2025_15-02-50"): # dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], color='blue')
        if key.endswith("2025_15-17-06"): # dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], color='blue')
        
    plt.title('Eval. Return for Double-DQN vs. DQN (LunarLander-v2)')
    plt.ylabel("Eval. Return")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "double_dqn_lunarlander_return.png"))


def double_dqn_lunarlander_q_value_graph():
    data_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_23-06-2025_16-01-29")\
            or exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_23-06-2025_16-21-38")\
                or exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_doubleq_23-06-2025_19-58-02"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'q_values')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df
        if exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_23-06-2025_14-51-13")\
            or exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_23-06-2025_15-02-50")\
                or exp.startswith("hw3_dqn_dqn_LunarLander-v2_s64_l2_d0.99_23-06-2025_15-17-06"):
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
        if key.endswith("2025_16-01-29"): # double-dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], label='double-dqn', color='red')
        if key.endswith("2025_16-21-38"): # double-dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], color='red')
        if key.endswith("2025_19-58-02"): # double-dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], color='red')
        if key.endswith("2025_14-51-13"): # dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], label='dqn', color='blue')
        if key.endswith("2025_15-02-50"): # dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], color='blue')
        if key.endswith("2025_15-17-06"): # dqn
            plt.plot(data_dict[key]['step'], data_dict[key]['value'], color='blue')
        
    plt.title('Q-Values for Double-DQN vs. DQN (LunarLander-v2)')
    plt.ylabel("Q-Values")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "double_dqn_lunarlander_q_values.png"))


if __name__ == "__main__":
    # double_dqn_lunarlander_return_graph()
    double_dqn_lunarlander_q_value_graph()

