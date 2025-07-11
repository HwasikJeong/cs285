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

def sac_stabilizing_return_graph():
    data_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("hw3_sac_sac_hopper_clipq_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'eval_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["clip-q"] = df
        if exp.startswith("hw3_sac_sac_hopper_doubleq_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'eval_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["double-q"] = df
        if exp.startswith("hw3_sac_sac_hopper_redq_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'eval_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["redq"] = df
        if exp.startswith("hw3_sac_sac_hopper_singlecritic_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'eval_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["single_critic"] = df

    li = ["single_critic", "double-q", "clip-q", "redq"]
    # plot
    plt.figure(figsize=(8, 5))
    for key in li:
        plt.plot(data_dict[key]['step'], data_dict[key]['value'], label=key)
        
    plt.title('SAC with Varying Q-Backup Methods (HalfCheetah-v4)')
    plt.ylabel("Eval. Return")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "sac_q_backup_return.png"))


def sac_stabilizing_q_value_graph():
    data_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("hw3_sac_sac_hopper_clipq_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'q_values')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["clip-q"] = df
        if exp.startswith("hw3_sac_sac_hopper_doubleq_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'q_values')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["double-q"] = df
        if exp.startswith("hw3_sac_sac_hopper_redq_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'q_values')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["redq"] = df
        if exp.startswith("hw3_sac_sac_hopper_singlecritic_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'q_values')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["single_critic"] = df

    li = ["single_critic", "double-q", "clip-q", "redq"]
    # plot
    plt.figure(figsize=(8, 5))
    for key in li:
        plt.plot(data_dict[key]['step'], data_dict[key]['value'], label=key)
        
    plt.title('SAC with Varying Q-Backup Methods (HalfCheetah-v4)')
    plt.ylabel("Q-Values")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "sac_q_backup_q_values.png"))


if __name__ == "__main__":
    sac_stabilizing_return_graph()
    sac_stabilizing_q_value_graph()

