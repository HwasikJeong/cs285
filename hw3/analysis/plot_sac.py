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

def sac_reinforce_return_graph():
    data_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("hw3_sac_reinforce1_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'eval_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["reinforce-1"] = df
        if exp.startswith("hw3_sac_reinforce10_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'eval_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["reinforce-10"] = df

    # plot
    plt.figure(figsize=(8, 5))
    for key, value in data_dict.items():
        plt.plot(data_dict[key]['step'], data_dict[key]['value'], label=key)
        
    plt.title('SAC REINFORCE Actor Update Across Varying Sample Sizes (HalfCheetah-v4)')
    plt.ylabel("Eval. Return")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "sac_REINFORCE.png"))


def sac_reparametrization_return_graph():
    data_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("hw3_sac_reinforce1_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'eval_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["reinforce-1"] = df
        if exp.startswith("hw3_sac_reinforce10_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'eval_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["reinforce-10"] = df
        if exp.startswith("hw3_sac_reparametrize_"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'eval_return')
            df = pd.DataFrame({
                'step': [s.step for s in return_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict["reparametrize"] = df

    # plot
    plt.figure(figsize=(8, 5))
    for key, value in data_dict.items():
        plt.plot(data_dict[key]['step'], data_dict[key]['value'], label=key)
        
    plt.title('SAC Actor Update: REINFORCE vs. REPARAMETRIZATION (HalfCheetah-v4)')
    plt.ylabel("Eval. Return")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "sac_REPARAMETRIZATION.png"))


if __name__ == "__main__":
    # sac_reinforce_return_graph()
    sac_reparametrization_return_graph()

