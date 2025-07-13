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

def hw2_gae_lunar_return_graph():
    data_return_dict = {}
    for exp in EXPERIMENTS:
        for i in [0.0, 0.95, 0.98, 0.99, 1.0]:
            if exp.startswith(f"q2_pg_lunar_lander_lambda{i}_LunarLander-v2"):
                log_dir = os.path.join(DATA_DIR, exp)
                return_scalars = get_data(log_dir, 'Eval_AverageReturn')
                step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
                df_return = pd.DataFrame({
                    'step': [s.value for s in step_scalars],
                    'value': [s.value for s in return_scalars],
                })
                data_return_dict[exp] = df_return

    # plot return
    plt.figure(figsize=(8, 5))
    for i in [0.0, 0.95, 0.98, 0.99, 1.0]:
        for key, value in data_return_dict.items():
            if key.startswith(f"q2_pg_lunar_lander_lambda{i}_LunarLander-v2"):
                plt.plot(data_return_dict[key]['step'], data_return_dict[key]['value'], label=f"lambda{i}")
    plt.title("Evaluation Return for different lambda in GAE")
    plt.ylabel("Eval. Average Return")
    plt.xlabel("Env. Step")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=200, color='red', linestyle='-')
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "hw2_gae_lambda_return.png"))


def hw2_gae_lunar_return_std_graph():
    data_return_dict = {}
    for exp in EXPERIMENTS:
        for i in [0.0, 0.95, 0.98, 0.99, 1.0]:
            if exp.startswith(f"q2_pg_lunar_lander_lambda{i}_LunarLander-v2"):
                log_dir = os.path.join(DATA_DIR, exp)
                return_scalars = get_data(log_dir, 'Eval_StdReturn')
                step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
                df_return = pd.DataFrame({
                    'step': [s.value for s in step_scalars],
                    'value': [s.value for s in return_scalars],
                })
                data_return_dict[exp] = df_return

    # plot return
    plt.figure(figsize=(8, 5))
    for i in [0.0, 0.95, 0.98, 0.99, 1.0]:
        for key, value in data_return_dict.items():
            if key.startswith(f"q2_pg_lunar_lander_lambda{i}_LunarLander-v2"):
                plt.plot(data_return_dict[key]['step'], data_return_dict[key]['value'], label=f"lambda{i}")
    plt.title("Evaluation Standard Deviation of Return for different lambda in GAE")
    plt.ylabel("Eval. Std Return")
    plt.xlabel("Env. Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "hw2_gae_lambda_return_std.png"))

if __name__ == "__main__":
    hw2_gae_lunar_return_graph()
    # hw2_gae_lunar_return_std_graph()