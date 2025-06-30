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

def hw2_cartpole_graph():
    data_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("q2_pg_cartpole_CartPole-v0"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'Eval_AverageReturn')
            step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
            df = pd.DataFrame({
                'step': [s.value for s in step_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df
        elif exp.startswith("q2_pg_cartpole_na_CartPole"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'Eval_AverageReturn')
            step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
            df = pd.DataFrame({
                'step': [s.value for s in step_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df
        elif exp.startswith("q2_pg_cartpole_rtg_CartPole"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'Eval_AverageReturn')
            step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
            df = pd.DataFrame({
                'step': [s.value for s in step_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df
        elif exp.startswith("q2_pg_cartpole_rtg_na_CartPole"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'Eval_AverageReturn')
            step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
            df = pd.DataFrame({
                'step': [s.value for s in step_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df

    li = ["q2_pg_cartpole_CartPole-v0", 
            "q2_pg_cartpole_rtg_CartPole-v0",
            "q2_pg_cartpole_na_CartPole-v0",
            "q2_pg_cartpole_rtg_na_CartPole-v0"
            ]
    # plot
    plt.figure(figsize=(8, 5))
    for l in li:
        for key, value in data_dict.items():
            if key.startswith(l):
                plt.plot(data_dict[key]['step'], data_dict[key]['value'], label=key[3:-20])
    plt.title("Learning Curves (average return vs. number of environment steps)")
    plt.ylabel("Eval. Average Return")
    plt.xlabel("Env. Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "hw2_cartpole.png"))


def hw2_cartpole_lb_graph():
    data_dict = {}
    for exp in EXPERIMENTS:
        if exp.startswith("q2_pg_cartpole_lb_CartPole-v0"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'Eval_AverageReturn')
            step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
            df = pd.DataFrame({
                'step': [s.value for s in step_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df
        elif exp.startswith("q2_pg_cartpole_lb_na_CartPole"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'Eval_AverageReturn')
            step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
            df = pd.DataFrame({
                'step': [s.value for s in step_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df
        elif exp.startswith("q2_pg_cartpole_lb_rtg_CartPole"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'Eval_AverageReturn')
            step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
            df = pd.DataFrame({
                'step': [s.value for s in step_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df
        elif exp.startswith("q2_pg_cartpole_lb_rtg_na_CartPole"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'Eval_AverageReturn')
            step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
            df = pd.DataFrame({
                'step': [s.value for s in step_scalars],
                'value': [s.value for s in return_scalars],
            })
            data_dict[exp] = df

    li = ["q2_pg_cartpole_lb_CartPole-v0", 
            "q2_pg_cartpole_lb_rtg_CartPole-v0",
            "q2_pg_cartpole_lb_na_CartPole-v0",
            "q2_pg_cartpole_lb_rtg_na_CartPole-v0"
            ]
    # plot
    plt.figure(figsize=(8, 5))
    for l in li:
        for key, value in data_dict.items():
            if key.startswith(l):
                plt.plot(data_dict[key]['step'], data_dict[key]['value'], label=key[3:-20])
    plt.title("Learning Curves (average return vs. number of environment steps)")
    plt.ylabel("Eval. Average Return")
    plt.xlabel("Env. Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "hw2_cartpole_lb.png"))

if __name__ == "__main__":
    hw2_cartpole_graph()
    hw2_cartpole_lb_graph()

