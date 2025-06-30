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

def hw2_cheetah_baseline_loss_graph():
    for exp in EXPERIMENTS:
        if exp.startswith("q2_pg_cheetah_baseline_HalfCheetah-v4_05"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'Baseline_Loss')
            step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
            baseline_df = pd.DataFrame({
                'step': [s.value for s in step_scalars],
                'value': [s.value for s in return_scalars],
            })
        
    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(baseline_df['step'], baseline_df['value'], label="baseline")
    plt.title("Learning Curves (Loss) for Vanilla vs. Baseline")
    plt.ylabel("Baseline Loss")
    plt.xlabel("Env. Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "hw2_exp2_halfcheetah_baseline_loss.png"))


def hw2_cheetah_baseline_return_graph():
    for exp in EXPERIMENTS:
        if exp.startswith("q2_pg_cheetah_baseline_HalfCheetah-v4_05"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'Eval_AverageReturn')
            step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
            baseline_df = pd.DataFrame({
                'step': [s.value for s in step_scalars],
                'value': [s.value for s in return_scalars],
            })
        if exp.startswith("q2_pg_cheetah_HalfCheetah-v4_05"):
            log_dir = os.path.join(DATA_DIR, exp)
            return_scalars = get_data(log_dir, 'Eval_AverageReturn')
            step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
            vanilla_df = pd.DataFrame({
                'step': [s.value for s in step_scalars],
                'value': [s.value for s in return_scalars],
            })

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(baseline_df['step'], baseline_df['value'], label="baseline")
    plt.plot(vanilla_df['step'], vanilla_df['value'], label="vanilla")
    plt.title("Learning Curves for Eval. Return")
    plt.ylabel("Eval_AverageReturn")
    plt.xlabel("Env. Step")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=300, color='red', linestyle='-')
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "hw2_exp2_halfcheetah_baseline_return.png"))


def hw2_cheetah_baseline_bgs_graph():
    data_loss_dict = {}
    data_return_dict = {}
    for exp in EXPERIMENTS:
        for i in range(1, 6):
            if exp.startswith(f"q2_pg_cheetah_baseline_HalfCheetah-v4_bgs{i}_blr0.01"):
                log_dir = os.path.join(DATA_DIR, exp)
                loss_scalars = get_data(log_dir, 'Baseline_Loss')
                return_scalars = get_data(log_dir, 'Eval_AverageReturn')
                step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
                df_loss = pd.DataFrame({
                    'step': [s.step for s in loss_scalars],
                    'value': [s.value for s in loss_scalars],
                })
                df_return = pd.DataFrame({
                    'step': [s.value for s in step_scalars],
                    'value': [s.value for s in return_scalars],
                })
                data_loss_dict[exp] = df_loss
                data_return_dict[exp] = df_return

    # plot return
    plt.figure(figsize=(8, 5))
    for i in range(1, 6):
        for key, value in data_return_dict.items():
            if key.startswith(f"q2_pg_cheetah_baseline_HalfCheetah-v4_bgs{i}_blr0.01"):
                plt.plot(data_return_dict[key]['step'], data_return_dict[key]['value'], label=f"bgs{i}")
    plt.title("Evaluation Return for different Baseline Gradient Steps")
    plt.ylabel("Eval. Average Return")
    plt.xlabel("Env. Step")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=300, color='red', linestyle='-')
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "hw2_bgs_return.png"))

     # plot loss
    plt.figure(figsize=(8, 5))
    for i in range(1, 6):
        for key, value in data_loss_dict.items():
            if key.startswith(f"q2_pg_cheetah_baseline_HalfCheetah-v4_bgs{i}_blr0.01"):
                plt.plot(data_loss_dict[key]['step'], data_loss_dict[key]['value'], label=f"bgs{i}")
    plt.title("Baseline Loss for different Baseline Gradient Steps")
    plt.ylabel("Baseline Loss")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "hw2_bgs_loss.png"))


def hw2_cheetah_baseline_blr_graph():
    data_loss_dict = {}
    data_return_dict = {}
    for exp in EXPERIMENTS:
        for i in [0.01, 0.001, 0.0001]:
            if exp.startswith(f"q2_pg_cheetah_baseline_HalfCheetah-v4_bgs5_blr{i}"):
                log_dir = os.path.join(DATA_DIR, exp)
                loss_scalars = get_data(log_dir, 'Baseline_Loss')
                return_scalars = get_data(log_dir, 'Eval_AverageReturn')
                step_scalars = get_data(log_dir, 'Train_EnvstepsSoFar')
                df_loss = pd.DataFrame({
                    'step': [s.step for s in loss_scalars],
                    'value': [s.value for s in loss_scalars],
                })
                df_return = pd.DataFrame({
                    'step': [s.value for s in step_scalars],
                    'value': [s.value for s in return_scalars],
                })
                data_loss_dict[exp] = df_loss
                data_return_dict[exp] = df_return

    # plot return
    plt.figure(figsize=(8, 5))
    for i in [0.01, 0.001, 0.0001]:
        for key, value in data_return_dict.items():
            if key.startswith(f"q2_pg_cheetah_baseline_HalfCheetah-v4_bgs5_blr{i}"):
                plt.plot(data_return_dict[key]['step'], data_return_dict[key]['value'], label=f"lr{i}")
    plt.title("Evaluation Return for different Baseline Learning Rate")
    plt.ylabel("Eval. Average Return")
    plt.xlabel("Env. Step")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=300, color='red', linestyle='-')
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "hw2_blr_return.png"))

     # plot loss
    plt.figure(figsize=(8, 5))
    for i in [0.01, 0.001, 0.0001]:
        for key, value in data_loss_dict.items():
            if key.startswith(f"q2_pg_cheetah_baseline_HalfCheetah-v4_bgs5_blr{i}"):
                plt.plot(data_loss_dict[key]['step'], data_loss_dict[key]['value'], label=f"lr{i}")
    plt.title("Baseline Loss for different Baseline Gradient Steps")
    plt.ylabel("Baseline Loss")
    plt.xlabel("Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('./analysis/graph', "hw2_blr_loss.png"))



if __name__ == "__main__":
    # hw2_cheetah_baseline_loss_graph()
    hw2_cheetah_baseline_return_graph()
    # hw2_cheetah_baseline_bgs_graph()
    # hw2_cheetah_baseline_blr_graph()

