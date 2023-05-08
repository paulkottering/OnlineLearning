import os
import json
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_regrets(file_path, output_path):
    with open(file_path) as f:
        data = json.load(f)

    parameters = data["parameters"]
    nash_regrets = np.cumsum(np.array(data["nash"]), axis=1)
    potential_regrets = np.cumsum(np.array(data["potential"]), axis=1)
    nikaido_isoda_regrets = np.cumsum(np.array(data["nikaido_isoda"]), axis=1)

    sns.set(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    x = np.arange(nash_regrets.shape[1])

    axes[0].plot(nash_regrets.mean(axis=0))
    axes[0].fill_between(x, nash_regrets.mean(axis=0) - nash_regrets.std(axis=0),
                         nash_regrets.mean(axis=0) + nash_regrets.std(axis=0), alpha=0.2)
    axes[0].set_title("Nash Regret")

    axes[1].plot(potential_regrets.mean(axis=0))
    axes[1].fill_between(x, potential_regrets.mean(axis=0) - potential_regrets.std(axis=0),
                         potential_regrets.mean(axis=0) + potential_regrets.std(axis=0), alpha=0.2)
    axes[1].set_title("Potential Regret")

    axes[2].plot(nikaido_isoda_regrets.mean(axis=0))
    axes[2].fill_between(x, nikaido_isoda_regrets.mean(axis=0) - nikaido_isoda_regrets.std(axis=0),
                         nikaido_isoda_regrets.mean(axis=0) + nikaido_isoda_regrets.std(axis=0), alpha=0.2)
    axes[2].set_title("Nikaido-Isoda Regret")

    for ax in axes:
        ax.set_xlabel("Timesteps")

    axes[0].set_ylabel("Cumulative Regret")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    # Create textbox with parameter list
    parameter_text = ", ".join([f"{k}: {v}" for k, v in parameters.items()])
    plt.figtext(0.5, 0.1, parameter_text, ha="center", fontsize=12, bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    plt.savefig(output_path)

def find_file(filename):
    folder = "log_files"
    for file in os.listdir(folder):
        if file == filename:
            return os.path.join(folder, file)
    return None

def create_output_path(filename):
    folder = "run_plots"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(folder, f"run_{filename[:-5]}.png")

def main():
    file_number = 1682611244
    filename = "simulation_" + str(file_number) + ".json"
    file_path = find_file(filename)
    output_path = create_output_path(filename)
    print(file_path)
    plot_regrets(file_path, output_path)

if __name__ == "__main__":
    main()
