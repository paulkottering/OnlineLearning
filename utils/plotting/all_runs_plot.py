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
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    x = np.arange(nash_regrets.shape[1])
    for i in range(parameters["runs"]):
        axes[0].plot(np.cumsum(data["nash"][i]))
    axes[0].set_title("Nash Regret")

    for i in range(parameters["runs"]):
        axes[1].plot(np.cumsum(data["potential"][i]))
    axes[1].set_title("Potential Regret")

    for i in range(parameters["runs"]):
        axes[2].plot(np.cumsum(data["nikaido_isoda"][i]))
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
    folder = "all_run_plots"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(folder, f"run_{filename[:-5]}.png")

def main():
    file_number = 1682585069
    filename = "simulation_" + str(file_number) + ".json"
    file_path = find_file(filename)
    output_path = create_output_path(filename)
    print(file_path)
    plot_regrets(file_path, output_path)

if __name__ == "__main__":
    main()
