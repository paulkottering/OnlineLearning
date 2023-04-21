import pandas as pd
import os
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_bar_charts(varying_values_choice, data):
    sns.set_theme()  # Set seaborn theme
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    bar_width = 0.35

    x_labels = varying_values_choice["noise"]
    colors = ['orange', 'blue']

    title_mapping = {
        'random': 'Uniform',
        'neg_skewed': 'Negatively Skewed',
        'pos_skewed': 'Positively Skewed',
        'tailed_skewed': 'Tail Skewed',
        'mid_skewed': 'Symmetrical',
        'cooperative': 'Cooperative'
    }

    for row in range(2):
        for col in range(3):
            index = np.arange(3)
            for i in range(2):
                axes[row, col].bar(
                    index + bar_width * i,
                    data[3 * row + col, index, i],
                    bar_width,
                    color=colors[i],
                    label=f'Group {i + 1}',
                )
            # Set the title above the graph and increase the font size
            axes[row, col].set_title(title_mapping[varying_values_choice["game"][3 * row + col]], fontsize=20)
            axes[row, col].title.set_position([.5, 0.02])

            # Set x-axis labels with larger font size
            axes[row, col].set_xticks(index + bar_width / 2)
            axes[row, col].set_xticklabels(x_labels, fontsize=14)

            # Set y-axis labels with larger font size
            axes[row, col].tick_params(axis='y', which='major', labelsize=14)

            # Set x-axis label for all subplots
            axes[row, col].set_xlabel('Noise Level', fontsize=16)

            # Set y-axis label for all subplots
            axes[row, col].set_ylabel('Cumulative Nash Regret', fontsize=16)

    # Create a custom legend at the top of the graph
    legend_elements = [
        plt.Line2D([0], [0], color=colors[i], lw=4, label=varying_values_choice["solver"][i])
        for i in range(2)
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.05),
               fontsize=14)

    plt.tight_layout()
    if not os.path.exists("barcharts"):
        os.makedirs("barcharts")
    title = f"barcharts/bar_chart.png"
    plt.savefig(title, bbox_inches="tight")
    print("Plot saved to", title)


def read_simulation_log_files(folder="log_files"):
    log_files = []

    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), "r") as file:
                data = json.load(file)
                log_files.append((data, filename))
    return log_files

def filter_logs_by_2_parameters(log_files, fixed_parameters, varying_parameters, varying_values_choice):
    filtered_logs = {}
    file_names = {}  # Keep track of file names for the logs

    for log, file_name in log_files:
        parameters = log["parameters"]
        match = all(parameters[k] == fixed_parameters[k] for k in fixed_parameters if k not in varying_parameters)
        #match2 = all(parameters[k] in varying_values_choice[k] for k in varying_parameters)
        if parameters["solver"] == "optimistic":
            match3 = True if parameters["constant"] == 0.1 else False
            match4 = True if parameters["alpha"] == 0.8 else False
            match2 = True if parameters["runs"] == 5 else False
        elif parameters["solver"] == "nash_ca":
            match3 = True if parameters["constant"] == 0.2 else False
            match4 = True if parameters["alpha"] == 0.1 else False
            match2 = True if parameters["runs"] == 50 else False
        else:
            match3 = False
            match4 = False
            match2 = False

        if (match and match2) and (match3 and match4):
            varying_values = tuple(parameters[param] for param in varying_parameters)
            if varying_values not in filtered_logs:
                filtered_logs[varying_values] = []
                file_names[varying_values] = []
            else:
                print("duplicate",file_name)
                print(parameters)
            filtered_logs[varying_values].append(log)
            file_names[varying_values].append(file_name)

    return filtered_logs

def load_data(varying_parameters, fixed_parameters,varying_values_choice):
    log_files = read_simulation_log_files()
    filtered_logs = filter_logs_by_2_parameters(log_files, fixed_parameters, varying_parameters,varying_values_choice)

    unique_varying_values = {param: set() for param in varying_parameters}
    for log, _ in log_files:  # Unpack the tuple to separate the log data and the file name
        for param in varying_parameters:
            if log["parameters"][param] in varying_values_choice[param]:
                unique_varying_values[param].add(log["parameters"][param])

    varying_values = [sorted(unique_varying_values[param]) for param in varying_parameters]
    data_ne = np.zeros(tuple(len(values) for values in varying_values))
    data_p = np.zeros(tuple(len(values) for values in varying_values))
    data_ni = np.zeros(tuple(len(values) for values in varying_values))

    for idx in np.ndindex(*data_ne.shape):

        print(idx)

        idx_varying_values = tuple(varying_values[i][idx[i]] for i in range(len(idx)))

        print(idx_varying_values)

        if idx_varying_values in filtered_logs:
            logs = filtered_logs[idx_varying_values]
            cumulative_nash_regret = [0,0,0]
            for log in logs:
                cumulative_nash_regret[0] += np.sum(log["nash"]) / log["parameters"]["runs"]
                cumulative_nash_regret[1] += np.sum(log["potential"]) / log["parameters"]["runs"]
                cumulative_nash_regret[2] += np.sum(log["nikaido_isoda"]) / log["parameters"]["runs"]

            data_ne[idx] = cumulative_nash_regret[0]
            data_p[idx] = cumulative_nash_regret[1]
            data_ni[idx] = cumulative_nash_regret[2]

    return varying_values, data_ne, data_p, data_ni

if __name__ == "__main__":
    varying_parameters = ["game","noise","solver"] # Specify the parameters you want to vary
    varying_values_choice = {
        "game": ["cooperative","mid_skewed","neg_skewed","pos_skewed","random","tailed_skewed"],
        "noise": [0.05, 0.1, 0.2],
        "solver": ["nash_ca","optimistic"]
    }
    fixed_parameters = {
        "dimension": 5,
        "timesteps": 5000,
        "players": 3
    }
    varying_values, data_ne, data_p, data_ni = load_data(varying_parameters, fixed_parameters,varying_values_choice)
    plot_bar_charts(varying_values_choice,data_p)
