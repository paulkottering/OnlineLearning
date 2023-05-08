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
        'mid_skewed': 'Symmetrical Unimodal',
        'cooperative': 'Cooperative'
    }
    bar_positions = [0 - bar_width, 1, 2 + bar_width]
    for row in range(2):
        for col in range(3):
            for index in range(3):
                bar_position = bar_positions[index]
                for i in range(2):
                    bp = axes[row, col].boxplot(
                        data[3 * row + col][index][i],
                        positions=[bar_position+i*0.35],
                        widths=bar_width,
                        patch_artist=True,
                        flierprops={'color': colors[i]},
                        medianprops={'color': 'black'}
                    )
                    for patch in bp["boxes"]:
                        patch.set_facecolor(colors[i])
                    plt.setp(bp['fliers'], color=colors[i], markersize=7, markerfacecolor=colors[i])

            # Set the title above the graph and increase the font size
            axes[row, col].set_title(title_mapping[varying_values_choice["game"][3 * row + col]], fontsize=20)
            axes[row, col].title.set_position([.5, 0.02])
            index = np.arange(3)
            print(index + bar_width / 2)
            # Set x-axis labels with larger font size
            axes[row, col].set_xticks(index + bar_width / 2)
            axes[row, col].set_xticklabels(x_labels, fontsize=14)

            # Set y-axis labels with larger font size
            axes[row, col].tick_params(axis='y', which='major', labelsize=14)

            # Set x-axis label for all subplots
            axes[row, col].set_xlabel('Noise Level', fontsize=16)

            # Set y-axis label for all subplots
            axes[row, col].set_ylabel('Cumulative Potential Regret', fontsize=16)

    # Create a custom legend at the top of the graph
    legend_elements = [
        plt.Line2D([0], [0], color=colors[i], lw=4, label=varying_values_choice["solver"][i])
        for i in range(2)
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.05),
               fontsize=14)

    plt.tight_layout()
    if not os.path.exists("Figures"):
        os.makedirs("Figures")
    title = f"Figures/5_6.png"
    plt.savefig(title, bbox_inches="tight")
    print("Plot saved to", title)


def read_simulation_log_files(log_file="simulation_log.txt"):
    log_files = []

    with open(log_file, "r") as file:
        for line in file:
            log_files.append(json.loads(line.strip()))
    return log_files


def filter_logs_by_2_parameters(log_entries, fixed_parameters, varying_parameters, varying_values_choice):
    filtered_logs = {}
    file_names = {}  # Keep track of file names for the logs

    for entry in log_entries:
        parameters = entry["parameters"]
        match = all(parameters[k] == fixed_parameters[k] for k in fixed_parameters if k not in varying_parameters)
        match2 = all(parameters[k] in varying_values_choice[k] for k in varying_parameters)
        if parameters["solver"] == "optimistic":
            match3 = True if (parameters["constant"] == 0.1) or (parameters["constant"] == None) else False
            match4 = True if parameters["alpha"] == 0.8 or (parameters["alpha"] == None) else False
        elif parameters["solver"] == "nash_ca":
            match3 = True if parameters["constant"] == 0.2 or (parameters["constant"] == None) else False
            match4 = True if parameters["alpha"] == 0.1 or (parameters["alpha"] == None) else False
        if (match and match2) and (match3 and match4):
            varying_values = tuple(parameters[param] for param in varying_parameters)
            if varying_values not in filtered_logs:
                filtered_logs[varying_values] = []
                file_names[varying_values] = []
                filtered_logs[varying_values].append(entry)
                file_names[varying_values].append("log_files/simulation_{}.json".format(entry["timestamp"]))
            else:
                filtered_logs[varying_values].append(entry)
                file_names[varying_values].append("log_files/simulation_{}.json".format(entry["timestamp"]))
                print("duplicate", entry["timestamp"])
                print(parameters)

    return filtered_logs, file_names


def load_data(varying_parameters, fixed_parameters, varying_values_choice):
    log_entries = read_simulation_log_files()
    filtered_logs, file_names = filter_logs_by_2_parameters(log_entries, fixed_parameters, varying_parameters,
                                                            varying_values_choice)

    unique_varying_values = {param: set() for param in varying_parameters}
    for log in log_entries:
        for param in varying_parameters:
            if log["parameters"][param] in varying_values_choice[param]:
                unique_varying_values[param].add(log["parameters"][param])

    varying_values = [sorted(unique_varying_values[param]) for param in varying_parameters]
    dimensions = [len(values) for values in varying_values]
    data_ne = create_k_dimensional_array(dimensions)
    data_p = create_k_dimensional_array(dimensions)
    data_ni = create_k_dimensional_array(dimensions)
    std_ne = np.zeros(tuple(len(values) for values in varying_values))


    for idx in np.ndindex(*std_ne.shape):
        idx_varying_values = tuple(varying_values[i][idx[i]] for i in range(len(idx)))

        if idx_varying_values in filtered_logs:
            log_timestamps = file_names[idx_varying_values]

            cumulative_nash_regret = [0, 0, 0]
            for timestamp in log_timestamps:
                file_path = f"{timestamp}"
                if os.path.exists(file_path):
                    with open(file_path, "r") as file:
                        log_data = json.load(file)
                else:
                    print(timestamp)
                print(log_data["parameters"])

                data_ne[idx[0]][idx[1]][idx[2]].extend(np.sum(log_data["nash"],axis=1).tolist())
                data_p[idx[0]][idx[1]][idx[2]].extend(np.sum(log_data["potential"],axis=1).tolist())
                data_ni[idx[0]][idx[1]][idx[2]].extend(np.sum(log_data["nikaido_isoda"],axis=1).tolist())

    return varying_values, data_ne, data_p, data_ni

def create_k_dimensional_array(dimensions):
    if len(dimensions) == 1:
        return [ [] for _ in range(dimensions[0]) ]
    else:
        return [ create_k_dimensional_array(dimensions[1:]) for _ in range(dimensions[0]) ]

if __name__ == "__main__":
    varying_parameters = ["game", "noise", "solver"]  # Specify the parameters you want to vary
    varying_values_choice = {
        "game": ["cooperative", "mid_skewed", "neg_skewed", "pos_skewed", "random", "tailed_skewed"],
        "noise": [0.05, 0.1, 0.2],
        "solver": ["nash_ca", "optimistic"]
    }
    fixed_parameters = {
        "dimension": 5,
        "timesteps": 5000,
        "players": 3
    }
    varying_values, data_ne, data_p, data_ni = load_data(varying_parameters, fixed_parameters, varying_values_choice)
    plot_bar_charts(varying_values_choice, data_p)
