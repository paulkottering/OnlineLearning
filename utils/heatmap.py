import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(varying_parameters, varying_values, data_ne, data_p, data_ni, fixed_parameters):
    sns.set()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(data_ne, annot=True, fmt=".2f", xticklabels=varying_values[1], yticklabels=varying_values[0],
                ax=axes[0])
    axes[0].set_xlabel(varying_parameters[1])
    axes[0].set_ylabel(varying_parameters[0])
    axes[0].set_title('Nash Regret')

    sns.heatmap(data_p, annot=True, fmt=".2f", xticklabels=varying_values[1], yticklabels=varying_values[0], ax=axes[1])
    axes[1].set_xlabel(varying_parameters[1])
    axes[1].set_ylabel(varying_parameters[0])
    axes[1].set_title('Potential Regret')

    sns.heatmap(data_ni, annot=True, fmt=".2f", xticklabels=varying_values[1], yticklabels=varying_values[0],
                ax=axes[2])
    axes[2].set_xlabel(varying_parameters[1])
    axes[2].set_ylabel(varying_parameters[0])
    axes[2].set_title('Nikaido-Isoda Regret')

    plt.tight_layout()

    fixed_params_text = ', '.join(f'{k} = {v}' for k, v in fixed_parameters.items())
    fig.text(0.5, -0.1, f"Fixed Parameters: {fixed_params_text}", ha='center', fontsize=12, wrap=True)

    if not os.path.exists("heatmaps"):
        os.makedirs("heatmaps")
    title = "heatmaps/{}_{}_heatmap.png".format(varying_parameters[0], varying_parameters[1])
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

def filter_logs_by_2_parameters(log_files, fixed_parameters, varying_parameters):
    filtered_logs = {}
    file_names = {}  # Keep track of file names for the logs

    for log, file_name in log_files:
        parameters = log["parameters"]
        match = all(parameters[k] == fixed_parameters[k] for k in fixed_parameters if k not in varying_parameters)

        if match:
            varying_values = tuple(parameters[param] for param in varying_parameters)
            if varying_values not in filtered_logs:
                filtered_logs[varying_values] = []
                file_names[varying_values] = []

            if log in filtered_logs[varying_values]:  # Check for duplicate logs
                existing_file_name = file_names[varying_values][filtered_logs[varying_values].index(log)]
                raise ValueError(f"Duplicate logs found: '{file_name}' and '{existing_file_name}' have the same varying parameters.")

            filtered_logs[varying_values].append(log)
            file_names[varying_values].append(file_name)

    return filtered_logs

def load_data(varying_parameters, fixed_parameters):
    log_files = read_simulation_log_files()
    filtered_logs = filter_logs_by_2_parameters(log_files, fixed_parameters, varying_parameters)

    unique_varying_values = {param: set() for param in varying_parameters}
    for log, _ in log_files:  # Unpack the tuple to separate the log data and the file name
        for param in varying_parameters:
            unique_varying_values[param].add(log["parameters"][param])

    varying_values = [sorted(unique_varying_values[param]) for param in varying_parameters]
    data_ne = np.zeros(tuple(len(values) for values in varying_values))
    data_p = np.zeros(tuple(len(values) for values in varying_values))
    data_ni = np.zeros(tuple(len(values) for values in varying_values))

    for idx in np.ndindex(*data_ne.shape):
        idx_varying_values = tuple(varying_values[i][idx[i]] for i in range(len(idx)))

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
    varying_parameters = ["constant", "alpha"]  # Specify the parameters you want to vary
    fixed_parameters = {
        "dimension": 10,
        "players": 3,
        "timesteps": 10000,
        "runs": 30,
        "noise": 0.1,
        "solver": "nash_ca",
        "game": "random",
    }
    varying_values, data_ne, data_p, data_ni = load_data(varying_parameters, fixed_parameters)
    plot_heatmap(varying_parameters, varying_values, data_ne, data_p, data_ni, fixed_parameters)
