import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(varying_parameters, varying_values, data_ne, data_p, data_ni, fixed_parameters):
    sns.set()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(data_ne, annot=True, fmt=".2f", xticklabels=varying_values[1], yticklabels=varying_values[0],
                ax=axes[0],cmap='Oranges')
    axes[0].set_xlabel("kappa")
    axes[0].set_ylabel("c_1")
    axes[0].set_title('Nash Regret')

    sns.heatmap(data_p, annot=True, fmt=".2f", xticklabels=varying_values[1], yticklabels=varying_values[0], ax=axes[1],cmap='Oranges')
    axes[1].set_xlabel("kappa")
    axes[1].set_ylabel("c_1")
    axes[1].set_title('Potential Regret')

    sns.heatmap(data_ni, annot=True, fmt=".2f", xticklabels=varying_values[1], yticklabels=varying_values[0],
                ax=axes[2],cmap='Oranges')
    axes[2].set_xlabel("kappa")
    axes[2].set_ylabel("c_1")
    axes[2].set_title('Nikaido-Isoda Regret')

    plt.tight_layout()

    # fixed_params_text = ', '.join(f'{k} = {v}' for k, v in fixed_parameters.items())
    # fig.text(0.5, -0.1, f"Fixed Parameters: {fixed_params_text}", ha='center', fontsize=12, wrap=True)


    if not os.path.exists("Figures"):
        os.makedirs("Figures")
    title = "Figures/5_2.png".format(varying_parameters[0], varying_parameters[1])
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
        if match and match2:
            print(parameters)
            varying_values = tuple(parameters[param] for param in varying_parameters)
            if varying_values not in filtered_logs:
                filtered_logs[varying_values] = []
                file_names[varying_values] = []
                filtered_logs[varying_values].append(entry)
                file_names[varying_values].append("log_files/simulation_{}.json".format(entry["timestamp"]))
            else:
                print("duplicate", entry["timestamp"])
                print(parameters)

    return filtered_logs, file_names

def load_data(varying_parameters, fixed_parameters, varying_values_choice):
    log_entries = read_simulation_log_files()
    filtered_logs, file_names = filter_logs_by_2_parameters(log_entries, fixed_parameters, varying_parameters, varying_values_choice)

    unique_varying_values = {param: set() for param in varying_parameters}
    for log in log_entries:
        for param in varying_parameters:
            if log["parameters"][param] in varying_values_choice[param]:
                unique_varying_values[param].add(log["parameters"][param])

    varying_values = [sorted(unique_varying_values[param]) for param in varying_parameters]
    data_ne = np.zeros(tuple(len(values) for values in varying_values))
    data_p = np.zeros(tuple(len(values) for values in varying_values))
    data_ni = np.zeros(tuple(len(values) for values in varying_values))

    std_ne = np.zeros(tuple(len(values) for values in varying_values))
    std_p = np.zeros(tuple(len(values) for values in varying_values))
    std_ni = np.zeros(tuple(len(values) for values in varying_values))

    for idx in np.ndindex(*data_ne.shape):
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

                cumulative_nash_regret[0] += np.sum(log_data["nash"]) / log_data["parameters"]["runs"]
                cumulative_nash_regret[1] += np.sum(log_data["potential"]) / log_data["parameters"]["runs"]
                cumulative_nash_regret[2] += np.sum(log_data["nikaido_isoda"]) / log_data["parameters"]["runs"]

                std_ne[idx] = np.std(np.sum(log_data["nash"],axis=1))
                std_p[idx] = np.std(np.sum(log_data["potential"],axis=1))
                std_ni[idx] = np.std(np.sum(log_data["nikaido_isoda"],axis=1))

                data_ne[idx] = cumulative_nash_regret[0]
                data_p[idx] = cumulative_nash_regret[1]
                data_ni[idx] = cumulative_nash_regret[2]

    return varying_values, data_ne, data_p, data_ni


if __name__ == "__main__":
    varying_parameters = ["constant", "alpha"]
    varying_values_choice = {
        "constant": [0.1,0.2,1,2],
        "alpha": [0.001,0.05,0.1,0.2,0.5]}
    fixed_parameters = {
        "dimension": 5,
        "timesteps": 5000,
        "runs": 5,
        "solver": "nash_ca",
        "game": "random",
        "players": 3,
        "noise": 0.2
    }
    varying_values, data_ne, data_p, data_ni = load_data(varying_parameters, fixed_parameters,varying_values_choice)
    plot_heatmap(varying_parameters, varying_values, data_ne, data_p, data_ni, fixed_parameters)
