import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_lineplot_with_errorbars(varying_parameters, varying_values, data_ne, data_p, data_ni, fixed_parameters,std_ne,std_p,std_ni):
    sns.set()
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharey='row')

    fmts = ['o-','x-','*-']
    colors = ['green','orange']
    for i in range(3):
        noise = varying_values[1][i]
        for s in range(2):
            axes[0,i].errorbar(varying_values[0], data_ne[:, i, s], yerr=std_ne[:, i, s], fmt=fmts[s], capsize=5, label= varying_values[2][s], color = colors[s])
            axes[1,i].errorbar(varying_values[0], data_p[:, i, s], yerr=std_p[:, i, s], fmt=fmts[s], capsize=5, label= varying_values[2][s], color = colors[s])
            axes[2,i].errorbar(varying_values[0], data_ni[:, i, s], yerr=std_ni[:, i, s], fmt=fmts[s], capsize=5, label= varying_values[2][s], color = colors[s])
            axes[2, i].set_xticks(varying_values[0])
            axes[2, i].set_xticklabels(varying_values[0])
            axes[1, i].set_xticks(varying_values[0])
            axes[1, i].set_xticklabels(varying_values[0])
            axes[0, i].set_xticks(varying_values[0])
            axes[0, i].set_xticklabels(varying_values[0])
        axes[2,i].set_xlabel('Agents')
        axes[0,i].set_title(f'Noise = {noise}')
    axes[0, 0].set_axis_on()
    axes[1, 0].set_axis_on()
    axes[2, 0].set_axis_on()

    axes[0,0].set_ylabel('Cumulative Nash Regret')
    axes[1,0].set_ylabel('Cumulative Potential Regret')
    axes[2,0].set_ylabel('Cumulative Nikaido-Isoda Regret')

    # fixed_params_text = ', '.join(f'{k} = {v}' for k, v in fixed_parameters.items())
    # fig.text(0.5, -0.1, f"Fixed Parameters: {fixed_params_text}", ha='center', fontsize=12, wrap=True)

    legend_elements = [
        plt.Line2D([0], [0],lw=4,color=colors[i], label=varying_values_choice["solver"][i])
        for i in range(2)
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.05),
               fontsize=12)
    plt.tight_layout()

    if not os.path.exists("Figures"):
        os.makedirs("Figures")
    title = "Figures/5_8.png".format(varying_parameters[0], varying_parameters[1])
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
            else:
                print("duplicate", entry["timestamp"])
                print(parameters)
            filtered_logs[varying_values].append(entry)
            file_names[varying_values].append("log_files/simulation_{}.json".format(entry["timestamp"]))

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

    return varying_values, data_ne, data_p, data_ni,std_ne,std_p,std_ni


if __name__ == "__main__":
    varying_parameters = ["players", "noise","solver"]
    varying_values_choice = {
        "players": [2,3,4,5,6,7,8,9],
        "noise": [0.05,0.1,0.2],
        "solver": ["exp_weight","nash_ca"]}

    fixed_parameters = {
        "dimension": 3,
        "timesteps": 5000,
        "runs": 5,
        "game": "random",
        "constant":None,
        "alpha": None
    }
    varying_values, data_ne, data_p, data_ni,std_ne,std_p,std_ni = load_data(varying_parameters, fixed_parameters, varying_values_choice)
    plot_lineplot_with_errorbars(varying_parameters, varying_values, data_ne, data_p, data_ni, fixed_parameters,std_ne,std_p,std_ni )