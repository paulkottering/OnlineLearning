import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches


def plot_lineplot_with_errorbars(varying_values_choice, varying_parameters, varying_values, data_ne, data_p, data_ni,
                                 fixed_parameters):

    max_iteration = 250
    ylim = 15

    sns.set()
    fig, axes = plt.subplots(3, 3, figsize=(15, 7), sharey=True)
    x = np.arange(max_iteration)
    fmts = ['o-', 'x-', '*-']
    colors = ['royalblue', 'mediumblue', 'darkblue']

    for n in range(3):
        for p in range(3):
            for r in range(200):
                axes[n, p].plot(x, data_ne[p, n, r, :max_iteration], color=colors[p], linewidth=0.5, alpha=0.5)

            # Add x and y labels
            if n == 2:
                axes[n, p].set_xlabel('Iterations')
            if p == 0:
                axes[n, p].set_ylabel('')

            # Add column titles
            if n == 0:
                axes[n, p].set_title(f'Agents = {p + 2}')

            # Add white rectangle
            rect = patches.Rectangle((max_iteration, 0), 50, ylim,
                                     linewidth=0, facecolor='white', zorder=1)
            axes[n, p].add_patch(rect)

            # Add violin plots, axis=1)
            cumulative_regret = data_ne[p, n, :, max_iteration-1]
            viol = axes[n, p].violinplot(cumulative_regret, positions=[max_iteration+25], widths=[20], showmeans=True, showextrema=False,
                                  showmedians=False, quantiles=None)
            for pc in viol['bodies']:
                pc.set_facecolor(colors[p])
                pc.set_alpha(0.8)
            axes[n, p].set_xlim([0, max_iteration+50])
            axes[n, p].set_ylim([0, ylim])
            axes[n, p].set_xticks([0,100,200])

            # Add row titles
    noise_levels = ['0.05', '0.1', '0.2']
    for idx, noise_level in enumerate(noise_levels):
        fig.text(-0.03, 0.715 - idx * 0.3, f'Noise Level = {noise_level}', fontsize=12,rotation='vertical')

    # Add the y-axis label
    fig.text(0, 0.5, 'Cumulative Nash Regret', rotation='vertical', va='center', fontsize=12)

    # fixed_params_text = ', '.join(f'{k} = {v}' for k, v in fixed_parameters.items())
    # fig.text(0.5, -0.1, f"Fixed Parameters: {fixed_params_text}", ha='center', fontsize=12, wrap=True)

    plt.tight_layout()

    if not os.path.exists("Figures"):
        os.makedirs("Figures")
    title = "Figures/5_7.png".format(varying_parameters[0], varying_parameters[1])
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
    data_ne = np.zeros(tuple(len(values) for values in varying_values)+(200,500))
    data_p = np.zeros(tuple(len(values) for values in varying_values)+(200,500))
    data_ni = np.zeros(tuple(len(values) for values in varying_values)+(200,500))

    data = np.zeros(tuple(len(values) for values in varying_values))


    for idx in np.ndindex(*data.shape):
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

                for i in range(200):
                    a = log_data["nash"][i]
                    b = idx + (i,slice(None))
                    data_ne[b] = np.cumsum(log_data["nash"][i])
                    data_p[b] = np.cumsum(log_data["potential"][i])
                    data_ni[b] = np.cumsum(log_data["nikaido_isoda"][i])


    return varying_values, data_ne, data_p, data_ni


if __name__ == "__main__":
    varying_parameters = ["players", "noise"]
    varying_values_choice = {
        "players": [2,3,4],
        "noise": [0.05,0.1,0.2]}
    fixed_parameters = {
        "solver": "optimistic",
        "dimension": 3,
        "timesteps": 500,
        "runs": 200,
        "game": "random",
        "constant": None,
        "alpha": None
    }
    varying_values, data_ne, data_p, data_ni = load_data(varying_parameters, fixed_parameters, varying_values_choice)
    plot_lineplot_with_errorbars(varying_values_choice,varying_parameters, varying_values, data_ne, data_p, data_ni, fixed_parameters )