import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cumulative_regret(changing_parameter, changing_values, fixed_parameters, data_ne, data_p, data_ni):
    sns.set()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),sharey=True)
    colors = ['orange', 'blue','green']

    for i in range(3):
        axes[0].plot(np.cumsum(data_ne[i][0]),color=colors[i])
        axes[1].plot(np.cumsum(data_p[i][0]),color=colors[i])
        axes[2].plot(np.cumsum(data_ni[i][0]),color=colors[i])

        axes[i].set_xlim([0, 1000])
        axes[i].set_xticks([0, 200, 400,600,800,1000])

    axes[0].set_title('Nash Regret')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Cumulative Regret')

    axes[1].set_title('Potential Regret')
    axes[1].set_xlabel('Iterations')

    axes[2].set_title('Nikaido-Isoda Regret')
    axes[2].set_xlabel('Iterations')


    # fixed_params_text = ', '.join(f'{k} = {v}' for k, v in fixed_parameters.items() if k != changing_parameter)
    # fig.text(0.5, -0.1, f"Fixed Parameters: {fixed_params_text}", ha='center', fontsize=12, wrap=True)

    legend_elements = [
        plt.Line2D([0], [0], color=colors[i], lw=4, label=varying_values_choice[i])
        for i in range(3)
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.05),
               fontsize=14)


    if not os.path.exists("Figures"):
        os.makedirs("Figures")
    title = f"Figures/5_1.png"
    plt.savefig(title, bbox_inches="tight")
    print("Plot saved to", title)

def read_simulation_log_files(log_file="simulation_log.txt"):
    log_files = []

    with open(log_file, "r") as file:
        for line in file:
            log_files.append(json.loads(line.strip()))
    return log_files

def filter_logs_by_1_parameter(log_files, fixed_parameters, varying_parameter, varying_values):
    filtered_logs = []
    file_names = []  # Keep track of file names for the logs

    for log in log_files:
        parameters = log["parameters"]

        match = all(parameters[k] == fixed_parameters[k] for k in fixed_parameters if k is not varying_parameter)
        match2 = parameters[varying_parameter] in varying_values

        if match and match2:
            print(match,match2)
            varying_value = parameters[varying_parameter]


            filtered_logs.append(log)
            file_names.append("log_files/simulation_{}.json".format(log["timestamp"]))


    return filtered_logs,file_names

def load_data(changing_parameter, fixed_parameters, changing_values):
    log_files = read_simulation_log_files()
    filtered_logs,file_names = filter_logs_by_1_parameter(log_files, fixed_parameters, changing_parameter, changing_values)

    data_ne = []
    data_p = []
    data_ni = []

    for i,log in enumerate(file_names):
        timestamp = log
        file_path = f"{timestamp}"
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                log_data = json.load(file)
        else:
            print(timestamp)
        print(log_data["parameters"])
        data_ne.append(log_data['nash'])
        data_p.append(log_data['potential'])
        data_ni.append(log_data['nikaido_isoda'])


    return data_ne, data_p, data_ni


if __name__ == "__main__":
    varying_parameter = "solver" # Specify the parameters you want to vary
    varying_values_choice =  ["nash_ca","optimistic","exp_weight"]
    fixed_parameters = {
        "dimension": 5,
        "timesteps": 1000,
        "runs": 1,
        "noise": 0.1,
        "players": 3,
        "game": "random",
        "constant": None,
        "alpha": None
    }
    data_ne, data_p, data_ni  = load_data(varying_parameter, fixed_parameters, varying_values_choice)
    plot_cumulative_regret(varying_parameter, varying_values_choice, fixed_parameters, data_ne, data_p, data_ni)
