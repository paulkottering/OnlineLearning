import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cumulative_regret(changing_parameter, changing_values, fixed_parameters, data_ne, data_p, std_ne, std_p):
    sns.set()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6),sharey=True)

    for idx, value in enumerate(changing_values):
        axes[0].plot(data_ne[idx], label=f"{changing_parameter} = {value}")
        axes[1].plot(data_p[idx], label=f"{changing_parameter} = {value}")

    axes[0].set_title('Nash Regret')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Cumulative Nash Regret')
    axes[0].legend()

    axes[1].set_title('Potential Regret')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Cumulative Potential Regret')
    axes[1].legend()


    plt.tight_layout()

    fixed_params_text = ', '.join(f'{k} = {v}' for k, v in fixed_parameters.items() if k != changing_parameter)
    fig.text(0.5, -0.1, f"Fixed Parameters: {fixed_params_text}", ha='center', fontsize=12, wrap=True)

    if not os.path.exists("cumulative_regrets"):
        os.makedirs("cumulative_regrets")
    title = f"cumulative_regrets/{changing_parameter}_cumulative_regret.png"
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

def filter_logs_by_1_parameter(log_files, fixed_parameters, varying_parameter, varying_values):
    filtered_logs = {}
    file_names = {}  # Keep track of file names for the logs

    for log, file_name in log_files:
        parameters = log["parameters"]

        match = all(parameters[k] == fixed_parameters[k] for k in fixed_parameters if k is not varying_parameter)

        if parameters["solver"] == "optimistic":
            match3 = True if parameters["constant"] == 0.1 else False
            match4 = True if parameters["alpha"] == 0.8 else False
            match2 = True if parameters["runs"] == 1 else False
        elif parameters["solver"] == "nash_ca":
            match3 = True if parameters["constant"] == 0.2 else False
            match4 = True if parameters["alpha"] == 0.1 else False
            match2 = True if parameters["runs"] == 1 else False
        elif parameters["solver"] == "exp_weight":
            match3 = True if parameters["constant"] == 0.4 else False
            match4 = True if parameters["alpha"] == 0.5 else False
            match2 = True if parameters["runs"] == 1 else False
        elif parameters["solver"] == "nash_ucb":
            match3 = True if parameters["constant"] == 0.1 else False
            match4 = True if parameters["alpha"] == 0.8 else False
            match2 = True if parameters["runs"] == 1 else False
        else:
            match3 = False
            match4 = False
            match2 = False

        if (match and match2) and (match3 and match4):
            print(match,match2)
            print(parameters)
            varying_value = parameters[varying_parameter]

            if varying_value in filtered_logs:
                print("duplicate", file_name)
                print(parameters)

            filtered_logs[varying_value] = []
            file_names[varying_value] = []

            filtered_logs[varying_value].append(log)
            file_names[varying_value].append(file_name)


    return filtered_logs

def load_data(changing_parameter, fixed_parameters, changing_values):
    log_files = read_simulation_log_files()
    filtered_logs = filter_logs_by_1_parameter(log_files, fixed_parameters, changing_parameter, changing_values)

    data_ne = []
    data_p = []

    std_ne = []
    std_p = []

    for value in changing_values:
        if value in filtered_logs:
            logs = filtered_logs[value]

            cumulative_nash_regret = np.mean(np.cumsum(logs[0]["nash"],axis=1),axis=0)
            std_list = np.std(np.cumsum(logs[0]["nash"],axis=1),axis=0)

            cumulative_potential_regret = np.mean(np.cumsum(logs[0]["potential"],axis=1),axis=0)
            std_p_list = np.std(np.cumsum(logs[0]["potential"],axis=1),axis=0)

            data_ne.append(cumulative_nash_regret)
            data_p.append(cumulative_potential_regret)

            std_ne.append(std_list)
            std_p.append(std_p_list)

    return data_ne, data_p, std_ne, std_p


if __name__ == "__main__":
    varying_parameter = "solver" # Specify the parameters you want to vary
    varying_values_choice =  ["nash_ca","optimistic","exp_weight"]
    fixed_parameters = {
        "dimension": 5,
        "timesteps": 2000,
        "noise": 0.1,
        "players": 2,
        "game": "random"
    }
    data_ne, data_p, std_ne, std_p = load_data(varying_parameter, fixed_parameters, varying_values_choice)
    plot_cumulative_regret(varying_parameter, varying_values_choice, fixed_parameters, data_ne, data_p, std_ne, std_p)
