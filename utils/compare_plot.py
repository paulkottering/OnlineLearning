import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set fixed parameters and the varying parameter
fixed_parameters = {
    "runs": 3,
    "players": 2,
    "timesteps": 200,
    "noise": 0.3,
    "constant": 0.2,
    "alpha": 0.1,
    "game": "congestion",
    "solver": "nash_ca",
    "regret": "nash"
}

varying_parameter = "dimension"

def read_simulation_log_files(folder="log_files"):
    log_files = []

    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), "r") as file:
                data = json.load(file)
                log_files.append(data)

    return log_files

def filter_logs_by_parameters(log_files, fixed_parameters, varying_parameter):
    filtered_logs = {}

    for log in log_files:
        parameters = log["parameters"]
        match = all(parameters[k] == fixed_parameters[k] for k in fixed_parameters if k != varying_parameter)

        if match:
            varying_value = parameters[varying_parameter]
            if varying_value not in filtered_logs:
                filtered_logs[varying_value] = []

            filtered_logs[varying_value].append(log)

    return filtered_logs

def plot_cumulative_regrets(filtered_logs, varying_parameter):
    plt.figure()

    for varying_value, logs in filtered_logs.items():
        for log in logs:
            regrets_av = np.cumsum(np.mean(log["regrets"],axis=0))
            plt.plot(regrets_av, label="{}={}".format(varying_parameter, varying_value))

    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.title("Cumulative Regret Comparison")

    title = "yoo"
    plt.title(title)
    # set the title of the plot
    plt.savefig(title)

def main():
    log_files = read_simulation_log_files()
    filtered_logs = filter_logs_by_parameters(log_files, fixed_parameters, varying_parameter)
    plot_cumulative_regrets(filtered_logs, varying_parameter)

if __name__ == "__main__":
    main()
