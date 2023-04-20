import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():
    """
    Helper function to set up command line argument parsing
    """
    parser = argparse.ArgumentParser()
    # Set up arguments
    parser.add_argument("-v", "--varying", default="solver", type=str,
                        help='varying parameter')
    parser.add_argument("-n", "--dimension", default=5, type=int,
                        help='Number of Strategies for each player')
    parser.add_argument("-k", "--players", default=3, type=int,
                        help='Number of Players')
    parser.add_argument("-t", "--timesteps", default=1000, type=int,
                        help='Number of timesteps')
    parser.add_argument("-r", "--runs", default=10, type=int,
                        help='Number of Runs')
    parser.add_argument("-e", "--regret", default="nash", type=str,
                        help='Type of Regret')
    parser.add_argument("-nl", "--noise", default=0.05, type=float,
                        help='Noise Level')
    parser.add_argument("-c", "--constant", default=0.2, type=float,
                        help='Constant')
    parser.add_argument("-a", "--alpha", default=0.5, type=float,
                        help='Alpha')
    parser.add_argument("-g", "--game", default="skewed", type=str,
                        help='Game Type')
    parser.add_argument("-s", "--solver", default="optimistic", type=str,
                        help='Which solver to use')

    return parser.parse_args()

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

def plot_cumulative_regrets(filtered_logs, varying_parameter,fixed_parameters,regret, std_width=1):
    plt.figure()

    for varying_value, logs in filtered_logs.items():
        for log in logs:
            all_regrets = np.cumsum(log[regret], axis=1)
            mean_regrets = np.mean(all_regrets, axis=0)
            std_regrets = np.std(all_regrets, axis=0)
            plt.plot(mean_regrets, label="{}={}".format(varying_parameter, varying_value))
            plt.fill_between(np.arange(len(mean_regrets)), mean_regrets - std_width * std_regrets,
                             mean_regrets + std_width * std_regrets, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative {} Regret Comparison ({})".format(regret, varying_parameter))

    # add legend outside plot on right
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    # add fixed parameter values to plot
    fixed_str = "\n".join(["{}: {}".format(k, v) for k, v in fixed_parameters.items()])
    plt.text(1.05, 0.2, fixed_str, transform=plt.gcf().transFigure)

    if not os.path.exists("compare_figures"):
        os.makedirs("compare_figures")
    title = "compare_figures/{}.png".format(varying_parameter)
    plt.savefig(title, bbox_inches="tight")
    print("Plot saved to", title)

def main(**kwargs):

    # Set fixed parameters and the varying parameter
    fixed_parameters = {
        "runs": kwargs.get("runs"),
        "players": kwargs.get("players"),
        "timesteps": kwargs.get("timesteps"),
        "noise": kwargs.get("noise"),
        "constant": kwargs.get("constant"),
        "alpha": kwargs.get("alpha"),
        "game": kwargs.get("game"),
        "solver": kwargs.get("solver"),
        "dimension": kwargs.get("dimension")
    }

    varying_parameter = kwargs.get("varying")
    regret = "nash"
    log_files = read_simulation_log_files()
    filtered_logs = filter_logs_by_parameters(log_files, fixed_parameters, varying_parameter)
    plot_cumulative_regrets(filtered_logs, varying_parameter,fixed_parameters,regret)


if __name__ == "__main__":
    """
    Entry point for the script. Parses command line arguments and runs the main function.
    """
    # Parse command line arguments
    args = vars(parse_args())
    # Execute the main function with the parsed arguments
    main(**args)


