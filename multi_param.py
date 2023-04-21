import argparse
from itertools import product
from runner import main as run_simulation
from utils.compare_plot import main as plot_comparison


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--varying", nargs='+', default=["noise", "constant"],
                        help='List of parameters to vary')
    parser.add_argument("-n", "--dimension", default=5, type=int,
                        help='Number of Strategies for each player')
    parser.add_argument("-k", "--players", default=2, type=int,
                        help='Number of Players')
    parser.add_argument("-t", "--timesteps", default=3000, type=int,
                        help='Number of timesteps')
    parser.add_argument("-r", "--runs", default=3, type=int,
                        help='Number of Runs')
    parser.add_argument("-nl", "--noise", default=0.1, type=float,
                        help='Noise Level')
    parser.add_argument("-c", "--constant", default=0.1, type=float,
                        help='Constant')
    parser.add_argument("-a", "--alpha", default=0.8, type=float,
                        help='Alpha')
    parser.add_argument("-g", "--game", default="congestion", type=str,
                        help='Game Type')
    parser.add_argument("-s", "--solver", default="nash_ucb", type=str,
                        help='Which solver to use')
    return parser.parse_args()


def main(**kwargs):
    varying_parameters = kwargs.get("varying")

    param_values = {
        "noise": [0.1],
        "constant": [0.1,0.2,0.5,1,10],
        # Add more parameter options as needed
    }

    # Get the varying values for each parameter
    varying_values = [param_values[param] for param in varying_parameters]

    # Run simulations for each combination of varying parameter values
    for values in product(*varying_values):
        print(
            f"Running simulation with {', '.join(f'{param} = {value}' for param, value in zip(varying_parameters, values))}")
        kwargs.update(dict(zip(varying_parameters, values)))
        run_simulation(**kwargs)


if __name__ == "__main__":
    args = vars(parse_args())
    main(**args)

