import argparse
from itertools import product
from runner import main as run_simulation
from utils.compare_plot import main as plot_comparison

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--varying", default="solver",
                        help='Which parameter to vary')
    parser.add_argument("-n", "--dimension", default=8, type=int,
                        help='Number of Strategies for each player')
    parser.add_argument("-k", "--players", default=2, type=int,
                        help='Number of Players')
    parser.add_argument("-t", "--timesteps", default=1000, type=int,
                        help='Number of timesteps')
    parser.add_argument("-r", "--runs", default=20, type=int,
                        help='Number of Runs')
    parser.add_argument("-nl", "--noise", default=0.1, type=float,
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

def main(**kwargs):
    varying_parameter = kwargs.get("varying")
    varying_values = ["nash_ca","optimistic"]

    # Run simulations for each varying parameter value
    for value in varying_values:
        print(f"Running simulation with {varying_parameter} = {value}")
        kwargs[varying_parameter] = value
        run_simulation(**kwargs)


if __name__ == "__main__":
    args = vars(parse_args())
    main(**args)
