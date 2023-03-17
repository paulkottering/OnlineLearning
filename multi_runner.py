import argparse
import numpy as np
import matplotlib.pyplot as plt
from game import game

from utils.Nash import FindNash
from utils.sample_strategy import sample_index
from utils.selection_strategy import select_index
from utils.plotter import plot_one,plot_many
from utils.initial_strategy import initialize_game


def parse_args():
    """
    Helper function to set up command line argument parsing
    """
    parser = argparse.ArgumentParser()
    # Set up arguments
    parser.add_argument("-n", "--dimension", default=5, type=int,
                        help='Number of Strategies for each player')
    parser.add_argument("-k", "--players", default=5, type=int,
                        help='Number of Players')
    parser.add_argument("-sa", "--sample_strategy", default="pa", type=str,
                        help='sample_strategy',
                        choices=["pa","da"] )
    parser.add_argument("-si", "--initial_strategy", default="o", type=str,
                        help = 'Initial strategy',
                        choices=["d", "o"])
    parser.add_argument("-t", "--timesteps", default=100, type=int,
                        help='Number of timesteps')
    parser.add_argument("-r", "--runs", default=1, type=int,
                        help='Number of Runs')
    return parser.parse_args()

def main(**kwargs):
    # Dimension of Problem
    n = kwargs.get("dimension")
    k = kwargs.get("players")
    sa = kwargs.get("sample_strategy")
    si = kwargs.get("initial_strategy")
    t_max = kwargs.get("timesteps")
    runs = kwargs.get("runs")

    iterations = t_max

    regrets = []
    cumulative_regrets = []
    number_of_nonactive = []
    percent_sampled = []

    for r in range(runs):
        print(r)
        regrets.append([])
        cumulative_regrets.append([])
        number_of_nonactive.append([])
        percent_sampled.append([])

        Game = game(n,k)

        initialize_game(Game, si)
        cumulative_regret = 0
        for t in range(iterations):
            print("______")
            print(t)

            sample_tuple, prob = sample_index(Game, sa)

            print("Sample: ",sample_tuple)
            Game.sample(tuple(sample_tuple))


            #ind1, ind2 = select_index(Game, se)
            regrets[r].append(np.max(Game.UnknownGame) - np.sum(Game.UnknownGame * prob))
            print("Regret: ",np.max(Game.UnknownGame) - np.sum(Game.UnknownGame * prob))

            cumulative_regret += regrets[r][t]
            cumulative_regrets[r].append(cumulative_regret)
            print("Cumulative Regret: ",cumulative_regret)

            number_of_nonactive[r].append(np.count_nonzero(Game.OptPhi < np.max(Game.PesPhi)) / (n ** 2) * 100)

            percent_sampled[r].append(np.count_nonzero(np.isnan(Game.KnownUs[0]))/(n**k)*100)

            if not np.any(np.isnan(Game.KnownUs[0])):
                break

            Game.check_bounds()

            # Update time index
            t += 1
    print(cumulative_regrets)
    # create a figure and axis object
    plot_many(kwargs, regrets,np.array(cumulative_regrets), number_of_nonactive,percent_sampled, np.max(Game.UnknownGame)-(np.sum(Game.UnknownGame)/(n**k)))


if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)