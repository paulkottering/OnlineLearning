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

    Regrets = np.empty((runs,iterations))
    CumRegrets = np.empty((runs, iterations))
    Nash = np.empty((runs,iterations))
    Gaps = np.empty((runs,iterations))
    PercentBoundedPhi = np.empty((runs,iterations))
    percent_sampled = np.empty((runs,iterations))
    print(k)

    for r in range(runs):
        print(r)

        Game = game(n,k)

        initialize_game(Game, si)

        for t in range(iterations):
            print("______")
            print(t)

            sample_tuple, prob = sample_index(Game, sa)

            print(sample_tuple)
            Game.sample(tuple(sample_tuple))


            #ind1, ind2 = select_index(Game, se)
            Regrets[r,t] = np.max(Game.UnknownGame) - np.sum(Game.UnknownGame * prob)
            Gaps[r,t] = (np.count_nonzero(Game.OptPhi < np.max(Game.PesPhi)) / (n ** 2) * 100)

            #PercentBoundedPhi[r,t] = np.count_nonzero(Game.OptPhi < np.max(Game.UnknownGame)) / (n ** 2) * 100
            percent_sampled[r,t] = np.count_nonzero(np.isnan(Game.KnownUs[0]))/(n**k)*100

            if not np.any(np.isnan(Game.KnownUs[0])):
                break

            #Game.check_bounds()

            # Update time index
            t += 1

        CumRegrets[r, :] = np.cumsum(Regrets[r,:])

    # create a figure and axis object
    plot_many(kwargs, Regrets,CumRegrets, Gaps,percent_sampled)


if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)