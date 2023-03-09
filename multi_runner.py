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
    parser.add_argument("-sa", "--sample_strategy", default="or", type=str,
                        help='sample_strategy',
                        choices=["d", "ra", "nr", "r", "or", "lr","pr","pm","pa"] )
    parser.add_argument("-se", "--selection_strategy", default="o", type=str,
                        help='Selection strategy',
                        choices=["o", "p"] )
    parser.add_argument("-si", "--initial_strategy", default="d", type=str,
                        help = 'Initial strategy',
                        choices=["d", "o"])
    parser.add_argument("-t", "--timesteps", default=100, type=int,
                        help='Number of timesteps')
    parser.add_argument("-k", "--optimismconstant", default=0.5, type=float,
                        help='Optimistic Constant')
    parser.add_argument("-r", "--runs", default=5, type=int,
                        help='Number of Runs')
    return parser.parse_args()

def main(**kwargs):
    # Dimension of Problem
    n = kwargs.get("dimension")
    sa = kwargs.get("sample_strategy")
    se = kwargs.get("selection_strategy")
    si = kwargs.get("initial_strategy")
    t_max = kwargs.get("timesteps")
    runs = kwargs.get("runs")

    if si == "d":
        iterations = n**2 - n + 1
    elif si == "o":
        iterations = n ** 2 + 1
    iterations = t_max

    Regrets = np.empty((runs,iterations))
    Nash = np.empty((runs,iterations))
    Gaps = np.empty((runs,iterations))
    PercentBoundedPhi = np.empty((runs,iterations))
    percent_sampled = np.empty((runs,iterations))
    for r in range(runs):
        print(r)

        Game = game(n)
        NashIndices = FindNash(Game)

        initialize_game(Game, si)

        for t in range(iterations):
            print("______")
            print(t)

            i, j, prob = sample_index(Game, sa)

            print(i, j)
            Game.sample(i, j)


            #ind1, ind2 = select_index(Game, se)
            Regrets[r,t] = np.max(Game.UnknownGame) - np.sum(Game.UnknownGame * prob)
            Gaps[r,t] = (np.count_nonzero(Game.OptPhi < np.max(Game.PesPhi)) / (n ** 2) * 100)

            if [i, j] in NashIndices:
                Nash[r,t] = 100
            else:
                Nash[r,t] = 0

            PercentBoundedPhi[r,t] = np.count_nonzero(Game.OptPhi < np.max(Game.UnknownGame)) / (n ** 2) * 100
            percent_sampled[r,t] = np.count_nonzero(np.isnan(Game.KnownU1))/(n**2)*100

            if not np.any(np.isnan(Game.KnownU1)):
                break

            #Game.check_bounds()

            # Update time index
            t += 1

        Regrets[r, :] = np.cumsum(Regrets[r,:])

    # create a figure and axis object
    plot_many(kwargs, Regrets, Nash, Gaps, PercentBoundedPhi,percent_sampled)


if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)