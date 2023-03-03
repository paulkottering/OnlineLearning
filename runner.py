import argparse
import numpy as np
import matplotlib.pyplot as plt
from game import game

from utils.Nash import FindNash
from utils.sample_strategy import sample_index
from utils.selection_strategy import select_index
from utils.plotter import plot_one
from utils.initial_strategy import initialize_game
def parse_args():
    """
    Helper function to set up command line argument parsing
    """
    parser = argparse.ArgumentParser()
    # Set up arguments
    parser.add_argument("-n", "--dimension", default=5, type=int,
                        help = 'Number of Strategies for each player')
    parser.add_argument("-sa", "--sample_strategy", default="or", type=str,
                        help = 'sample_strategy',
                        choices=["d", "ra", "nr", "r","or","lr"],)
    parser.add_argument("-se", "--selection_strategy", default="o", type=str,
                        help = 'Selection strategy',
                        choices=["o","p"],)
    parser.add_argument("-si", "--initial_strategy", default="d", type=str,
                        help = 'Initial strategy',
                        choices=["d","o"],)
    parser.add_argument("-t", "--timesteps", default=100, type=int,
                        help='Number of timesteps')
    parser.add_argument("-k", "--optimismconstant", default=0.5, type=float,
                        help='Optimistic Constant')
    return parser.parse_args()

def main(**kwargs):

    # Dimension of Problem
    n = kwargs.get("dimension")
    sa = kwargs.get("sample_strategy")
    se = kwargs.get("selection_strategy")
    si = kwargs.get("initial_strategy")
    t_max = kwargs.get("timesteps")

    t = 1

    Vs = []
    Percent = []
    Nash = []
    Gaps = []
    PercentBoundedPhi = []

    Game = game(n)
    NashIndices = FindNash(Game)

    initialize_game(Game, si)

    while t<t_max:

        ind1, ind2 = select_index(Game,se)

        Vs.append(Game.UnknownGame[ind1, ind2])

        Gaps.append(np.count_nonzero(Game.OptPhi < np.max(Game.PesPhi)) / (n ** 2) * 100)

        if [ind1,ind2] in NashIndices:
            Nash.append(1)
        else:
            Nash.append(0)

        Percent.append(100-(np.count_nonzero(np.isnan(Game.KnownU1)) / n**2) * 100)
        PercentBoundedPhi.append(np.count_nonzero(Game.OptPhi < np.max(Game.UnknownGame)) / (n ** 2) * 100)

        if not np.any(np.isnan(Game.KnownU1)):
            break

        i,j = sample_index(Game,sa)
        Game.sample(i,j)
        Game.check_bounds()

        #Update time index
        t += 1

    # create a figure and axis object
    plot_one(Game, Vs, Percent, Nash, Gaps, PercentBoundedPhi)

if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)