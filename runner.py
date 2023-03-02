import argparse
import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt
from game import game

from utils.Nash import FindNash
from selection_strategy import select_index

def parse_args():
    """
    Helper function to set up command line argument parsing
    """
    parser = argparse.ArgumentParser()
    # Set up arguments
    parser.add_argument("-n", "--dimension", default=5, type=int,
                        help = 'Number of Strategies for each player')
    parser.add_argument("-s", "--strategy", default="or", type=str,
                        help = 'Sample selection strategy',
                        choices=["d", "ra", "nr", "r","or"],)
    parser.add_argument("-t", "--timesteps", default=100, type=int,
                        help='Number of timesteps')
    parser.add_argument("-k", "--optimismconstant", default=0.5, type=float,
                        help='Optimistic Constant')
    return parser.parse_args()

def main(**kwargs):

    # Dimension of Problem
    n = kwargs.get("dimension")
    s = kwargs.get("strategy")
    t_max = kwargs.get("timesteps")

    t = 1

    Vs = []
    Percent = []
    Nash = []
    Gaps = []
    PercentBoundedPhi = []

    Game = game(n)
    NashIndices = FindNash(Game)

    Game.initial_samples()

    while t<t_max:

        ind1, ind2 = np.unravel_index(np.argmax(Game.OptPhi, axis=None),(n,n))
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

        i,j = select_index(Game,s)
        Game.sample(i,j)
        Game.check_bounds()

        #Update time index
        t += 1

    print(Game.number_samples)
    print(np.sum(np.abs(Game.OptPhi - Game.PesPhi)))
    print(np.sum(np.abs(Game.UnknownGame - Game.OptPhi)))

    # create a figure and axis object
    fig, ax1 = plt.subplots()
    fig.set_figwidth(15)
    # plot the first array using the left y-axis
    ax1.plot(Vs, color='red')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('True Potential Value of Optimistic Potential Estimate Maximum', color='red')
    ax1.axhline(y=np.max(Game.UnknownGame), color='r', linestyle='--')
    ax1.set_ylim([np.min(Game.UnknownGame), np.max(Game.UnknownGame)+0.01])

    # create a twin axis object on the right side
    ax2 = ax1.twinx()

    # plot the second array using the right y-axis
    ax2.plot(Percent, color='blue',label = 'Percent of Utility Values Sampled')
    ax2.set_ylim([0,100])
    ax2.fill_between(range(len(Nash)), -5, 5, where=Nash > np.zeros(len(Nash)), color='green', alpha=0.5)
    # plot the second array using the right y-axis
    ax2.plot(Gaps, color='orange', label='Percentage of Strategy Profiles "Non-Active"')
    ax2.plot(PercentBoundedPhi, color='purple', label='Percentage of Optimistic Phi < Phi_max')
    ax2.set_ylabel('%')
    ax2.spines.right.set_position(("axes", 1.05))

    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
              fancybox=True, shadow=True, ncol=5)

    # set the title of the plot
    plt.savefig("Figures/Test")

if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)