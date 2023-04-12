import argparse
import numpy as np
from game import potential_game, congestion_game

from utils.plotter import plot_many
from utils.algos import optimistic_solver,nash_ucb
from utils.regret import potential_regret,nash_regret,Nikaido_Isoda_regret
from utils.game_maker import make_game

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
    parser.add_argument("-t", "--timesteps", default=100, type=int,
                        help='Number of timesteps')
    parser.add_argument("-r", "--runs", default=1, type=int,
                        help='Number of Runs')
    parser.add_argument("-nl", "--noise", default=0, type=float,
                        help='Noise Level')
    parser.add_argument("-c", "--ucb_constant", default=0.1, type=float,
                        help='UCB Constant')
    parser.add_argument("-a", "--alpha", default=0.1, type=float,
                        help='Alpha')
    parser.add_argument("-g", "--game", default="congestion", type=str,
                        help='Game Type')
    parser.add_argument("-s", "--solver", default="nash_ucb", type=str,
                        help='Which solver to use')

    return parser.parse_args()

def main(**kwargs):
    # Dimension of Problem
    n = kwargs.get("dimension")
    k = kwargs.get("players")
    t_max = kwargs.get("timesteps")
    runs = kwargs.get("runs")
    nl = kwargs.get("noise")
    c = kwargs.get("ucb_constant")
    alpha = kwargs.get("alpha")
    g = kwargs.get("game")
    s = kwargs.get("solver")

    iterations = t_max

    regrets = []
    cumulative_regrets = []

    for r in range(runs):
        print(r)
        regrets.append([])
        cumulative_regrets.append([])

        if g == "random":
            Potential, unknown_utilitys = make_game(g, n, k)
            Game = potential_game(Potential, unknown_utilitys,nl)
        if g == "congestion":
            number_facilities, number_agents, facility_means = make_game(g, n, k)
            Game = congestion_game(facility_means,number_agents,nl)

        if s == "optimistic":
            algorithm = optimistic_solver(Game,c, alpha)
        if s == "nash_ucb":
            algorithm = nash_ucb(Game,iterations)

        cumulative_regret = 0
        for t in range(iterations):

            prob = algorithm.next_sample_prob(Game)

            choice = np.random.choice(np.arange(prob.size), p=prob.flatten())
            sample_tuple = np.unravel_index(choice, prob.shape)

            Game.sample(tuple(sample_tuple))

            regrets[r].append(potential_regret(Game,prob))

            cumulative_regret += regrets[r][t]
            cumulative_regrets[r].append(cumulative_regret)

            print("______")
            print(t)
            print("Sample: ", sample_tuple)
            print("Regret: ", regrets[r][t])
            print("Cumulative Regret: ", cumulative_regret)

    # create a figure and axis object
    plot_many(kwargs, np.array(cumulative_regrets), np.max(Game.Potential)-(np.sum(Game.Potential)/(n**k)))


if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)