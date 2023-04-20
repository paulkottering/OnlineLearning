import argparse
import numpy as np
from game import potential_game, congestion_game

from utils.plotter import plot_many
from utils.algos import optimistic_solver,nash_ucb,exponential_weights_annealing,nash_ca
from utils.regret import regret
from utils.game_maker import make_game
from utils.updates import opt_pes_make

import json
import os
import time

def generate_filename():
    """
    Generate a filename based on the current timestamp.

    Returns:
        str: Filename string.
    """
    timestamp = int(time.time())
    filename = "simulation_{}.json".format(timestamp)

    # Add the 'log_files' folder to the filename
    folder = "log_files"
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)

    return filepath, timestamp

def update_log_file(timestamp, kwargs):
    """
    Update the log file with the parameters of the simulation and the associated timestamp.

    Args:
        timestamp (int): Timestamp for the simulation.
        kwargs (dict): Dictionary of parameters.

    Returns:
        None
    """
    log_entry = {
        "timestamp": timestamp,
        "parameters": kwargs
    }

    log_filename = "simulation_log.txt"

    with open(log_filename, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def save_simulation_results(filename, kwargs, regrets, av_regret_val, run_times):
    """
    Save the regrets, run times, and parameters of the simulation to a JSON file.

    Args:
        filename (str): Filename for the JSON file.
        kwargs (dict): Dictionary of parameters.
        regrets (list): List of regrets for each run.
        av_regret_val (list): Average regret value.
        run_times (list): List of run times for each simulation run.

    Returns:
        None
    """
    # Create a dictionary to store the results and parameters
    data = {
        "parameters": kwargs,
        "nash": regrets[0],
        "potential": regrets[1],
        "nikaido_isoda": regrets[2],
        "ne_average_regret": av_regret_val[0],
        "ni_average_regret": av_regret_val[1],
        "p_average_regret": av_regret_val[2],
        "run_times": run_times  # Add run times to the data dictionary
    }

    # Save the data to a JSON file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

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
    parser.add_argument("-nl", "--noise", default=0.3, type=float,
                        help='Noise Level')
    parser.add_argument("-c", "--constant", default=0.1, type=float,
                        help='Constant')
    parser.add_argument("-a", "--alpha", default=0.1, type=float,
                        help='Alpha')
    parser.add_argument("-g", "--game", default="random", type=str,
                        help='Game Type')
    parser.add_argument("-s", "--solver", default="optimistic", type=str,
                        help='Which solver to use')

    return parser.parse_args()

def main(**kwargs):
    """
    Main function to run the simulation for different types of games, solvers and regret types.

    Keyword Args:
        dimension (int): Number of strategies for each player.
        players (int): Number of players.
        timesteps (int): Number of timesteps.
        runs (int): Number of runs.
        noise (float): Noise level.
        constant (float): Constant.
        alpha (float): Alpha parameter.
        game (str): Game type (e.g. "congestion" or "random").
        solver (str): Solver to use (e.g. "nash_ca" or "nash_ucb").

    Returns:
        None
    """

    # Parse keyword arguments
    n = kwargs.get("dimension")
    k = kwargs.get("players")
    t_max = kwargs.get("timesteps")
    runs = kwargs.get("runs")
    nl = kwargs.get("noise")
    c = kwargs.get("constant")
    alpha = kwargs.get("alpha")
    g = kwargs.get("game")
    s = kwargs.get("solver")

    iterations = t_max

    regrets = [[],[],[]]

    run_times = []  # Initialize a list to store the run times

    if s == "optimistic":
        matrices = opt_pes_make(n, k)

    # Iterate through the specified number of runs
    for r in range(runs):
        start_time = time.time()  # Record the start time of the simulation

        print(r)
        regrets[0].append([])
        regrets[1].append([])
        regrets[2].append([])


        # Initialize the game and solver based on the provided game type and solver type
        if g == "random" or g == "skewed" or g == "cooperative":
            Potential, unknown_utilitys = make_game(g, n, k)
            Game = potential_game(Potential, unknown_utilitys,nl)
        elif g == "congestion":
            number_facilities, number_agents, facility_means = make_game(g, n, k)
            Game = congestion_game(facility_means,number_agents,nl)
        else:
            raise RuntimeError("Not a valid game!")

        if s == "optimistic":
            algorithm = optimistic_solver(Game,c, alpha,matrices)
        elif s == "nash_ucb":
            algorithm = nash_ucb(Game, c, iterations)
        elif s == "exp_weight":
            algorithm = exponential_weights_annealing(Game, c, alpha)
        elif s == "nash_ca":
            algorithm = nash_ca(Game, c, alpha*n**k)
        else:
            raise RuntimeError("Not a valid algorithm!")

        # Instantiate a regret object and initialize cumulative regret
        reg = regret(Game)

        # Run the simulation for the specified number of iterations
        for t in range(iterations):

            #Generate probability tensor over all choices
            prob = algorithm.next_sample_prob(Game)

            #Sample choice from probability tensor
            choice = np.random.choice(np.arange(prob.size), p=prob.flatten())
            sample_tuple = np.unravel_index(choice, prob.shape)

            #Sample this joint action from the game
            Game.sample(tuple(sample_tuple))

            #Calculate regret and append lists
            regrets[0][r].append(reg.regrets("nash",prob))
            regrets[1][r].append(reg.regrets("potential",prob))
            regrets[2][r].append(reg.regrets("nikaido_isoda",prob))

            #Output log
            # print("______")
            # print(t)
            # print("Sample: ", sample_tuple)
            # print("Nash Regret: ", regrets[0][r][t])

        end_time = time.time()  # Record the end time of the simulation
        run_duration = end_time - start_time  # Calculate the duration of the run
        run_times.append(run_duration)  # Append the run duration to the run_times list

    av_regret_vals = reg.av_regret()

    filename, timestamp = generate_filename()
    save_simulation_results(filename, kwargs, regrets, av_regret_vals, run_times)  # Pass run_times to the function
    update_log_file(timestamp, kwargs)

if __name__ == "__main__":
    """
    Entry point for the script. Parses command line arguments and runs the main function.
    """
    # Parse command line arguments
    args = vars(parse_args())
    # Execute the main function with the parsed arguments
    main(**args)