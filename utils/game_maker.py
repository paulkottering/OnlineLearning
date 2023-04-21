from numpy import random as rand
import numpy as np

def make_game(game_type, n, k):
    """
    Creates a game instance based on the given game type, number of strategies (n), and number of players (k).

    Args:
        game_type (str): The type of the game to be created, either "random" or "congestion".
        n (int): The number of strategies for each player.
        k (int): The number of players.

    Returns:
        tuple: A tuple containing the game instance parameters depending on the game type.
    """

    if game_type == "random":
        # Create a random potential game with specified dimensions
        shape = [n] * k
        Potential = rand.randint(-12500, 12501, shape) / 100000

        # Initialize unknown utility matrices for each player
        unknown_utilitys = [np.zeros(shape) for i in range(k)]

        # Calculate the unknown utility matrices for each player based on the potential function
        for p in range(k):
            for i in range(1, n):
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i - 1] + [Ellipsis]

                unknown_utilitys[p][tuple(rel_slice)] = unknown_utilitys[p][tuple(prev_slice)] + Potential[
                    tuple(rel_slice)] - Potential[tuple(prev_slice)]

        # Add a constant to each player's unknown utility matrix
        for p in range(k):
            unknown_utilitys[p] += 0.25

        return Potential, unknown_utilitys

    elif game_type == "neg_skewed":
        # Create a random potential game with specified dimensions
        shape = [n] * k

        Potential = rand.beta(1.0, 3.0, shape) / 2 - 0.25

        # Initialize unknown utility matrices for each player
        unknown_utilitys = [np.zeros(shape) for i in range(k)]

        # Calculate the unknown utility matrices for each player based on the potential function
        for p in range(k):
            for i in range(1, n):
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i - 1] + [Ellipsis]

                unknown_utilitys[p][tuple(rel_slice)] = unknown_utilitys[p][tuple(prev_slice)] + Potential[
                    tuple(rel_slice)] - Potential[tuple(prev_slice)]

        # Add a constant to each player's unknown utility matrix
        for p in range(k):
            unknown_utilitys[p] += 0.25

        return Potential, unknown_utilitys

    elif game_type == "pos_skewed":
        # Create a random potential game with specified dimensions
        shape = [n] * k

        Potential = rand.beta(3.0, 1.0, shape) / 2 - 0.25

        # Initialize unknown utility matrices for each player
        unknown_utilitys = [np.zeros(shape) for i in range(k)]

        # Calculate the unknown utility matrices for each player based on the potential function
        for p in range(k):
            for i in range(1, n):
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i - 1] + [Ellipsis]

                unknown_utilitys[p][tuple(rel_slice)] = unknown_utilitys[p][tuple(prev_slice)] + Potential[
                    tuple(rel_slice)] - Potential[tuple(prev_slice)]

        # Add a constant to each player's unknown utility matrix
        for p in range(k):
            unknown_utilitys[p] += 0.25

        return Potential, unknown_utilitys

    elif game_type == "tailed_skewed":
        # Create a random potential game with specified dimensions
        shape = [n] * k

        Potential = rand.beta(0.5, 0.5, shape) / 2 - 0.25

        # Initialize unknown utility matrices for each player
        unknown_utilitys = [np.zeros(shape) for i in range(k)]

        # Calculate the unknown utility matrices for each player based on the potential function
        for p in range(k):
            for i in range(1, n):
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i - 1] + [Ellipsis]

                unknown_utilitys[p][tuple(rel_slice)] = unknown_utilitys[p][tuple(prev_slice)] + Potential[
                    tuple(rel_slice)] - Potential[tuple(prev_slice)]

        # Add a constant to each player's unknown utility matrix
        for p in range(k):
            unknown_utilitys[p] += 0.25

        return Potential, unknown_utilitys

    elif game_type == "mid_skewed":
        # Create a random potential game with specified dimensions
        shape = [n] * k

        Potential = rand.beta(5.0, 5.0, shape) / 2 - 0.25

        # Initialize unknown utility matrices for each player
        unknown_utilitys = [np.zeros(shape) for i in range(k)]

        # Calculate the unknown utility matrices for each player based on the potential function
        for p in range(k):
            for i in range(1, n):
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i - 1] + [Ellipsis]

                unknown_utilitys[p][tuple(rel_slice)] = unknown_utilitys[p][tuple(prev_slice)] + Potential[
                    tuple(rel_slice)] - Potential[tuple(prev_slice)]

        # Add a constant to each player's unknown utility matrix
        for p in range(k):
            unknown_utilitys[p] += 0.25

        return Potential, unknown_utilitys

    elif game_type == "cooperative":
        # Create a random potential game with specified dimensions
        shape = [n] * k

        Potential = rand.randint(-25000, 25001, shape) / 100000

        # Initialize unknown utility matrices for each player
        unknown_utilitys = [Potential for i in range(k)]

        # Add a constant to each player's unknown utility matrix
        for p in range(k):
            unknown_utilitys[p] += 0.25

        return Potential, unknown_utilitys

    elif game_type == "congestion":
        # Create a congestion game with specified number of facilities and agents
        number_facilities = n
        number_agents = k
        congestion_functions_means = [np.random.uniform(0, 1, size=k) for i in range(number_facilities)]

        return number_facilities, number_agents, congestion_functions_means
    else:
        raise RuntimeError("Not a valid game choice!")
