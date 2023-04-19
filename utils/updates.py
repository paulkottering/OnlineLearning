import numpy as np
from itertools import combinations
import itertools

def opt_pes_recalc(matrices,opt_us,pes_us,n,k):
    """
    Calculate optimistic and pessimistic estimates based on pre-calculated weight matrices.

    Args:
    matrices (list): Pre-calculated weight matrices
    opt_us (array): Optimistic utilities
    pes_us (array): Pessimistic utilities
    n (int): Number of actions for each agent
    k (int): Number of agents

    Returns:
    tuple: Optimistic and pessimistic estimates
    """

    # Initialize arrays
    shape = [n] * k
    opt_phi = np.zeros(shape)
    pes_phi = np.zeros(shape)
    opt_phi_max = np.zeros(shape)
    pes_phi_min = np.zeros(shape)

    # Generate index tuples
    tuples = list(np.indices(shape).reshape(k, -1).T)

    # Iterate through combinations
    for i in range(k):
        combo_list = list(combinations(np.arange(k), i))
        for l in range(len(combo_list)):

            # Initialize optimistic and pessimistic estimates
            opt = np.ones(shape)*float(np.inf)
            pes = np.ones(shape)*float(-np.inf)

            opt_max = np.ones(shape)*float(-np.inf)
            pes_min = np.ones(shape)*float(np.inf)

            # Get inactive agents list
            inactive = combo_list[l]

            # Iterate through active agents
            for utility_index in [i for i in range(k) if i not in inactive]:
                opt_y = np.zeros(shape)
                pes_y = np.zeros(shape)

                # Calculate optimistic and pessimistic estimates for each agent
                for t in range(len(tuples)):
                    tuple_ = tuples[t]
                    opt_y[tuple(tuple_)] = np.sum(np.multiply(matrices[i][l][t][0],opt_us[utility_index])) + np.sum(np.multiply(matrices[i][l][t][2],pes_us[utility_index]))
                    pes_y[tuple(tuple_)] = np.sum(np.multiply(matrices[i][l][t][3],opt_us[utility_index])) + np.sum(np.multiply(matrices[i][l][t][1],pes_us[utility_index]))

                # Update optimistic and pessimistic min and max
                opt = np.minimum(opt, opt_y)
                pes = np.maximum(pes, pes_y)

                opt_max = np.maximum(opt_max, opt_y)
                pes_min = np.minimum(pes_min, pes_y)

            # Update summation of components
            opt_phi += opt
            pes_phi += pes

            opt_phi_max += opt_max
            pes_phi_min += pes_min

    return opt_phi, pes_phi

def opt_pes_recalc2(matrices,opt_us,pes_us,n,k):
    """
    Calculate optimistic and pessimistic estimates based on pre-calculated weight matrices.

    Args:
    matrices (list): Pre-calculated weight matrices
    opt_us (array): Optimistic utilities
    pes_us (array): Pessimistic utilities
    n (int): Number of actions for each agent
    k (int): Number of agents

    Returns:
    tuple: Optimistic and pessimistic estimates
    """

    # Initialize arrays
    shape = [n] * k
    opt_phi = np.zeros(shape)
    pes_phi = np.zeros(shape)
    opt_phi_max = np.zeros(shape)
    pes_phi_min = np.zeros(shape)

    # Generate index tuples
    tuples = list(np.indices(shape).reshape(k, -1).T)

    # Iterate through combinations
    for i in range(k):
        combo_list = list(combinations(np.arange(k), i))
        for l in range(len(combo_list)):

            # Initialize optimistic and pessimistic estimates
            opt = np.ones(shape)*float(np.inf)
            pes = np.ones(shape)*float(-np.inf)

            opt_max = np.ones(shape)*float(-np.inf)
            pes_min = np.ones(shape)*float(np.inf)

            # Get inactive agents list
            inactive = combo_list[l]

            # Iterate through active agents
            utility_index = [i for i in range(k) if i not in inactive][0]
            opt_y = np.zeros(shape)
            pes_y = np.zeros(shape)

            # Calculate optimistic and pessimistic estimates for each agent
            for t in range(len(tuples)):
                tuple_ = tuples[t]
                opt_y[tuple(tuple_)] = np.sum(np.multiply(matrices[i][l][t][0],opt_us[utility_index])) + np.sum(np.multiply(matrices[i][l][t][2],pes_us[utility_index]))
                pes_y[tuple(tuple_)] = np.sum(np.multiply(matrices[i][l][t][3],opt_us[utility_index])) + np.sum(np.multiply(matrices[i][l][t][1],pes_us[utility_index]))

            # Update summation of components
            opt_phi += opt_y
            pes_phi += opt_y

    return opt_phi, pes_phi

def opt_pes_make(n,k):
    """
    Make and store matrices for calculating optimistic and pessimistic potential estimates. For all possible combinations.

    Args:
    n (int): Number of actions per agent
    k (int): Number of agents

    Returns:
    list: List of the 4 weight matrices for a given component
    """
    ks = []

    # Iterate through combinations
    for i in range(k):
        ls = []
        combo_list = list(combinations(np.arange(k),i))

        # Calculate weight matrices for each combination
        for l in range(len(combo_list)):
            inactive = combo_list[l]
            ls.append(opt_pes_mat_make(n, k, inactive))

        ks.append(ls)

    return ks


def opt_pes_mat_make(n, k, inactive):
    """
    Generate a list of optimistic and pessimistic matrices for a given n, k, and inactive agents.

    :param n: int, size of the actions dimensions
    :param k: int, number of agents
    :param inactive: list of int, indices of inactive agents
    :return: list of numpy arrays, containing the optimistic and pessimistic matrices
    """

    shape = [n] * k
    opt_pes = []
    indices = np.indices(shape).reshape(k, -1).T
    tuples = list(map(tuple, indices))

    for t in tuples:
        # Generate the four matrices for the current tuple
        sum_opt_opt, sum_pes_pes, sum_opt_pes, sum_pes_opt = opt_pes_tuple_make(n, k, inactive, t)

        # Add the matrices to the opt_pes list
        opt_pes.append([sum_opt_opt, sum_pes_pes, sum_opt_pes, sum_pes_opt])

    return opt_pes

def opt_pes_tuple_make(n, k, inactive, ind_tuple):
    """
    Generate the four matrices for a given n, k, inactive agents and index tuple.

    :param n: int, size of the actions dimensions
    :param k: int, number of agents
    :param inactive: list of int, indices of inactive agents
    :param ind_tuple: tuple of int, index tuple
    :return: tuple of numpy arrays, containing the four matrices
    """

    shape = [n] * k
    num_inactive = len(inactive)

    sum_opt_opt = np.zeros(shape)
    sum_pes_pes = np.zeros(shape)
    sum_opt_pes = np.zeros(shape)
    sum_pes_opt = np.zeros(shape)

    tuples = list(itertools.product(*[range(dim) for dim in shape]))

    for tuple_ in tuples:
        const = 1

        for ks in [i for i in range(k) if i not in inactive]:
            if tuple_[ks] == ind_tuple[ks]:
                const *= (1 - 1 / n)
            else:
                const *= (-1 / n)

        const *= (1 / n) ** num_inactive

        if const >= 0:
            sum_opt_opt[tuple_] = const
            sum_pes_pes[tuple_] = const
        else:
            sum_opt_pes[tuple_] = const
            sum_pes_opt[tuple_] = const

    return sum_opt_opt, sum_pes_pes, sum_opt_pes, sum_pes_opt

