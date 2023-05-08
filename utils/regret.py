import numpy as np
import itertools

class regret():
    """
    A class to compute and store different types of regrets for a given game.

    Attributes:
        nash_regret_matrix (numpy.ndarray): A matrix to store Nash regret values for each action combination.
        potential_regret_matrix (numpy.ndarray): A matrix to store potential regret values for each action combination.
        ni_regret_matrix (numpy.ndarray): A matrix to store Nikaido-Isoda regret values for each action combination.

    Methods:
        __init__(self, game): Initialize the regret object with a given game.
        av_regret(self, e): Compute the average regret for a specific regret type.
        regrets(self, e, prob): Compute the total regret for a specific regret type with given probabilities.
    """
    def __init__(self,game,s):

        self.s = s
        self.k = game.k

        if (s != "nash_ucb"):

            tuples = list(itertools.product(*[range(dim) for dim in game.shape]))

            self.nash_regret_matrix = np.zeros(game.shape)
            self.potential_regret_matrix = np.zeros(game.shape)
            self.ni_regret_matrix = np.zeros(game.shape)

            for tuple in tuples:
                self.nash_regret_matrix[tuple] = nash_regret(game, tuple)
                self.potential_regret_matrix[tuple] = potential_regret(game, tuple)
                self.ni_regret_matrix[tuple] = nikaido_isoda_regret(game, tuple)

    def av_regret(self):
            return [np.mean(self.nash_regret_matrix), np.mean(self.potential_regret_matrix),np.mean(self.ni_regret_matrix)]

    def regrets(self,e,prob):
        if e == "nash":
            return np.sum(self.nash_regret_matrix*prob)
        if e == "potential":
            return np.sum(self.potential_regret_matrix*prob)
        if e == "nikaido_isoda":
            return np.sum(self.ni_regret_matrix*prob)

    def regret_congestion(self,game,sample_tuple):

        max_ni_regret = 0
        max_nash_regret = -np.inf
        for p in range(self.k):
            indexs = [tuple_changer(sample_tuple, k, p) for k in range(len(game.actions[p]))]
            vector = [calculate_utility(game,tuple_, p) for tuple_ in indexs]
            regret = np.max(vector) - calculate_utility(game,sample_tuple, p)
            max_nash_regret = max_nash_regret if max_nash_regret > regret else regret
            max_ni_regret += regret

        return [max_nash_regret,0,max_ni_regret]



def potential_regret(game, sample_tuple):
    """
    Compute the potential regret for a given game and action combination.

    Args:
        game (object): The game object to compute the regret for.
        sample_tuple (tuple): The action combination for which to compute the potential regret.

    Returns:
        float: The potential regret for the given game and action combination.
    """
    return np.max(game.Potential) - game.Potential[sample_tuple]

def nash_regret(game,sample_tuple):
    """
    Compute the Nash regret for a given game and action combination.

    Args:
        game (object): The game object to compute the regret for.
        sample_tuple (tuple): The action combination for which to compute the Nash regret.

    Returns:
        float: The Nash regret for the given game and action combination.
    """
    max_nash_regret = -np.inf
    for p in range(game.k):
        cut_slice = [slice(None) if j == p else int(sample_tuple[j]) for j in range(len(sample_tuple))]
        regret = np.max(game.utility_matrices[p][cut_slice]) - game.utility_matrices[p][sample_tuple]
        max_nash_regret = max_nash_regret if max_nash_regret > regret else regret
    return max_nash_regret

def nikaido_isoda_regret(game,sample_tuple):
    """
    Compute the Nikaido-Isoda regret for a given game and action combination.

    Args:
        game (object): The game object to compute the regret for.
        sample_tuple (tuple): The action combination for which to compute the Nikaido-Isoda regret.

    Returns:
        float: The Nikaido-Isoda regret for the given game and action combination.
    """
    max_ni_regret = 0
    for p in range(game.k):
        cut_slice = [slice(None) if j == p else int(sample_tuple[j]) for j in range(len(sample_tuple))]
        regret = np.max(game.utility_matrices[p][cut_slice]) - game.utility_matrices[p][sample_tuple]
        max_ni_regret += regret
    return max_ni_regret

def tuple_changer(policy_tuple,k,p):
    tuple_list = list(policy_tuple)
    tuple_list[p] = k
    return tuple(tuple_list)

def calculate_utility(game,tuple,p):
    # Calculate the number of agents visiting each facility for the given action combination
    numbers = game.number_for_each_facility(tuple)
    # Get the facilities visited by the agent in the action combination
    facilities_visited = game.actions[p][tuple[p]]
    # Calculate the utility for the agent
    utility = 0
    for facility in facilities_visited:
        utility += game.facility_means[facility][int(numbers[facility]) - 1]
    # Assign the utility value to the corresponding position in the utility matrix
    return utility