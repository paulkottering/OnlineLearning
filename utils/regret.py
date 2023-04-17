import numpy as np
import itertools


class regret():
    def __init__(self,game):

        tuples = list(itertools.product(*[range(dim) for dim in game.shape]))

        self.nash_regret_matrix = np.zeros(game.shape)
        self.potential_regret_matrix = np.zeros(game.shape)
        self.ni_regret_matrix = np.zeros(game.shape)

        for tuple in tuples:
            self.nash_regret_matrix[tuple] = nash_regret(game, tuple)
            self.potential_regret_matrix[tuple] = potential_regret(game, tuple)
            self.ni_regret_matrix[tuple] = nikaido_isoda_regret(game, tuple)

    def av_regret(self,e):
        if e == "nash":
            return np.mean(self.nash_regret_matrix)
        if e == "potential":
            return np.mean(self.potential_regret_matrix)
        if e == "nikaido_isoda":
            return np.mean(self.ni_regret_matrix)

    def regrets(self,e,prob):
        if e == "nash":
            return np.sum(self.nash_regret_matrix*prob)
        if e == "potential":
            return np.sum(self.potential_regret_matrix*prob)
        if e == "nikaido_isoda":
            return np.sum(self.ni_regret_matrix*prob)


def potential_regret(game, sample_tuple):
    return np.max(game.Potential) - game.Potential[sample_tuple]

def nash_regret(game,sample_tuple):
    max_nash_regret = -np.inf
    for p in range(game.k):
        cut_slice = [slice(None) if j == p else int(sample_tuple[j]) for j in range(len(sample_tuple))]
        regret = np.max(game.utility_matrices[p][cut_slice]) - game.utility_matrices[p][sample_tuple]
        max_nash_regret = max_nash_regret if max_nash_regret > regret else regret
    return max_nash_regret

def nikaido_isoda_regret(game,sample_tuple):
    max_ni_regret = -np.inf
    for p in range(game.k):
        cut_slice = [slice(None) if j == p else int(sample_tuple[j]) for j in range(len(sample_tuple))]
        regret = np.max(game.utility_matrices[p][cut_slice]) - game.utility_matrices[p][sample_tuple]
        max_ni_regret += regret
    return max_ni_regret

