import numpy as np

def potential_regret(game, prob):
    return np.max(game.Potential) - np.sum(game.Potential * prob)

def nash_regret(game,sample_tuple):
    max = 0
    for p in range(game.k):
        slice = game.UnknownUs[p][tuple(slice(None) if i == p else idx for i, idx in enumerate(sample_tuple))]
        regret = np.max(slice) - game.UnknownUs[p][sample_tuple]
        max = np.max(max,regret)
    return max

def Nikaido_Isoda_regret(game,sample_tuple):
    max = 0
    for p in range(game.k):
        slice = game.UnknownUs[p][tuple(slice(None) if i == p else idx for i, idx in enumerate(sample_tuple))]
        regret = np.max(slice) - game.UnknownUs[p][sample_tuple]
        max += regret
    return max
