import numpy as np

def FindNash(game):
    one_max_indices = np.argmax(game.UnknownGame, axis=1)
    two_max_indices = np.argmax(game.UnknownGame, axis=0)

    NashIndices = []

    n = len(one_max_indices)

    for i in range(n):
        if two_max_indices[one_max_indices[i]] == i:
            NashIndices.append([i,one_max_indices[i]])

    return NashIndices
