import numpy as np
from numpy import random as rand
import itertools

def initialize_game(Game,i):
    n = Game.n
    k = Game.k

    if i == "d":
        for k in range(n):
            Game.sample(k, k)

    if i == "o":
        zero = np.ones(k).astype(int)
        Game.sample(tuple(zero),1)

    if i == "a":
        shape = [n]*k
        tuples = list(itertools.product(*[range(dim) for dim in shape]))
        for tuple_ in tuples:
            Game.number[tuple_] = 1
            for p in range(k):

                u_val = Game.UnknownUs[p][tuple_] + np.random.normal(0, Game.nl)
                Game.sum[p][tuple_] = u_val
                Game.sum_squared[p][tuple_] = u_val ** 2

                Game.OptUs[p][tuple_] = u_val + Game.c * np.sqrt(np.log(2) / 1)
                Game.PesUs[p][tuple_] = u_val - Game.c * np.sqrt(np.log(2) / 1)

        zero = np.ones(k).astype(int)
        Game.sample(tuple(zero), 1)

