from numpy import random as rand
import numpy as np

def make_game(game_type,n,k):

    if game_type == "random":

        shape = [n]*k

        Potential = rand.randint(-12500, 12501, shape) / 100000

        unknown_utilitys = [np.zeros(shape) for i in range(k)]

        for p in range(k):
            for i in range(1, n):
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i - 1] + [Ellipsis]

                unknown_utilitys[p][tuple(rel_slice)] = unknown_utilitys[p][tuple(prev_slice)] + Potential[
                    tuple(rel_slice)] - Potential[tuple(prev_slice)]

        for p in range(k):
            unknown_utilitys[p] += 0.25

        return Potential,unknown_utilitys

    if game_type == "congestion":
        number_facilities = n
        number_agents = k
        congestion_functions_means = [np.random.randint(1, 5, size=k) for i in range(number_facilities)]

        return number_facilities, number_agents, congestion_functions_means