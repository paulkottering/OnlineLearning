import numpy as np
from itertools import combinations
import itertools

def opt_pes_recalc(matrices,opt_us,pes_us,n,k):

    shape = [n] * k
    opt_phi = np.zeros(shape)
    pes_phi = np.zeros(shape)

    opt_phi_max = np.zeros(shape)
    pes_phi_min = np.zeros(shape)

    tuples = list(np.indices(shape).reshape(k, -1).T)

    for i in range(k):

        combo_list = list(combinations(np.arange(k), i))

        for l in range(len(combo_list)):

            opt = np.ones(shape)*float(np.inf)
            pes = np.ones(shape)*float(-np.inf)

            opt_max = np.ones(shape)*float(-np.inf)
            pes_min = np.ones(shape)*float(np.inf)

            inactive = combo_list[l]

            for utility_index in [i for i in range(k) if i not in inactive]:
                opt_y = np.zeros(shape)
                pes_y = np.zeros(shape)

                for t in range(len(tuples)):
                    tuple_ = tuples[t]
                    opt_y[tuple(tuple_)] = np.sum(np.multiply(matrices[i][l][t][0],opt_us[utility_index])) + np.sum(np.multiply(matrices[i][l][t][2],pes_us[utility_index]))
                    pes_y[tuple(tuple_)] = np.sum(np.multiply(matrices[i][l][t][3],opt_us[utility_index])) + np.sum(np.multiply(matrices[i][l][t][1],pes_us[utility_index]))

                opt = np.minimum(opt, opt_y)
                pes = np.maximum(pes, pes_y)

                opt_max = np.maximum(opt_max, opt_y)
                pes_min = np.minimum(pes_min, pes_y)

            opt_phi += opt
            pes_phi += pes

            opt_phi_max += opt_max
            pes_phi_min += pes_min
    diff = np.sum(np.abs(opt_phi-opt_phi_max))
    return opt_phi, pes_phi,diff

def opt_pes_recalc_make(n,k):

    shape = [n] * k

    ks = []

    for i in range(k):
        ls = []
        combo_list = list(combinations(np.arange(k),i))

        for l in range(len(combo_list)):
            inactive = combo_list[l]

            ls.append(opt_pes_mat_recalc_make(n, k, inactive))

        ks.append(ls)

    return ks

def opt_pes_mat_recalc_make(n, k, inactive):

    shape = [n]* k
    opt_pes = []
    tuples = list(np.indices(shape).reshape(k, -1).T)

    for i in range(len(tuples)):
        weights = []
        one, two,three,four = opt_pes_gen_recalc_make(n, k, inactive, tuples[i])
        weights.append(one)
        weights.append(two)
        weights.append(three)
        weights.append(four)
        opt_pes.append(weights)
    return opt_pes

def opt_pes_gen_recalc_make(n,k,inactive,ind_tuple):
    shape = [n]*k
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
                const *= (1-1/n)
            else:
                const *= (-1/n)

        const *= (1/n)**num_inactive

        if const >= 0:

            sum_opt_opt[tuple_] = const
            sum_pes_pes[tuple_] = const

        else:

            sum_opt_pes[tuple_] = const
            sum_pes_opt[tuple_] = const

    return sum_opt_opt, sum_pes_pes, sum_opt_pes, sum_pes_opt
