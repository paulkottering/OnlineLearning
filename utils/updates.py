import numpy as np
from itertools import combinations
import itertools
def opt_pes_recalc(opt_us,pes_us,n,k):
    shape = [n] * k
    opt_phi = np.zeros(shape)
    pes_phi = np.zeros(shape)
    for i in range(k):
        for inactive in combinations(np.arange(k),i):
            opt = np.ones(shape)*float(np.inf)
            pes = np.ones(shape)*float(-np.inf)
            for utility_index in [i for i in range(k) if i not in inactive]:
                opt_y, pes_y = opt_pes_mat_recalc(opt_us[utility_index], pes_us[utility_index], n, k, inactive)
                opt = np.minimum(opt, opt_y)
                pes = np.maximum(pes, pes_y)
            opt_phi += opt
            pes_phi += pes
    return opt_phi,pes_phi

def opt_pes_mat_recalc(opt_u, pes_u, n, k, inactive):
    shape = [n] * k
    opt_y = np.ones(shape)
    pes_y = np.ones(shape)
    tuples = np.indices(shape).reshape(k, -1).T

    for tuple_ in tuples:
        opt_y[tuple(tuple_)], pes_y[tuple(tuple_)] = opt_pes_gen_recalc(opt_u, pes_u, n, k, inactive, tuple_)
    return opt_y, pes_y


def opt_pes_gen_recalc(OptU, PesU,n,k,inactive,ind_tuple):

    num_inactive = len(inactive)
    sum_opt = 0
    sum_pes = 0
    shape = [n]*k
    tuples = list(itertools.product(*[range(dim) for dim in shape]))

    for tuple_ in tuples:
        const = 1
        for ks in range(k):
            if tuple_[ks] == ind_tuple[ks]:
                const *= (1-1/n)
            else:
                const *= (-1/n)

        const *= (1/n)**num_inactive
        if const > 0:
            sum_opt += OptU[tuple_]*const
            sum_pes += PesU[tuple_] * const
        else:
            sum_opt += PesU[tuple_]*const
            sum_opt += OptU[tuple_] * const

    return sum_opt,sum_pes