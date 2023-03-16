import numpy as np
from itertools import combinations

def opt_pes_recalc(opt_us,pes_us,n,k):
    shape = [n] * k
    opt_phi = np.zeros(shape)
    pes_phi = np.zeros(shape)
    for i in range(k):
        for inactive in combinations(np.arange(k),i):
            opt = np.ones(shape)*float(np.inf)
            pes = np.ones(shape)*float(-np.inf)
            for utility_index in range(k)[~np.isin(range(k),inactive)]:
                opt_phi, pes_phi = opt_pes_mat_recalc(opt_us[utility_index], pes_us[utility_index], n, k, inactive)
                opt = np.min(opt, opt_phi)
                pes = np.max(pes, pes_phi)
            opt_phi += opt
            pes_phi += pes
    return

def opt_pes_mat_recalc(opt_u, pes_u, n, k, inactive):
    shape = [n] * k
    opt_phi = np.ones(shape)
    pes_phi = np.ones(shape)
    tuples = np.indices(shape).reshape(k, -1).T

    for tuple_ in tuples:
        opt_phi[tuple(tuple_)], pes_phi[tuple(tuple_)] = opt_pes_gen_recalc(opt_u, pes_u, n, k, inactive, tuple_)

    return opt_phi, pes_phi

def opt_pes_gen_recalc(opt_u, pes_u, n, k, inactive, ind_tuple):
    num_inactive = len(inactive)
    ind_tuple = np.array(ind_tuple)

    tuples = np.indices((n,)*n).reshape(n, -1).T

    const = np.product(np.where(tuples == ind_tuple, 1 - 1/n, -1/n), axis=1)
    const *= (1/n) ** num_inactive

    sum_opt = np.sum((opt_u[tuples] * (const > 0) + pes_u[tuples] * (const <= 0)) * const[:, np.newaxis], axis=0)
    sum_pes = np.sum((pes_u[tuples] * (const > 0) + opt_u[tuples] * (const <= 0)) * const[:, np.newaxis], axis=0)

    return sum_opt, sum_pes