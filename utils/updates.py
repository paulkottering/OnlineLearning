import numpy as np
from itertools import combinations

def OptPesY1Recalc(OptU,PesU,n):
    OptY1 = np.ones(n)
    PesY1 = np.ones(n)

    RowSumOpt = np.matmul(OptU, np.ones(n).T)
    OptSum = np.sum(OptU)

    RowSumPes = np.matmul(PesU, np.ones(n).T)
    PesSum = np.sum(PesU)

    for i in range(n):
        OptY1[i] = (n - 1) / (n ** 2) * RowSumOpt[i] - 1 / (n ** 2) * (PesSum - RowSumPes[i])
        PesY1[i] = (n - 1) / (n ** 2) * RowSumPes[i] - 1 / (n ** 2) * (OptSum - RowSumOpt[i])

    return OptY1, PesY1

def OptPesY2Recalc(OptU,PesU,n):
    OptY2 = np.ones(n)
    PesY2 = np.ones(n)

    ColSumOpt = np.matmul(np.ones(n), OptU)
    OptSum = np.sum(OptU)

    ColSumPes = np.matmul(np.ones(n), PesU)
    PesSum = np.sum(PesU)

    for j in range(n):
        OptY2[j] = (n - 1) / (n ** 2) * ColSumOpt[j] - 1 / (n ** 2) * (PesSum - ColSumPes[j])
        PesY2[j] = (n - 1) / (n ** 2) * ColSumPes[j] - 1 / (n ** 2) * (OptSum - ColSumOpt[j])

    return OptY2, PesY2

def OptPesYPRecalc(OptU,PesU,n):
        n = n
        OptYP = np.ones((n, n))
        PesYP = np.ones((n, n))

        OptMat = OptU
        ColSumOpt = np.matmul(np.ones(n), OptU)
        RowSumOpt = np.matmul(OptU, np.ones(n).T)
        OptSum = np.sum(OptU)

        PesMat = PesU
        ColSumPes = np.matmul(np.ones(n), PesU)
        RowSumPes = np.matmul(PesU, np.ones(n).T)
        PesSum = np.sum(PesU)

        for i in range(n):
            for j in range(n):
                OtherOpt = (1) / (n ** 2) * (OptSum - ColSumOpt[j] - RowSumOpt[i] + OptMat[i, j])
                OtherPes = (1) / (n ** 2) * (PesSum - ColSumPes[j] - RowSumPes[i] + PesMat[i, j])

                OptYP[i, j] = ((n - 1) ** 2) / (n ** 2) * OptMat[i, j] - (n - 1) / (n ** 2) * (
                            ColSumPes[j] - PesMat[i, j]) - (n - 1) / (n ** 2) * (RowSumPes[i] - PesMat[i, j]) + OtherOpt
                PesYP[i, j] = ((n - 1) ** 2) / (n ** 2) * PesMat[i, j] - (n - 1) / (n ** 2) * (
                            ColSumOpt[j] - OptMat[i, j]) - (n - 1) / (n ** 2) * (RowSumOpt[i] - OptMat[i, j]) + OtherPes

        return OptYP, PesYP


def OptPesYPRecalc(OptU, PesU, n):
    n = n
    OptYP = np.ones((n, n))
    PesYP = np.ones((n, n))

    OptMat = OptU
    ColSumOpt = np.matmul(np.ones(n), OptU)
    RowSumOpt = np.matmul(OptU, np.ones(n).T)
    OptSum = np.sum(OptU)

    PesMat = PesU
    ColSumPes = np.matmul(np.ones(n), PesU)
    RowSumPes = np.matmul(PesU, np.ones(n).T)
    PesSum = np.sum(PesU)

    for i in range(n):
        for j in range(n):
            OtherOpt = (1) / (n ** 2) * (OptSum - ColSumOpt[j] - RowSumOpt[i] + OptMat[i, j])
            OtherPes = (1) / (n ** 2) * (PesSum - ColSumPes[j] - RowSumPes[i] + PesMat[i, j])

            OptYP[i, j] = ((n - 1) ** 2) / (n ** 2) * OptMat[i, j] - (n - 1) / (n ** 2) * (
                    ColSumPes[j] - PesMat[i, j]) - (n - 1) / (n ** 2) * (RowSumPes[i] - PesMat[i, j]) + OtherOpt
            PesYP[i, j] = ((n - 1) ** 2) / (n ** 2) * PesMat[i, j] - (n - 1) / (n ** 2) * (
                    ColSumOpt[j] - OptMat[i, j]) - (n - 1) / (n ** 2) * (RowSumOpt[i] - OptMat[i, j]) + OtherPes

    return OptYP, PesYP

def OptPesGenRecalc(OptU, PesU,n,k,inactive,ind_tuple):

    num_inactive = len(inactive)
    sum_opt = 0
    sum_pes = 0

    tuples = np.stack(np.meshgrid(*[range(n)]*k), axis=-1).reshape(-1, k)

    for tuple in tuples:
        const = 1
        for ks in range(k):
            if tuple[ks] == ind_tuple[ks]:
                const *= (1-1/n)
            else:
                const *= (-1/n)

        const *= (1/n)**num_inactive
        if const > 0:
            sum_opt += OptU[tuple]*const
            sum_pes += PesU[tuple] * const
        else:
            sum_opt += PesU[tuple]*const
            sum_opt += OptU[tuple] * const

    return sum_opt,sum_pes

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