import numpy as np

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
