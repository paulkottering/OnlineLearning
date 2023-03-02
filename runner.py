import argparse
import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt

def parse_args():
    """
    Helper function to set up command line argument parsing
    """
    parser = argparse.ArgumentParser()
    # Set up arguments
    parser.add_argument("-n", "--dimension", default=5, type=int,
                        help = 'Number of Strategies for each player')
    parser.add_argument("-s", "--strategy", default=1, type=int,
                        help = 'Sample selection strategy')
    parser.add_argument("-t", "--timesteps", default=100, type=int,
                        help='Number of timesteps')
    parser.add_argument("-k", "--optimismconstant", default=0.5, type=float,
                        help='Optimistic Constant')
    return parser.parse_args()
def FindNash(game):
    one_max_indices = np.argmax(game, axis=1)
    two_max_indices = np.argmax(game, axis=0)

    NashIndices = []

    n = len(one_max_indices)

    for i in range(n):
        if two_max_indices[one_max_indices[i]] == i:
            NashIndices.append([i,one_max_indices[i]])

    return NashIndices
def OptPesYPRecalc(OptU,PesU, n):

    OptYP = np.ones((n, n))
    PesYP = np.ones((n, n))

    OptMat = OptU
    ColSumOpt = np.matmul(np.ones(n),OptU)
    RowSumOpt = np.matmul(OptU,np.ones(n).T)
    OptSum = np.sum(OptU)

    PesMat =  PesU
    ColSumPes = np.matmul(np.ones(n), PesU)
    RowSumPes = np.matmul(PesU, np.ones(n).T)
    PesSum = np.sum(PesU)

    for i in range(n):
        for j in range(n):
            OtherOpt = (1)/(n**2)*(OptSum - ColSumOpt[j] - RowSumOpt[i] + OptMat[i,j])
            OtherPes = (1) / (n ** 2) * (PesSum - ColSumPes[j] - RowSumPes[i] + PesMat[i, j])

            OptYP[i,j] = ((n-1)**2)/(n**2)*OptMat[i,j] -(n-1)/(n**2)*(ColSumPes[j] - PesMat[i,j])  -(n-1)/(n**2)*(RowSumPes[i]- PesMat[i,j]) + OtherOpt
            PesYP[i, j] = ((n-1)**2)/(n**2)*PesMat[i, j] -(n-1)/(n**2)*(ColSumOpt[j] - OptMat[i,j]) -(n-1)/(n**2)*(RowSumOpt[i]- OptMat[i,j]) + OtherPes

    return OptYP, PesYP
def OptPesY1Recalc(OptU,PesU, n):

    OptY1 = np.ones(n)
    PesY1 = np.ones(n)

    RowSumOpt = np.matmul(OptU,np.ones(n).T)
    OptSum = np.sum(OptU)

    RowSumPes = np.matmul(PesU, np.ones(n).T)
    PesSum = np.sum(PesU)

    for i in range(n):
        OptY1[i] = (n-1)/(n**2)*RowSumOpt[i] - 1/(n**2) *(PesSum - RowSumPes[i])
        PesY1[i] = (n-1)/(n**2)*RowSumPes[i] - 1/(n**2) *(OptSum - RowSumOpt[i])

    return OptY1, PesY1
def OptPesY2Recalc(OptU,PesU, n):

    OptY2 = np.ones(n)
    PesY2 = np.ones(n)

    ColSumOpt = np.matmul(np.ones(n),OptU)
    OptSum = np.sum(OptU)

    ColSumPes = np.matmul(np.ones(n), PesU)
    PesSum = np.sum(PesU)

    for j in range(n):
        OptY2[j] = (n-1)/(n**2)*ColSumOpt[j] - 1/(n**2) *(PesSum - ColSumPes[j])
        PesY2[j] = (n-1)/(n**2)*ColSumPes[j] - 1/(n**2) *(OptSum - ColSumOpt[j])

    return OptY2, PesY2
def OptPesU2Update(OptU2,PesU2,a,b, n, OptPhi,PesPhi,PhiMax,PhiMin,UnknownGame):

    for j in range(n):
        if j != b:
            OptRange = np.minimum(PhiMax,OptPhi[a,j])-np.maximum(PesPhi[a,b],PhiMin)
            PesRange = np.maximum(PhiMin,PesPhi[a,j])-np.minimum(OptPhi[a,b],PhiMax)

            # if PesU2[a, b] + PhiMin - PhiMax > UnknownGame[a,j]:
            #     print(PesU2[a, b] + PhiMin - PhiMax - UnknownGame[a,j])
            #     print('Miss5')

            OptU2[a, j] = np.minimum(OptU2[a, j], OptU2[a, b] + OptRange)
            PesU2[a, j] = np.maximum(PesU2[a, j], PesU2[a, b] + PesRange)
    return OptU2, PesU2
def OptPesU1Update(OptU1,PesU1,a,b, n, OptPhi,PesPhi,PhiMax,PhiMin):

    for i in range(n):
        if i != a:
            OptRange = np.minimum(PhiMax,OptPhi[i,b])-np.maximum(PesPhi[a,b],PhiMin)
            PesRange = np.maximum(PhiMin,PesPhi[i,b])-np.minimum(OptPhi[a,b],PhiMax)

            # MinRange = np.minimum(PhiMax,OptPhi[i, b])-np.maximum(PesPhi[a,b],PhiMin)
            # MinRange = OptPhi[i, b] - PesPhi[a,b]
            # MinRange = PhiMax-PhiMin

            OptU1[i, b] = np.minimum(OptU1[i, b], OptU1[a, b] + OptRange)
            PesU1[i, b] = np.maximum(PesU1[i, b], PesU1[a, b] + PesRange)
    return OptU1,PesU1
def sample(i,j,matrices, n,PhiMax,PhiMin,UnknownGame):

    UnknownU1 = matrices[0]
    UnknownU2 = matrices[1]

    KnownU1 = matrices[2]
    KnownU2 = matrices[3]

    OptPhi = matrices[4]
    PesPhi = matrices[5]

    OptU1 = matrices[6]
    PesU1 = matrices[7]

    OptU2 = matrices[8]
    PesU2 = matrices[9]

    if np.isnan(matrices[2][i,j]):
        # Sample UnknownU1 and Unknown U2
        U1Val = UnknownU1[i, j]
        U2Val = UnknownU2[i, j]

        # Update KnownU1 and KnownU2
        KnownU1[i, j] = U1Val
        KnownU2[i, j] = U2Val

        # Update Samples in OptU1, PesU1, OptU2, PesU2
        OptU1[i, j] = U1Val
        PesU1[i, j] = U1Val

        OptU2[i, j] = U2Val
        PesU2[i, j] = U2Val

        # Update OptU1, PesU1, OptU2, PesU2
        OptU1, PesU1 = OptPesU1Update(OptU1, PesU1, i, j, n, OptPhi, PesPhi, PhiMax, PhiMin)
        OptU2, PesU2 = OptPesU2Update(OptU2, PesU2, i, j, n, OptPhi, PesPhi, PhiMax,PhiMin,UnknownGame)

        # Update OptY1,PesY1
        OptY1, PesY1 = OptPesY1Recalc(OptU1, PesU1, n)
        OptY2, PesY2 = OptPesY2Recalc(OptU2, PesU2, n)

        # Recalculate OptYP1,PesYP1
        OptYP1, PesYP1 = OptPesYPRecalc(OptU1, PesU1, n)

        # Update OptYP2,PesYP2
        OptYP2, PesYP2 = OptPesYPRecalc(OptU2, PesU2, n)

        OptYP = np.minimum(OptYP1, OptYP2)
        PesYP = np.minimum(PesYP1, PesYP2)

        # Optimistic potential matrix estimate
        OptPhi = OptYP + np.array([OptY1] * n).T + np.array([OptY2] * n)

        # Pessimistic potential matrix estimate
        PesPhi = PesYP + np.array([PesY1] * n).T + np.array([PesY2] * n)


        # if (np.count_nonzero(OptPhi_old < PhiMax) < np.count_nonzero(OptPhi_new<PhiMax)) or (np.count_nonzero(PesPhi_old>PhiMin) < np.count_nonzero(PesPhi_new>PhiMin)):
        #
        #     print(np.count_nonzero(OptPhi_old < PhiMax))
        #
        #     OptPhi_old = np.copy(OptPhi_new)
        #     PesPhi_old = np.copy(PesPhi_new)
        #
        #     #Update OptU and PesU matrices
        #     for i in range(n):
        #         for j in range(n):
        #             OptU1, PesU1 = OptPesU1Update(OptU1, PesU1, i, j, n, OptPhi_old, PesPhi_old, PhiMax, PhiMin)
        #             OptU2, PesU2 = OptPesU2Update(OptU2, PesU2, i, j, n, OptPhi_old, PesPhi_old, PhiMax, PhiMin)
        #
        #     # Update OptY1,PesY1
        #     OptY1, PesY1 = OptPesY1Recalc(OptU1, PesU1, n)
        #     OptY2, PesY2 = OptPesY2Recalc(OptU2, PesU2, n)
        #
        #     # Recalculate OptYP1,PesYP1
        #     OptYP1, PesYP1 = OptPesYPRecalc(OptU1, PesU1, n)
        #
        #     # Update OptYP2,PesYP2
        #     OptYP2, PesYP2 = OptPesYPRecalc(OptU2, PesU2, n)
        #
        #     OptYP = OptYP1
        #     PesYP = PesYP1
        #
        #     # Optimistic potential matrix estimate
        #     OptPhi_new = OptYP + np.array([OptY1] * n).T + np.array([OptY2] * n)
        #
        #     # Pessimistic potential matrix estimate
        #     PesPhi_new = PesYP + np.array([PesY1] * n).T + np.array([PesY2] * n)

        matrices = [UnknownU1, UnknownU2, KnownU1, KnownU2, OptPhi, PesPhi, OptU1, PesU1, OptU2, PesU2]

    return matrices
def strategy(KnownU1,OptPhi,PesPhi,n,stratnum):
    nan_new_act = []
    new_act = []
    indices = []

    reverse_ascending_order = np.argsort(OptPhi - PesPhi, axis=None)[::-1]
    i = 0

    if stratnum == 1:
        for i in range(len(reverse_ascending_order)):
            index = reverse_ascending_order[i]
            row, col = np.unravel_index(index, OptPhi.shape)
            if OptPhi[row, col] > np.max(PesPhi):
                if np.isnan(OptPhi[row, col]):
                    return row, col
        if np.any(np.isnan(KnownU1)):
            rand_ind1 = np.argwhere(np.isnan(KnownU1))[0][0]
            rand_ind2 = np.argwhere(np.isnan(KnownU1))[0][1]
            return rand_ind1, rand_ind2

    if stratnum == 2:

        ind1, ind2 = np.unravel_index(np.argmax(OptPhi, axis=None), (n, n))

        if np.isnan(KnownU1[ind1, ind2]):
            return ind1, ind2

        nan_new_act = []
        new_act = []
        indices = []
        for index in np.ndindex((n,n)):
            indices.append(list(index))

        for ind in indices:
            if OptPhi[ind[0], ind[1]] >= np.max(PesPhi):
                if np.isnan(KnownU1[ind[0], ind[1]]):
                    nan_new_act.append(ind)
                new_act.append(ind)

        if len(nan_new_act) > 1:
            nan_active_indices = np.array(nan_new_act)
            rand_active_ind = np.random.choice(range(len(nan_active_indices)), size=1, replace=False)

            rand_active_ind1 = nan_active_indices[rand_active_ind][0][0]
            rand_active_ind2 = nan_active_indices[rand_active_ind][0][1]
            return rand_active_ind1, rand_active_ind2

        if np.any(np.isnan(KnownU1)):
            rand_ind1 = np.argwhere(np.isnan(KnownU1))[0][0]
            rand_ind2 = np.argwhere(np.isnan(KnownU1))[0][1]
            return rand_ind1, rand_ind2

    if stratnum == 3:

        ind1, ind2 = np.unravel_index(np.argmax(OptPhi, axis=None), (n, n))

        if np.isnan(KnownU1[ind1, ind2]):
            return ind1, ind2

        if 0 <= ind2-1 :
            if np.isnan(KnownU1[ind1,ind2-1]):
                return ind1,ind2-1

        if ind2+1 < n:
            if np.isnan(KnownU1[ind1,ind2+1]):
                return ind1,ind2+1

        if ind1+1 < n:
            if np.isnan(KnownU1[ind1+1, ind2]):
                return ind1+1, ind2

        if 0 <= ind1-1 :
            if np.isnan(KnownU1[ind1-1, ind2]):
                return ind1-1, ind2

        if np.any(np.isnan(KnownU1)):
            rand_ind1 = np.argwhere(np.isnan(KnownU1))[0][0]
            rand_ind2 = np.argwhere(np.isnan(KnownU1))[0][1]
            return rand_ind1, rand_ind2


    if stratnum == 4:
        if np.any(np.isnan(KnownU1)):
            rand_ind1 = np.argwhere(np.isnan(KnownU1))[0][0]
            rand_ind2 = np.argwhere(np.isnan(KnownU1))[0][1]
            return rand_ind1, rand_ind2

    if stratnum == 5:
        ind1, ind2 = np.unravel_index(np.argmax(OptPhi, axis=None), (n, n))

        if np.isnan(KnownU1[ind1, ind2]):
            return ind1, ind2

        if np.any(np.isnan(KnownU1)):
            rand_ind1 = np.argwhere(np.isnan(KnownU1))[0][0]
            rand_ind2 = np.argwhere(np.isnan(KnownU1))[0][1]
            return rand_ind1, rand_ind2
def setup(n):

    game = rand.randint(-12500, 12501, (n, n)) / 100000

    MaxPotential = np.max(game)
    MinPotential = np.min(game)
    NashIndices = FindNash(game)

    UnknownU2 = np.zeros((n, n))
    UnknownU1 = np.zeros((n, n))

    UnknownU2[:, -1] = 0
    UnknownU1[-1, :] = 0

    for i in range(n - 1):
        UnknownU2[:, -2 - i] = UnknownU2[:, -1 - i] + game[:, -2 - i] - game[:, -1 - i]
        UnknownU1[-2 - i, :] = UnknownU1[-1 - i, :] + game[-2 - i, :] - game[-1 - i]

    UnknownU1 += 0.25
    UnknownU2 += 0.25

    Phi = np.eye(n) - 1 / n * np.ones((n, n))
    Xi = 1 / n * np.ones((n, n))

    UnknownY1 = np.matmul(Phi, np.matmul(UnknownU1, Xi))
    UnknownY2 = np.matmul(Xi, np.matmul(UnknownU2, Phi))
    UnknownYP = np.matmul(Phi, np.matmul(UnknownU1, Phi))
    UnknownGame = UnknownYP + UnknownY1 + UnknownY2

    PhiMax = np.max(UnknownGame)
    PhiMin = np.min(UnknownGame)

    # Prepare arrays for known samples
    KnownU2 = np.full((n, n), np.nan)
    KnownU1 = np.full((n, n), np.nan)

    OptPhi = np.ones((n,n))*(PhiMax+2)
    PesPhi = np.ones((n,n))*(PhiMin-2)

    OptU1 = np.ones((n,n)) * 0.5
    PesU1 = np.ones((n, n)) * 0
    OptU2 = np.ones((n, n)) * 0.5
    PesU2 = np.ones((n, n)) * 0


    return [UnknownU1, UnknownU2, KnownU1, KnownU2, OptPhi, PesPhi ,OptU1,PesU1,OptU2,PesU2],UnknownGame, PhiMax, PhiMin
def init_strat(matrices, n, PhiMax, PhiMin, UnknownGame):

    matrices = sample(0, 0, matrices, n, PhiMax, PhiMin, UnknownGame)

    return matrices
def main(**kwargs):

    # Dimension of Problem
    n = kwargs.get("dimension")

    s = kwargs.get("strategy")

    t_max = kwargs.get("timesteps")

    t = 1

    Vs = []
    Percent = []
    Nash = []
    Gaps = []
    PercentBoundedPhi = []

    matrices, UnknownGame, PhiMax, PhiMin = setup(n)

    NashIndices = FindNash(UnknownGame)

    matrices = init_strat(matrices, n, PhiMax, PhiMin, UnknownGame)

    UnknownU1, UnknownU2, KnownU1, KnownU2, OptPhi, PesPhi, OptU1, PesU1, OptU2, PesU2 = matrices

    while t<t_max:

        ind1, ind2 = np.unravel_index(np.argmax(matrices[4], axis=None),(n,n))

        Vs.append(UnknownGame[ind1, ind2])

        Gaps.append(np.count_nonzero(matrices[4] < np.max(matrices[5])) / (n ** 2) * 100)

        if [ind1,ind2] in NashIndices:
            Nash.append(1)
        else:
            Nash.append(0)

        Percent.append(100-(np.count_nonzero(np.isnan(matrices[2])) / matrices[2].size) * 100)

        PercentBoundedPhi.append(np.count_nonzero(matrices[4] < np.max(UnknownGame)) / (n ** 2) * 100)

        if not np.any(np.isnan(matrices[2])):
            break

        i,j = strategy(matrices[2], OptPhi,PesPhi,n,s)

        matrices = sample(i,j, matrices, n, PhiMax, PhiMin, UnknownGame)

        UnknownU1, UnknownU2, KnownU1, KnownU2, OptPhi, PesPhi, OptU1, PesU1, OptU2, PesU2 = matrices

        #Update time index
        t += 1


    print(np.sum(np.abs(OptPhi - PesPhi)))
    print(np.sum(np.abs(UnknownGame - OptPhi)))

    # create a figure and axis object
    fig, ax1 = plt.subplots()
    fig.set_figwidth(15)
    # plot the first array using the left y-axis
    ax1.plot(Vs, color='red')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('True Potential Value of Optimistic Potential Estimate Maximum', color='red')
    ax1.axhline(y=np.max(UnknownGame), color='r', linestyle='--')
    ax1.set_ylim([np.min(UnknownGame), np.max(UnknownGame)+0.01])

    # create a twin axis object on the right side
    ax2 = ax1.twinx()

    # plot the second array using the right y-axis
    ax2.plot(Percent, color='blue',label = 'Percent of Utility Values Sampled')
    ax2.set_ylim([0,100])
    ax2.fill_between(range(len(Nash)), -5, 5, where=Nash > np.zeros(len(Nash)), color='green', alpha=0.5)
    # plot the second array using the right y-axis
    ax2.plot(Gaps, color='orange', label='Percentage of Strategy Profiles "Non-Active"')
    ax2.plot(PercentBoundedPhi, color='purple', label='Percentage of Optimistic Phi < Phi_max')
    ax2.set_ylabel('%')
    ax2.spines.right.set_position(("axes", 1.05))

    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
              fancybox=True, shadow=True, ncol=5)

    # set the title of the plot
    plt.savefig("Figures/Test")

if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)