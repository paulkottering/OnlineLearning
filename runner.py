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
    parser.add_argument("-s", "--strategynumber", default=5, type=int,
                        help = 'Number of Strategies for each player')
    parser.add_argument("-t", "--timesteps", default=100, type=int,
                        help='Number of timesteps')
    parser.add_argument("-k", "--optimismconstant", default=0.5, type=float,
                        help='Optimistic Constant')
    return parser.parse_args()

def OptYPInitCalc_old(mat, n,k):
    """
    Calculate the YP component
    """
    nex = np.zeros((n, n))
    for a in range(n):

        for b in range(n):
            val = 0

            for i in range(n):
                const1 = True if i == a else False

                for j in range(n):
                    const2 = True if j == b else False

                    if (not const1) and (not const2):
                        if not np.isnan(mat[i, j]):
                            val += mat[i, j] / (n ** 2)
                        else:
                            val += 1 / (n ** 2) * k

                    elif const1 and const2:
                        if not np.isnan(mat[i, j]):
                            val += mat[i, j] * (n - 1) ** 2 / (n ** 2)
                        else:
                            val += (n - 1) ** 2 / (n ** 2) * k
                    else:
                        if not np.isnan(mat[i, j]):
                            val += mat[i, j] * (-n + 1) / (n ** 2)
                        else:
                            val += (-n + 1) / (n ** 2) * (0.5-k)
            nex[a, b] = val
    return nex

def PesYPInitCalc_old(mat,n,k):
    """
    Calculate Pessimistic YP component
    """
    nex = np.zeros((n, n))
    for a in range(n):

        for b in range(n):
            val = 0

            for i in range(n):
                const1 = True if i == a else False

                for j in range(n):
                    const2 = True if j == b else False

                    if (not const1) and (not const2):
                        if not np.isnan(mat[i, j]):
                            val += mat[i, j] / (n ** 2)
                        else:
                            val += 1 / (n ** 2) * (0.5-k)

                    elif const1 and const2:
                        if not np.isnan(mat[i, j]):
                            val += mat[i, j] * (n - 1) ** 2 / (n ** 2)
                        else:
                            val += (n - 1) ** 2 / (n ** 2) * (0.5-k)

                    else:
                        if not np.isnan(mat[i, j]):
                            val += mat[i, j] * (-n + 1) / (n ** 2)
                        else:
                            val += (-n + 1) / (n ** 2) * (k)
            nex[a, b] = val
    return nex

def OptYPInitCalc(mat, n, k):
    """
    Calculate the YP component
    """
    nex = np.ones((n, n)) * ((n - 1) ** 2 / (n ** 2))

    a = 0
    for b in range(n):
        OptYPUpdate(mat, nex, n, a, b, mat[a, b], k)

    b = 0
    for a in range(n - 1):
        OptYPUpdate(mat, nex, n, a + 1, b, mat[a + 1, b], k)

    return nex

def PesYPInitCalc(mat, n, k):
    """
    Calculate the YP component
    """
    nex = -np.ones((n, n)) * ((n - 1) ** 2 / (n ** 2))

    a = 0
    for b in range(n):
        PesYPUpdate(mat, nex, n, a, b, k)

    b = 0
    for a in range(n - 1):
        PesYPUpdate(mat, nex, n, a + 1, b, k)

    return nex

def OptYPUpdate(mat, YP, n, a, b, newval,k):
    """
    Update the YP matrix based on a new sample in the utility matrix

    Arguments:
    mat -- the utility matrix
    YP -- the YP matrix
    n -- the size of the matrices
    a -- the row index of the new sample
    b -- the column index of the new sample
    newval -- new sample value

    Returns:
    mat -- the updated utility matrix
    YP -- the updated YP matrix
    """

    mat[a,b] = newval

    # Update the YP matrix using the new sample
    for i in range(n):
        # Check if the row index is the same as the new sample
        const1 = True if i == a else False
        for j in range(n):
            # Check if the column index is the same as the new sample
            const2 = True if j == b else False

            #Update the optimistic value with the true value with relevant weights
            if (not const1) and (not const2):
                YP[i, j] += mat[a, b] / (n ** 2) - 1 / (n ** 2) * k
            elif const1 and const2:
                YP[i, j] += mat[a, b] * (n - 1) ** 2 / (n ** 2) - (n - 1) ** 2 / (n ** 2) * k
            else:
                YP[i, j] += mat[a, b] * (-n + 1) / (n ** 2) - (-n + 1) / (n ** 2) * (0.5-k)
    return mat, YP

def PesYPUpdate(mat, YP, n, a, b,k):
    """
    Update the YP matrix based on a new sample in the utility matrix

    Arguments:
    mat -- the utility matrix
    YP -- the YP matrix
    n -- the size of the matrices
    a -- the row index of the new sample
    b -- the column index of the new sample
    newval -- new sample value

    Returns:
    mat -- the updated utility matrix
    YP -- the updated YP matrix
    """


    # Update the YP matrix using the new sample
    for i in range(n):
        # Check if the row index is the same as the new sample
        const1 = True if i == a else False
        for j in range(n):
            # Check if the column index is the same as the new sample
            const2 = True if j == b else False
            #Update the optimistic value with the true value with relevant weights
            if (not const1) and (not const2):
                YP[i, j] += mat[a, b] / (n ** 2) - 1 / (n ** 2) * (0.5-k)
            elif const1 and const2:
                YP[i, j] += mat[a, b] * (n - 1) ** 2 / (n ** 2) - (n - 1) ** 2 / (n ** 2) * (0.5-k)
            else:
                YP[i, j] += mat[a, b] * (-n + 1) / (n ** 2) - (-n + 1) / (n ** 2) * k
    return YP

def FindNash(game):
    one_max_indices = np.argmax(game, axis=0)
    two_max_indices = np.argmax(game, axis=1)

    NashIndices = []

    n = len(one_max_indices)

    for i in range(n):
        if two_max_indices[one_max_indices[i]] == i:
            NashIndices.append([i,one_max_indices[i]])

    return NashIndices

def main(**kwargs):

    # Dimension of Problem
    n = kwargs.get("strategynumber")

    game = rand.randint(-12500, 12501, (n, n)) / 100000
    #game = np.ones((n, n)) * -0.1
    #game[5,19] = 0.125

    MaxPotential = np.max(game)
    MinPotential = np.min(game)

    NashIndices = (FindNash(game))

    # Find max indices
    # Vmax1, Vmax2 = np.unravel_index(np.argmax(game, axis=None), game.shape)

    # Define the two unknown utility functions - rewards are between -0.25,0.25
    UnknownU2 = np.zeros((n, n))
    UnknownU1 = np.zeros((n, n))

    UnknownU2[:, -1] = 0
    UnknownU1[-1, :] = 0

    for i in range(n - 1):
        UnknownU2[:, -2 - i] = UnknownU2[:, -1 - i] + game[:, -2 - i] - game[:, -1 - i]
        UnknownU1[-2 - i, :] = UnknownU1[-1 - i, :] + game[-2 - i, :] - game[-1 - i]

    UnknownU1 += 0.25
    UnknownU2 += 0.25

    # Calculate the components for comparison
    Phi = np.eye(n) - 1 / n * np.ones((n, n))
    Xi = 1 / n * np.ones((n, n))

    UnknownY1 = np.matmul(Phi, np.matmul(UnknownU1, Xi))[:, 0]
    UnknownY2 = np.matmul(Xi, np.matmul(UnknownU2, Phi))[0, :]
    UnknownYP = np.matmul(Phi, np.matmul(UnknownU1, Phi))

    UnknownY1 = UnknownY1 - UnknownY1[0] + 0.5
    UnknownY2 = UnknownY2 - UnknownY2[0] + 0.5

    #Prepare arrays for savings the sampled utility difference sums
    Sum1 = np.zeros(n)
    Sum2 = np.zeros(n)

    Sum1[0] = 0.5
    Sum2[0] = 0.5

    # Prepare arrays for known samples
    KnownU2 = np.full((n, n), np.nan)
    KnownU1 = np.full((n, n), np.nan)

    # Initialize First Player Fixed Values
    KnownU1[0, 0] = UnknownU1[0, 0]
    KnownU2[0, 0] = UnknownU2[0, 0]

    # Update KnownU1, KnownU2, and Sum1/Sum2
    KnownU1[0, 1:] = UnknownU1[0, 1:]
    KnownU2[0, 1:] = UnknownU2[0, 1:]
    Sum1[1:] = KnownU1[0, 1:] - KnownU1[0, 0] + 0.5

    KnownU2[1:, 0] = UnknownU2[1:, 0]
    KnownU1[1:, 0] = UnknownU1[1:, 0]
    Sum2[1:] = KnownU2[1:, 0] - KnownU2[0, 0] + 0.5

    N1 = np.ones(n)
    N2 = np.ones(n)

    t = 1
    previous = 0
    eps = 0.01
    diff = 2

    Vs = []
    Ps = []
    Percent1 = []
    Percent2 = []
    Nash = []
    Gaps = []

    k = kwargs.get("optimismconstant")

    YP1 = OptYPInitCalc(KnownU1, n,k)
    YP2 = OptYPInitCalc(KnownU2, n,k)

    YP1Pes = PesYPInitCalc(KnownU1, n,k)
    YP2Pes = PesYPInitCalc(KnownU2, n,k)

    ## Loop Until we reach convergence

    t_max = kwargs.get("timesteps")

    active_indices = []
    for index in np.ndindex(game.shape):
        active_indices.append(list(index))

    while t < t_max:
        # Optimistic estimates of factor matrix vectors
        OptY1 = Sum1 / N1 + np.sqrt(np.log(t)/N1)
        OptY2 = Sum2 / N2 + np.sqrt(np.log(t)/N2)

        # Pessimistic estimates of factor matrix vectors
        PesY1 = Sum1 / N1 - np.sqrt(np.log(t) / N1)
        PesY2 = Sum2 / N2 - np.sqrt(np.log(t) / N2)

        YP = (YP1)

        YPes = (YP1Pes)

        # Optimistic potential matrix estimate
        CurrentPotentialEstimate = YP + np.array([OptY1] * len(OptY1)).T + np.array([OptY2] * len(OptY2))
        CurrentPotentialEstimate = YP

        # Pessimistic potential matrix estimate
        PessimisticPotentialEstimate = YPes + np.array([OptY1] * len(OptY1)).T + np.array([OptY2] * len(OptY2))
        PessimisticPotentialEstimate = YPes

        # Find maximum of potential matrix estimate
        ind1, ind2 = np.unravel_index(np.argmax(CurrentPotentialEstimate, axis=None),
                                      CurrentPotentialEstimate.shape)

        new_act = []

        for ind in active_indices:
            if CurrentPotentialEstimate[ind[0],ind[1]] > np.max(PessimisticPotentialEstimate):
                new_act.append(ind)

        one = False
        two = False

        if len(new_act) != 0:
            active_indices = np.array(new_act)
            rand_active_ind = np.random.choice(range(len(active_indices)), size=1, replace=False)

            rand_ind1 = active_indices[rand_active_ind][0][0]
            rand_ind2 = active_indices[rand_active_ind][0][1]

            if np.isnan(KnownU1[rand_ind1, rand_ind2]):
                KnownU1,YP1 = OptYPUpdate(KnownU1, YP1, n, rand_ind1, rand_ind2, UnknownU1[rand_ind1, rand_ind2],k)
                YP1Pes = PesYPUpdate(KnownU1, YP1Pes, n, rand_ind1, rand_ind2, k)

                One = KnownU1[rand_ind1, rand_ind2] - KnownU1[0, rand_ind2]
                Sum1[rand_ind1] += One + 0.5
                N1[rand_ind1] += 1
                one = True
            if np.isnan(KnownU2[rand_ind1, rand_ind2]):
                KnownU2,YP2 = OptYPUpdate(KnownU2, YP2, n, rand_ind1, rand_ind2, UnknownU2[rand_ind1, rand_ind2],k)
                YP2Pes = PesYPUpdate(KnownU2, YP2Pes, n, rand_ind1, rand_ind2 ,k)

                # Update samples for other components
                Two = KnownU2[rand_ind1, rand_ind2] - KnownU2[rand_ind1, 0]
                Sum2[rand_ind2] += Two + 0.5
                N2[rand_ind2] += 1
                two = True

        if (two is False) and (one is False) and (np.isnan(KnownU1).any()):
            print(np.argwhere(np.isnan(KnownU1))[0])
            rand_ind1 = np.argwhere(np.isnan(KnownU1))[0][0]
            rand_ind2 = np.argwhere(np.isnan(KnownU1))[0][1]
            KnownU1, YP1 = OptYPUpdate(KnownU1, YP1, n, rand_ind1, rand_ind2, UnknownU1[rand_ind1, rand_ind2], k)
            YP1Pes = PesYPUpdate(KnownU1, YP1Pes, n, rand_ind1, rand_ind2, k)

            One = KnownU1[rand_ind1, rand_ind2] - KnownU1[0, rand_ind2]
            Sum1[rand_ind1] += One + 0.5
            N1[rand_ind1] += 1

        PesGap = np.count_nonzero(CurrentPotentialEstimate < np.max(PessimisticPotentialEstimate))/(n**2) * 100

        #Maximum Potential Value
        MaxV = CurrentPotentialEstimate[ind1, ind2]

        #Sample new estimates
        if np.any(np.isnan(KnownU1[ind1,:])):
            other2 = np.nanargmin(KnownU1[ind1,:])
        else:
            other2 = rand.randint(0, n)

        if np.any(np.isnan(KnownU2[:, ind2])):
            other1 = np.nanargmin(KnownU2[:, ind2])
        else:
            other1 = rand.randint(0, n)

        if np.isnan(KnownU1[ind1, other2]):
            KnownU1,YP1 = OptYPUpdate(KnownU1, YP1, n, ind1, other2, UnknownU1[ind1, other2], k)
            YP1Pes = PesYPUpdate(KnownU1, YP1Pes, n, ind1, other2,  k)

        if np.isnan(KnownU2[other1, ind2]):
            KnownU2,YP2 = OptYPUpdate(KnownU2, YP2, n, other1, ind2, UnknownU2[other1, ind2],k)
            YP2Pes = PesYPUpdate(KnownU2, YP2Pes, n, other1, ind2, k)

        if np.isnan(KnownU1[ind1, ind2]):
            KnownU1,YP1 = OptYPUpdate(KnownU1, YP1, n, ind1, ind2, UnknownU1[ind1, ind2],k)
            YP1Pes = PesYPUpdate(KnownU1, YP1Pes, n, ind1, ind2, k)

        if np.isnan(KnownU2[ind1, ind2]):
            KnownU2,YP2 = OptYPUpdate(KnownU2, YP2, n, ind1, ind2, UnknownU2[ind1, ind2],k)
            YP2Pes = PesYPUpdate(KnownU2, YP2Pes, n, ind1, ind2, k)

        #Update samples for other components
        One = KnownU1[ind1, other2] - KnownU1[0, other2]
        Two = KnownU2[other1, ind2] - KnownU2[other1, 0]

        Sum1[ind1] += One + 0.5
        Sum2[ind2] += Two + 0.5

        # Update sample counter for each difference value
        N1[ind1] += 1
        N2[ind2] += 1

        #Update time index
        t += 1

        #Max Index Potential Value
        #Vs.append(game[ind1, ind2])
        Vs.append(UnknownYP[ind1,ind2])

        Ps.append(YP[ind1, ind2])

        #Append gap
        Gaps.append(PesGap)

        #Check if point is a Nash Eq
        if [ind1,ind2] in NashIndices:
            Nash.append(MaxPotential)
        else:
            Nash.append(MinPotential)

        # Calculate the percentage of NaN values
        Percent1.append((np.count_nonzero(np.isnan(KnownU1)) / KnownU1.size) * 100)
        Percent2.append((np.count_nonzero(np.isnan(KnownU2)) / KnownU2.size) * 100)


    diff = OptYPInitCalc_old(KnownU1, n,k) - UnknownYP
    print((np.count_nonzero(np.isnan(KnownU1))))
    print(np.sum(np.abs(np.matmul(Phi, np.matmul(KnownU1, Phi)) - UnknownYP)))
    print(np.sum(np.abs(diff)))
    print(np.sum(np.abs(YP-UnknownYP)))



    # create a figure and axis object
    fig, ax1 = plt.subplots()
    fig.set_figwidth(15)
    # plot the first array using the left y-axis
    ax1.plot(Vs, color='red')
    #ax1.plot(Ps, color='green')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('True YP Value of Optimistic YP Estimate Maximum', color='red')
    ax1.axhline(y=np.max(UnknownYP), color='r', linestyle='--')
    #ax1.axhline(y=np.max(game), color='r', linestyle='--')

    ax1.fill_between(range(t_max-1), Nash, where=(np.array(Nash) > np.min(game)), alpha=0.5, color='green')

    # create a twin axis object on the right side
    ax2 = ax1.twinx()

    # plot the second array using the right y-axis
    ax2.plot(Percent1, color='blue',label = 'Percent of U1 Values Sampled')
    #ax2.plot(Percent2, color='green',label = 'Percent of U2 Values Sampled')
    ax2.set_ylabel('% of U1 Matrix Sampled', color='blue')
    ax2.set_ylim([0,100])

    # create a twin axis object on the right side
    ax3 = ax1.twinx()

    # plot the second array using the right y-axis
    ax3.plot(Gaps, color='orange', label='Gap')
    ax3.set_ylabel('% of YP Optimistic Estimates less than Pessimistic YP Maximum', color='orange')
    ax3.spines.right.set_position(("axes", 1.05))

    # set the title of the plot
    plt.savefig("Figures/Test")

if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())
    # Execute main
    main(**args)