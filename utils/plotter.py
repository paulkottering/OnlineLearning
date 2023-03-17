import numpy as np
import matplotlib.pyplot as plt

def plot_one(Game,Vs,Percent,Nash,Gaps,PercentBoundedPhi):
    fig, ax1 = plt.subplots()
    fig.set_figwidth(15)
    # plot the first array using the left y-axis
    ax1.plot(Vs, color='red')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('True Potential Value of Optimistic Potential Estimate Maximum', color='red')
    ax1.axhline(y=np.max(Game.UnknownGame), color='r', linestyle='--')
    ax1.set_ylim([np.min(Game.UnknownGame), np.max(Game.UnknownGame)+0.01])

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

def plot_many(kwargs, Regrets,CumRegrets, Gaps,percent_sampled):

    fig, ax1 = plt.subplots()
    fig.set_figwidth(15)
    # plot the first array using the left y-axis
    #ax1.plot(np.mean(Regrets,axis=0), color='red')
    ax1.plot(np.mean(CumRegrets, axis=0), color='red')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cumulative Regret', color='red')

    # create a twin axis object on the right side
    ax2 = ax1.twinx()

    # plot the second array using the right y-axis
    ax2.set_ylim([0,100])

    # plot the second array using the right y-axis
    ax2.plot(np.mean(Gaps,axis=0), color='orange', label='Percentage of Strategy Profiles "Non-Active"')
    ax2.plot(np.mean(percent_sampled,axis=0), color='purple', label='Percentage of Utility Matrix Unsampled')
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

    n = kwargs.get("dimension")
    k = kwargs.get("players")
    sa = kwargs.get("sample_strategy")
    si = kwargs.get("initial_strategy")
    runs = kwargs.get("runs")
    title = str(n)+str(k)+"_"+sa+"_"+si+"_"+str(runs)+"_"
    plt.title(title)
    # set the title of the plot
    plt.savefig("Figures/Test_"+title)
