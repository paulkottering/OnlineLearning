import numpy as np
import matplotlib.pyplot as plt


def plot_many(kwargs, regrets,cumulative_regret, Gaps,percent_sampled,game_mean,max_tuples):

    fig, ax1 = plt.subplots()
    fig.set_figwidth(15)
    # plot the first array using the left y-axis
    ax1.plot(np.mean(cumulative_regret, axis=0), color='red', label='Random Strategy')
    ax1.plot(game_mean*range(len(np.mean(cumulative_regret, axis=0))), color='blue', label='Strategy')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cumulative Regret', color='red')

    # create a twin axis object on the right side
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    #ax3.plot(np.mean(max_tuples, axis=0), color='green')
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
    title = "Number of Players: "+str(k)+"  Number of strategies per player:  "+str(n)+"  Sampling Strategy:  "+sa
    plt.title(title)
    # set the title of the plot
    plt.savefig("Figures/Test_"+str(n)+str(k)+"_"+sa+"_"+si+"_"+str(runs)+"_")
