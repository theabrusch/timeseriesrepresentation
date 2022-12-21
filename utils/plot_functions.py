import matplotlib.pyplot as plt


def plot_contrastive_losses(output, savepath):

    for loss in output.keys():
        plt.plot(output[loss], label = loss)
    
    plt.legend()
    plt.savefig(savepath)