import matplotlib.pyplot as plt

colours_list = ['blue', 'red', 'green', 'purple']
colours_list = ['xkcd:'+k for k in colours_list]


def hd_hist(data, name, x_range, y_range, xname, yname, bins, labels, legend=False):
    for c, d in enumerate(data):
        plt.hist(d
                 , bins=bins
                 , color=colours_list[c]
                 , label=labels[c]
                 , alpha=0.5)

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel(xname)
    plt.ylabel(yname)
    if legend:
        plt.legend(loc='upper right')
    plt.savefig(name)
    plt.clf()


def scatter(x, y, xlim, ylim, xlabel, ylabel, title, savename, line=True):
    fig = plt.figure()
    plt.scatter(x, y)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if line:
        plt.plot([0.0, 1.0], [0.0, 1.0], 'k-')
    fig.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(savename)
    plt.clf()
