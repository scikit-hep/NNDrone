import matplotlib.pyplot as plt

colours_list = ['blue', 'red', 'green', 'purple']
colours_list = ['xkcd:'+k for k in colours_list]


def hd_hist(data, name, x_range, y_range, xname, yname, bins, labels):
    for c, d in enumerate(data):
        plt.hist(d
                 , bins=bins
                 , color=colours_list[c]
                 , label=labels[c])

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.legend(loc='upper right')
    plt.savefig(name)
    plt.clf()
