from matplotlib import pyplot as plt
import math
import seaborn as sns


def plot_layers_heads_attention(attns, mask=None):
    x = attns.shape[0]
    y = attns.shape[1]

    if mask is not None:

        assert attns.shape[:2] == mask.shape

        nplots = mask.sum()
        plot_grid_size = math.floor(math.sqrt(nplots))
        if plot_grid_size * plot_grid_size == nplots:
            nrows, ncols = plot_grid_size, plot_grid_size
        else:
            nrows, ncols = plot_grid_size + 1, plot_grid_size + 1
    else:
        nrows = attns.shape[0]
        ncols = attns.shape[1]

    figsize = (10, 10)
    if nrows * ncols > 25:
        figsize = (20, 20)
    fig, axes = plt.subplots(nrows, ncols, sharey=True, figsize=figsize)

    count = 0
    for i in range(x):
        for j in range(y):

            plt_x = count // nrows
            plt_y = count % ncols

            if mask is not None:
                if mask[i][j] > 0:
                    ax = axes[plt_x][plt_y]
                    ax.set_title(f"L: {i}, H: {j}")
                    sns.heatmap(attns[i][j], ax=ax, cbar=False)
                    count += 1
            else:
                ax = axes[plt_x][plt_y]
                ax.set_title(f"L: {i}, H: {j}")
                sns.heatmap(attns[i][j], ax=ax, cbar=False)
                count += 1
    plt.subplots_adjust(hspace=0.5)
    plt.show()
