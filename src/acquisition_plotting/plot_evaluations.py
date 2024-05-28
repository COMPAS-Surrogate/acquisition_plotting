# -*- encoding: UTF-8 -*-
"""Plotting functions."""
import sys
import numpy as np
from itertools import count
from functools import partial
from scipy.optimize import OptimizeResult
from .utils import _get_fig


from collections import Counter

# For plot tests, matplotlib must be set to headless mode early
if 'pytest' in sys.modules:
    import matplotlib

    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator, FuncFormatter  # noqa: E402
from .utils import _format_scatter_plot_axes, _map_categories



def plot_evaluations(result:OptimizeResult, bins=20, dimensions=None, plot_dims=None):
    """Visualize the order in which points were sampled during optimization.

    This creates a 2-d matrix plot where the diagonal plots are histograms
    that show the distribution of samples for each search-trieste_space dimension.

    The plots below the diagonal are scatter-plots of the samples for
    all combinations of search-trieste_space dimensions.

    The order in which samples
    were evaluated is encoded in each point's color.

    A red star shows the best found parameters.

    Parameters
    ----------
    result : `OptimizeResult`
        The optimization results from calling e.g. `gp_minimize()`.

    bins : int, bins=20
        Number of bins to use for histograms on the diagonal.

    dimensions : list of str, default=None
        Labels of the dimension
        variables. `None` defaults to `trieste_space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    plot_dims : list of str and int, default=None
        List of dimension names or dimension indices from the
        search-space dimensions to be included in the plot.
        If `None` then use all dimensions except constant ones
        from the search-trieste_space.

    Returns
    -------
    ax : `Matplotlib.Axes`
        A 2-d matrix of Axes-objects with the sub-plots.

    """
    space = result.space
    # Convert categoricals to integers, so we can ensure consistent ordering.
    # Assign indices to categories in the order they appear in the Dimension.
    # Matplotlib's categorical plotting functions are only present in v 2.1+,
    # and may order categoricals differently in different plots anyway.
    samples, minimum, iscat = _map_categories(space, result.x_iters, result.x)
    order = range(samples.shape[0])

    if plot_dims is None:
        # Get all dimensions.
        plot_dims = []
        for row in range(space.n_dims):
            if space.dimensions[row].is_constant:
                continue
            plot_dims.append((row, space.dimensions[row]))
    else:
        plot_dims = space[plot_dims]
    # Number of search-trieste_space dimensions we are using.
    n_dims = len(plot_dims)
    if dimensions is not None:
        assert len(dimensions) == n_dims

    fig, ax = plt.subplots(n_dims, n_dims,
                           figsize=(2 * n_dims, 2 * n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                index, dim = plot_dims[i]
                if iscat[j]:
                    bins_ = len(dim.categories)
                elif dim.prior == 'log-uniform':
                    low, high = space.bounds[index]
                    bins_ = np.logspace(np.log10(low), np.log10(high), bins)
                else:
                    bins_ = bins
                if n_dims == 1:
                    ax_ = ax
                else:
                    ax_ = ax[i, i]
                ax_.hist(samples[:, index], bins=bins_,
                         range=None if iscat[j] else dim.bounds)

            # lower triangle
            elif i > j:
                index_i, dim_i = plot_dims[i]
                index_j, dim_j = plot_dims[j]
                ax_ = ax[i, j]
                ax_.scatter(samples[:, index_j], samples[:, index_i],
                            c=order, s=40, lw=0., cmap='viridis')
                ax_.scatter(minimum[index_j], minimum[index_i],
                            c=['r'], s=100, lw=0., marker='*')

    # Make various adjustments to the plots.
    return _get_fig(_format_scatter_plot_axes(ax, space, ylabel="Number of samples",
                                     plot_dims=plot_dims,
                                     dim_labels=dimensions))

