# -*- encoding: UTF-8 -*-
"""Plotting functions."""
import sys
from collections import Counter
from functools import partial
from itertools import count

import numpy as np
from scipy.optimize import OptimizeResult

from .utils import _get_fig

# For plot tests, matplotlib must be set to headless mode early
if "pytest" in sys.modules:
    import matplotlib

    matplotlib.use("Agg")

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import (  # noqa: E402
    FuncFormatter,
    LogLocator,
    MaxNLocator,
)

from .utils import (
    _add_legend_to_grid,
    _add_truths,
    _format_scatter_plot_axes,
    _get_dim_names,
    _map_categories,
)


def plot_evaluations(
    result: OptimizeResult,
    bins=20,
    dim_labels=None,
    truths: List = None,
    cmap="viridis",
    truth_color: str = "tab:orange",
    minima_color: str = "tab:red",
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualize the order in which points were sampled during optimization.

    This creates a 2-d matrix plot where the diagonal plots are histograms
    that show the distribution of samples for each search-trieste_space dimension.

    The plots below the diagonal are scatter-plots of the samples for
    all combinations of search-trieste_space dim_labels.

    The order in which samples
    were evaluated is encoded in each point's color.

    A red star shows the best found parameters.

    Parameters
    ----------
    result : `OptimizeResult`
        The optimization results from calling e.g. `gp_minimize()`.

    bins : int, bins=20
        Number of bins to use for histograms on the diagonal.

    dim_labels : list of str, default=None
        Labels of the dimension
        variables. `None` defaults to `trieste_space.dim_labels[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    plot_dims : list of str and int, default=None
        List of dimension names or dimension indices from the
        search-space dim_labels to be included in the plot.
        If `None` then use all dim_labels except constant ones
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

    plot_dims = _get_dim_names(space)
    n_dims = len(plot_dims)

    fig, ax = plt.subplots(n_dims, n_dims, figsize=(2 * n_dims, 2 * n_dims))

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1
    )

    cbar = None
    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                index, dim = plot_dims[i]
                if iscat[j]:
                    bins_ = len(dim.categories)
                elif dim.prior == "log-uniform":
                    low, high = space.bounds[index]
                    bins_ = np.logspace(np.log10(low), np.log10(high), bins)
                else:
                    bins_ = bins
                if n_dims == 1:
                    ax_ = ax
                else:
                    ax_ = ax[i, i]
                ax_.hist(
                    samples[:, index],
                    bins=bins_,
                    range=None if iscat[j] else dim.bounds,
                )
                ax_.axvline(
                    minimum[index], linestyle="--", color=minima_color, lw=1
                )

            # lower triangle
            elif i > j:
                index_i, dim_i = plot_dims[i]
                index_j, dim_j = plot_dims[j]
                ax_ = ax[i, j]
                cbar = ax_.scatter(
                    samples[:, index_j],
                    samples[:, index_i],
                    c=order,
                    s=40,
                    lw=0.0,
                    cmap=cmap,
                )
                ax_.scatter(
                    minimum[index_j],
                    minimum[index_i],
                    c=[minima_color],
                    s=100,
                    lw=0.0,
                    marker="*",
                )

    # make adjustments to the plot
    ax = _format_scatter_plot_axes(ax, plot_dims, dim_labels)
    legend_labels = {"Observed Minima": minima_color}
    if truths:
        ax = _add_truths(ax, truths, truth_color)
        legend_labels["Injection"] = truth_color
    ax = _add_legend_to_grid(ax, legend_labels)

    # add cbar above last ax
    if cbar:
        cax = fig.colorbar(
            cbar,
            ax=ax[0, 1:],
            shrink=0.8,
            orientation="horizontal",
            location="top",
        )
        cax.set_label("Acquisition Order")

    fig = _get_fig(ax)
    return fig, ax
