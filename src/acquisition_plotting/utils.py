import sys
from collections import Counter
from functools import partial
from itertools import count
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.optimize import OptimizeResult
from scipy.optimize import OptimizeResult as SciPyOptimizeResult

from .space import Categorical, Real, Space

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


def _get_param_labels(truths):
    labels, truth_vals = None, None
    if isinstance(truths, dict):
        # get dict without lnl key
        truths = {k: v for k, v in truths.items() if k != "lnl"}
        labels = list(truths.keys())
        truth_vals = list(truths.values())
    return labels, truth_vals


def _get_fig(ax):
    if isinstance(ax, np.ndarray):
        fig = ax.flatten()[0].get_figure()
    else:
        fig = ax.get_figure()
    return fig


def _format_scatter_plot_axes(ax, plot_dims, dim_labels=None):
    if isinstance(ax, (list, np.ndarray)):
        n_dims, _ = ax.shape
    else:
        n_dims = 1

    # Work out min, max of y axis for the diagonal so we can adjust
    # them all to the same value
    diagonal_ylim = _get_ylim_diagonal(ax)

    if dim_labels is None:
        dim_labels = [
            "$X_{%i}$" % i if d.name is None else d.name for i, d in plot_dims
        ]
    # Axes for categorical dim_labels are really integers; we have to
    # label them with the category names
    iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]

    # Deal with formatting of the axes
    for i in range(n_dims):  # rows
        for j in range(n_dims):  # columns
            if n_dims > 1:
                ax_ = ax[i, j]
            else:
                ax_ = ax
            index_i, dim_i = plot_dims[i]
            index_j, dim_j = plot_dims[j]
            if j > i:
                ax_.axis("off")
            elif i > j:  # off-diagonal plots
                # plots on the diagonal are special, like Texas. They have
                # their own range so do not mess with them.
                if not iscat[i]:  # bounds not meaningful for categoricals
                    ax_.set_ylim(*dim_i.bounds)
                if iscat[j]:
                    # partial() avoids creating closures in a loop
                    ax_.xaxis.set_major_formatter(
                        FuncFormatter(partial(_cat_format, dim_j))
                    )
                else:
                    ax_.set_xlim(*dim_j.bounds)
                if j == 0:  # only leftmost column (0) gets y labels
                    ax_.set_ylabel(dim_labels[i])
                    if iscat[i]:  # Set category labels for left column
                        ax_.yaxis.set_major_formatter(
                            FuncFormatter(partial(_cat_format, dim_i))
                        )
                else:
                    ax_.set_yticklabels([])

                # for all rows except ...
                if i < n_dims - 1:
                    ax_.set_xticklabels([])
                # ... the bottom row
                else:
                    [l.set_rotation(45) for l in ax_.get_xticklabels()]
                    ax_.set_xlabel(dim_labels[j])

                # configure plot for linear vs log-scale
                if dim_j.prior == "log-uniform":
                    ax_.set_xscale("log")
                else:
                    ax_.xaxis.set_major_locator(
                        MaxNLocator(6, prune="both", integer=iscat[j])
                    )

                if dim_i.prior == "log-uniform":
                    ax_.set_yscale("log")
                else:
                    ax_.yaxis.set_major_locator(
                        MaxNLocator(6, prune="both", integer=iscat[i])
                    )

            else:  # diagonal plots
                ax_.set_ylim(*diagonal_ylim)
                if not iscat[i]:
                    low, high = dim_i.bounds
                    ax_.set_xlim(low, high)
                ax_.yaxis.tick_right()

                # remove yticks labels and ticks
                ax_.set_yticklabels([])
                ax_.yaxis.set_ticks_position("none")
                ax_.set_ylabel("")

                ax_.xaxis.tick_top()
                ax_.xaxis.set_label_position("top")
                ax_.set_xlabel(dim_labels[j])

                if dim_i.prior == "log-uniform":
                    ax_.set_xscale("log")
                else:
                    ax_.xaxis.set_major_locator(
                        MaxNLocator(6, prune="both", integer=iscat[i])
                    )
                    if iscat[i]:
                        ax_.xaxis.set_major_formatter(
                            FuncFormatter(partial(_cat_format, dim_i))
                        )

    return ax


def _map_categories(space, points, minimum):
    """
    Map categorical values to integers in a set of points.

    Returns
    -------
    mapped_points : np.array, shape=points.shape
        A copy of `points` with categoricals replaced with their indices in
        the corresponding `Dimension`.

    mapped_minimum : np.array, shape (trieste_space.n_dims,)
        A copy of `minimum` with categoricals replaced with their indices in
        the corresponding `Dimension`.

    iscat : np.array, shape (trieste_space.n_dims,)
       Boolean array indicating whether dimension `i` in the `trieste_space` is
       categorical.
    """
    points = np.asarray(points, dtype=object)  # Allow slicing, preserve cats
    iscat = np.repeat(False, space.n_dims)
    min_ = np.zeros(space.n_dims)
    pts_ = np.zeros(points.shape)
    for i, dim in enumerate(space.dimensions):
        if isinstance(dim, Categorical):
            iscat[i] = True
            catmap = dict(zip(dim.categories, count()))
            pts_[:, i] = [catmap[cat] for cat in points[:, i]]
            min_[i] = catmap[minimum[i]]
        else:
            pts_[:, i] = points[:, i]
            min_[i] = minimum[i]
    return pts_, min_, iscat


def _evenly_sample(dim, n_points):
    """Return `n_points` evenly spaced points from a Dimension.

    Parameters
    ----------
    dim : `Dimension`
        The Dimension to sample from.  Can be categorical; evenly-spaced
        category indices are chosen in order without replacement (result
        may be smaller than `n_points`).

    n_points : int
        The number of points to sample from `dim`.

    Returns
    -------
    xi : np.array
        The sampled points in the Dimension.  For Categorical
        dim_labels, returns the index of the value in
        `dim.categories`.

    xi_transformed : np.array
        The transformed values of `xi`, for feeding to a trieste_model.
    """
    cats = np.array(getattr(dim, "categories", []), dtype=object)
    if len(cats):  # Sample categoricals while maintaining order
        xi = np.linspace(0, len(cats) - 1, min(len(cats), n_points), dtype=int)
        xi_transformed = dim.transform(cats[xi])
    else:
        bounds = dim.bounds
        # XXX use linspace(*bounds, n_points) after python2 support ends
        xi = np.linspace(bounds[0], bounds[1], n_points)
        xi_transformed = dim.transform(xi)
    return xi, xi_transformed


def _cat_format(dimension, x, _):
    """Categorical axis tick formatter function.  Returns the name of category
    `x` in `dimension`.  Used with `matplotlib.ticker.FuncFormatter`."""
    return str(dimension.categories[int(x)])


def _get_ylim_diagonal(ax):
    """Get the min / max of the ylim for all diagonal plots.
    This is used in _adjust_fig() so the ylim is the same
    for all diagonal plots.

    Parameters
    ----------
    ax : `Matplotlib.Axes`
        2-dimensional matrix with Matplotlib Axes objects.

    Returns
    -------
    ylim_diagonal : tuple(int)
        The common min and max ylim for the diagonal plots.

    """

    # Number of search-trieste_space dim_labels used in this plot.
    if isinstance(ax, (list, np.ndarray)):
        n_dims = len(ax)
        # Get ylim for all diagonal plots.
        ylim = [ax[row, row].get_ylim() for row in range(n_dims)]
    else:
        n_dim = 1
        ylim = [ax.get_ylim()]

    # Separate into two lists with low and high ylim.
    ylim_lo, ylim_hi = zip(*ylim)

    # Min and max ylim for all diagonal plots.
    ylim_min = np.min(ylim_lo)
    ylim_max = np.max(ylim_hi)

    return ylim_min, ylim_max


def _add_truths(ax, truths: List, color: str = "tab:orange"):
    n_dims = len(truths)

    for i in range(n_dims):
        for j in range(n_dims):
            # diagonal
            if i == j:
                ax_ = ax if n_dims == 1 else ax[i, i]
                ax_.vlines(truths[i], *ax_.get_ylim(), color=color)
            # lower triangle
            elif i > j:
                ax_ = ax[i, j]
                ax_.vlines(truths[j], *ax_.get_ylim(), color=color)
                ax_.hlines(truths[i], *ax_.get_xlim(), color=color)
                ax_.scatter(
                    truths[j],
                    truths[i],
                    c="tab:orange",
                    s=50,
                    lw=0.0,
                    marker="s",
                )

    return ax


def _get_dim_names(space) -> List[Tuple[int, str]]:
    plot_dims = []
    for row in range(space.n_dims):
        if space.dimensions[row].is_constant:
            continue
        plot_dims.append((row, space.dimensions[row]))
    return plot_dims


def _add_legend_to_grid(ax: np.ndarray, legend_labels: Dict[str, str]):
    handles = [Patch(facecolor=c, label=l) for l, c in legend_labels.items()]
    ax[0, -1].legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        bbox_to_anchor=(0, 0.1),
    )
    return ax
