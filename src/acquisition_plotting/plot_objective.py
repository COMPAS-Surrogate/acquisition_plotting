# -*- encoding: UTF-8 -*-
"""Plotting functions."""
import sys
import numpy as np
from itertools import count
from functools import partial
from scipy.optimize import OptimizeResult
from bilby.core.prior import PriorDict

from collections import Counter

# For plot tests, matplotlib must be set to headless mode early
if 'pytest' in sys.modules:
    import matplotlib

    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator, FuncFormatter  # noqa: E402


from .utils import _format_scatter_plot_axes, _map_categories, _evenly_sample, _get_fig

def plot_objective(result:OptimizeResult, levels=10, n_points=40, n_samples=250, size=2,
                   zscale='linear',  dimensions=None, plot_dims=None,
                   show_points=True, cmap='viridis_r'):
    """Plot a 2-d matrix with so-called Partial Dependence plots
    of the objective function. This shows the influence of each
    search-trieste_space dimension on the objective function.

    This uses the last fitted trieste_model for estimating the objective function.

    The diagonal shows the effect of a single dimension on the
    objective function, while the plots below the diagonal show
    the effect on the objective function when varying two dimensions.

    The Partial Dependence is calculated by averaging the objective value
    for a number of random samples in the search-trieste_space,
    while keeping one or two dimensions fixed at regular intervals. This
    averages out the effect of varying the other dimensions and shows
    the influence of one or two dimensions on the objective function.

    Also shown are small black dots for the points that were sampled
    during optimization.

    A red star indicates per default the best observed minimum, but
    this can be changed by changing argument ´minimum´.

    .. note::
          The Partial Dependence plot is only an estimation of the surrogate
          trieste_model which in turn is only an estimation of the true objective
          function that has been optimized. This means the plots show
          an "estimate of an estimate" and may therefore be quite imprecise,
          especially if few samples have been collected during the
          optimization
          (e.g. less than 100-200 samples), and in regions of the search-trieste_space
          that have been sparsely sampled (e.g. regions away from the optimum).
          This means that the plots may change each time you run the
          optimization and they should not be considered completely reliable.
          These compromises are necessary because we cannot evaluate the
          expensive objective function in order to plot it, so we have to use
          the cheaper surrogate trieste_model to plot its contour. And in order to
          show search-spaces with 3 dimensions or more in a 2-dimensional
          plot,
          we further need to map those dimensions to only 2-dimensions using
          the Partial Dependence, which also causes distortions in the plots.

    Parameters
    ----------
    result : `OptimizeResult`
        The optimization results from calling e.g. `gp_minimize()`.

    levels : int, default=10
        Number of levels to draw on the contour plot, passed directly
        to `plt.contourf()`.

    n_points : int, default=40
        Number of points at which to evaluate the partial dependence
        along each dimension.

    n_samples : int, default=250
        Number of samples to use for averaging the trieste_model function
        at each of the `n_points` when `sample_method` is set to 'random'.

    size : float, default=2
        Height (in inches) of each facet.

    zscale : str, default='linear'
        Scale to use for the z axis of the contour plots. Either 'linear'
        or 'log'.

    dimensions : list of str, default=None
        Labels of the dimension
        variables. `None` defaults to `trieste_space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    plot_dims : list of str and int, default=None
        List of dimension names or dimension indices from the
        search-trieste_space dimensions to be included in the plot.
        If `None` then use all dimensions except constant ones
        from the search-trieste_space.


    minimum : str or list of floats, default = 'result'
        Defines the values for the red points in the plots.
        Valid strings:

        - 'result' - Use best observed parameters


.

    show_points: bool, default = True
        Choose whether to show evaluated points in the
        contour plots.

    cmap: str or Colormap, default = 'viridis_r'
        Color map for contour plots. Passed directly to
        `plt.contourf()`

    Returns
    -------
    ax : `Matplotlib.Axes`
        A 2-d matrix of Axes-objects with the sub-plots.

    """
    # Here we define the values for which to plot the red dot (2d plot) and
    # the red dotted line (1d plot).
    # These same values will be used for evaluating the plots when
    # calculating dependence. (Unless partial
    # dependence is to be used instead).

    # Here we define the values for which to plot the red dot (2d plot) and
    # the red dotted line (1d plot).
    # These same values will be used for evaluating the plots when
    # calculating dependence. (Unless partial
    # dependence is to be used instead).
    space = result.space

    plot_dims = []
    for row in range(space.n_dims):
        if space.dimensions[row].is_constant:
            continue
        plot_dims.append((row, space.dimensions[row]))
    n_dims = len(plot_dims)
    x_vals = result.x
    samples = space.transform(space.rvs(n_samples=n_samples))

    x_samples, minimum, _ = _map_categories(space, result.x_iters, x_vals)



    if zscale == 'log':
        locator = LogLocator()
    elif zscale == 'linear':
        locator = None
    else:
        raise ValueError("Valid values for zscale are 'linear' and 'log',"
                         " not '%s'." % zscale)

    fig, ax = plt.subplots(n_dims, n_dims,
                           figsize=(size * n_dims, size * n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                index, dim = plot_dims[i]
                xi, yi = _partial_dependence_1D(space, result.models[-1],
                                                index,
                                                samples=samples,
                                                n_points=n_points)
                if n_dims > 1:
                    ax_ = ax[i, i]
                else:
                    ax_ = ax
                ax_.plot(xi, yi)
                ax_.axvline(minimum[index], linestyle="--", color="r", lw=1)

            # lower triangle
            elif i > j:
                index1, dim1 = plot_dims[i]
                index2, dim2 = plot_dims[j]
                ax_ = ax[i, j]
                xi, yi, zi = _partial_dependence_2D(space, result.models[-1],
                                                    index1, index2,
                                                    samples, n_points)
                ax_.contourf(xi, yi, zi, levels,
                             locator=locator, cmap=cmap)
                if show_points:
                    ax_.scatter(x_samples[:, index2], x_samples[:, index1],
                                c='k', s=10, lw=0.)
                ax_.scatter(minimum[index2], minimum[index1],
                            c=['r'], s=100, lw=0., marker='*')
    ylabel = "Partial dependence"

    # Make various adjustments to the plots.
    return _get_fig(_format_scatter_plot_axes(ax, space, ylabel=ylabel,
                                     plot_dims=plot_dims,
                                     dim_labels=dimensions))



def _partial_dependence_1D(space, model, i, samples, n_points=40):
    """
    Calculate the partial dependence for a single dimension.

    This uses the given trieste_model to calculate the average objective value
    for all the samples, where the given dimension is fixed at
    regular intervals between its bounds.

    This shows how the given dimension affects the objective value
    when the influence of all other dimensions are averaged out.

    Parameters
    ----------
    space : `Space`
        The parameter trieste_space over which the minimization was performed.

    model
        Surrogate trieste_model for the objective function.

    i : int
        The dimension for which to calculate the partial dependence.

    samples : np.array, shape=(n_points, n_dims)
        Randomly sampled and transformed points to use when averaging
        the trieste_model function at each of the `n_points` when using partial
        dependence.

    n_points : int, default=40
        Number of points at which to evaluate the partial dependence
        along each dimension `i`.

    Returns
    -------
    xi : np.array
        The points at which the partial dependence was evaluated.

    yi : np.array
        The average value of the modelled objective function at
        each point `xi`.

    """
    # The idea is to step through one dimension, evaluating the trieste_model with
    # that dimension fixed and averaging either over random values or over
    # the given ones in x_val in all other dimensions.
    # (Or step through 2 dimensions when i and j are given.)
    # Categorical dimensions make this interesting, because they are one-
    # hot-encoded, so there is a one-to-many mapping of input dimensions
    # to transformed (trieste_model) dimensions.

    # dim_locs[i] is the (column index of the) start of dim i in
    # sample_points.
    # This is usefull when we are using one hot encoding, i.e using
    # categorical values
    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    def _calc(x):
        """
        Helper-function to calculate the average predicted
        objective value for the given trieste_model, when setting
        the index'th dimension of the search-trieste_space to the value x,
        and then averaging over all samples.
        """
        rvs_ = np.array(samples)  # copy
        # We replace the values in the dimension that we want to keep
        # fixed
        rvs_[:, dim_locs[i]:dim_locs[i + 1]] = x
        # In case of `x_eval=None` rvs conists of random samples.
        # Calculating the mean of these samples is how partial dependence
        # is implemented.
        return np.mean(model.predict(rvs_))

    xi, xi_transformed = _evenly_sample(space.dimensions[i], n_points)
    # Calculate the partial dependence for all the points.
    yi = [_calc(x) for x in xi_transformed]

    return xi, yi


def _partial_dependence_2D(space, model, i, j, samples, n_points=40):
    """
    Calculate the partial dependence for two dimensions in the search-trieste_space.

    This uses the given trieste_model to calculate the average objective value
    for all the samples, where the given dimensions are fixed at
    regular intervals between their bounds.

    This shows how the given dimensions affect the objective value
    when the influence of all other dimensions are averaged out.

    Parameters
    ----------
    space : `Space`
        The parameter trieste_space over which the minimization was performed.

    model
        Surrogate trieste_model for the objective function.

    i : int
        The first dimension for which to calculate the partial dependence.

    j : int
        The second dimension for which to calculate the partial dependence.

    samples : np.array, shape=(n_points, n_dims)
        Randomly sampled and transformed points to use when averaging
        the trieste_model function at each of the `n_points` when using partial
        dependence.

    n_points : int, default=40
        Number of points at which to evaluate the partial dependence
        along each dimension `i` and `j`.

    Returns
    -------
    xi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.

    yi : np.array, shape=n_points
        The points at which the partial dependence was evaluated.

    zi : np.array, shape=(n_points, n_points)
        The average value of the objective function at each point `(xi, yi)`.
    """
    # The idea is to step through one dimension, evaluating the trieste_model with
    # that dimension fixed and averaging either over random values or over
    # the given ones in x_val in all other dimensions.
    # (Or step through 2 dimensions when i and j are given.)
    # Categorical dimensions make this interesting, because they are one-
    # hot-encoded, so there is a one-to-many mapping of input dimensions
    # to transformed (trieste_model) dimensions.

    # dim_locs[i] is the (column index of the) start of dim i in
    # sample_points.
    # This is usefull when we are using one hot encoding, i.e using
    # categorical values
    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    def _calc(x, y):
        """
        Helper-function to calculate the average predicted
        objective value for the given trieste_model, when setting
        the index1'th dimension of the search-trieste_space to the value x
        and setting the index2'th dimension to the value y,
        and then averaging over all samples.
        """
        rvs_ = np.array(samples)  # copy
        rvs_[:, dim_locs[j]:dim_locs[j + 1]] = x
        rvs_[:, dim_locs[i]:dim_locs[i + 1]] = y
        return np.mean(model.predict(rvs_))

    xi, xi_transformed = _evenly_sample(space.dimensions[j], n_points)
    yi, yi_transformed = _evenly_sample(space.dimensions[i], n_points)
    # Calculate the partial dependence for all combinations of these points.
    zi = [[_calc(x, y) for x in xi_transformed] for y in yi_transformed]

    # Convert list-of-list to a numpy array.
    zi = np.array(zi)

    return xi, yi, zi


