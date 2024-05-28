from .plot_objective import plot_objective
from .plot_evaluations import plot_evaluations
from .space import Space, Real
from scipy.optimize import OptimizeResult
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from .utils import _get_param_labels

def plot_trieste_objective(in_pts,
        out_pts,
        trieste_model,
        trieste_space,
        truth: Dict[str, float] = None,
        **kwargs
) -> plt.Figure:
    """
    Plot the evaluation matrix --> a corner plot of the parameters,
    colored by the order in which they were evaluated.
    """
    labels, truths = _get_param_labels(truth)
    res = _trieste_to_scipy_res(in_pts, out_pts, trieste_space, trieste_model, labels)

    fig = plot_objective(
        res,
        dimensions=labels,
        n_points=kwargs.get("n_points", 50),
        n_samples=kwargs.get("n_samples", 50),
        levels=kwargs.get("levels", 10),
        zscale=kwargs.get("zscale", "linear"),
        **kwargs
    )
    if truths:
        _add_truths_to_ax(fig.get_axes(), truths)
    return fig


def plot_trieste_evaluations(
        in_pts,
        out_pts,
        trieste_model,
        trieste_space,
        truth: Dict = None,
        **kwargs
) -> plt.Figure:
    """
    Plot the evaluation matrix --> a corner plot of the parameters,
    colored by the order in which they were evaluated.
    """
    labels, truths = _get_param_labels(truth)
    res = _trieste_to_scipy_res(in_pts, out_pts, trieste_space, trieste_model, labels)
    fig = plot_evaluations(res)
    if truths:
        _add_truths_to_ax(fig.get_axes(), truths)
    return fig


def _add_truths_to_ax(ax, truths, ):
    labels, tru_vals = _get_param_labels(truths)
    n_dims = len(tru_vals)
    # t_vals = np.array([tru_vals])
    # overplot_lines(fig, t_vals, color="tab:orange")
    # overplot_points(
    #     fig,
    #     [[np.nan if t is None else t for t in t_vals]],
    #     marker="s",
    #     color="tab:orange"
    # )
    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:  # diagonal
                if n_dims == 1:
                    ax_ = ax
                else:
                    ax_ = ax[i, i]
                ax_.vlines(
                    tru_vals[i], *ax_.get_ylim(), color="tab:orange"
                )
            # lower triangle
            elif i > j:
                ax_ = ax[i, j]
                ax_.vlines(
                    tru_vals[j], *ax_.get_ylim(), color="tab:orange"
                )
                ax_.hlines(
                    tru_vals[i], *ax_.get_xlim(), color="tab:orange"
                )
                ax_.scatter(
                    tru_vals[j],
                    tru_vals[i],
                    c="tab:orange",
                    s=50,
                    lw=0.0,
                    marker="s",
                )


def _trieste_to_scipy_res(x, y, trieste_space, trieste_model, param_names=None):
    """
    Create a SciPyOptimizeResult object from the given y_pts and trieste_model.

    trieste_space

    """
    bounds = Space(
        [
            Real(trieste_space.lower[i].numpy(), trieste_space.upper[i].numpy())
            for i in range(trieste_space.dimension.numpy())
        ]
    )

    if param_names is not None:
        bounds = Space([Real(*b, name=n) for b, n in zip(bounds, param_names)])

    min_idx = np.argmin(y)
    return OptimizeResult(
        dict(
            fun=y[min_idx],
            x=x[min_idx],
            success=True,
            func_vals=y,
            x_iters=x,
            models=[trieste_model],
            space=bounds,
        )
    )