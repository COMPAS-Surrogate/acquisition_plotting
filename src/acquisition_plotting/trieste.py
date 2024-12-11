from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult

from .plot_evaluations import plot_evaluations
from .plot_objective import plot_objective
from .space import Real, Space
from .utils import _get_param_labels


def plot_trieste_objective(
    in_pts,
    out_pts,
    trieste_model,
    trieste_space,
    truths: Dict[str, float] = None,
    dim_labels=None,
    **kwargs,
) -> plt.Figure:
    """
    Plot the evaluation matrix --> a corner plot of the parameters,
    colored by the order in which they were evaluated.
    """
    labels, true_vals = _get_param_labels(truths)
    dim_labels = dim_labels or labels
    res = _trieste_to_scipy_res(
        in_pts, out_pts, trieste_space, trieste_model, labels
    )

    kwgs = kwargs.copy()
    kwgs["n_points"] = kwgs.get("n_points", 50)
    kwgs["n_samples"] = kwgs.get("n_samples", 50)
    kwgs["levels"] = kwgs.get("levels", 10)
    kwgs["zscale"] = kwgs.get("zscale", "linear")
    kwgs["reference_param"] = kwgs.get("reference_param", true_vals)

    fig, _ = plot_objective(
        res,
        dim_labels=dim_labels,
        **kwgs,
    )
    return fig


def plot_trieste_evaluations(
    in_pts,
    out_pts,
    trieste_model,
    trieste_space,
    truths: Dict = None,
    dim_labels=None,
    **kwargs,
) -> plt.Figure:
    """
    Plot the evaluation matrix --> a corner plot of the parameters,
    colored by the order in which they were evaluated.
    """
    labels, true_vals = _get_param_labels(truths)
    dim_labels = dim_labels or labels
    res = _trieste_to_scipy_res(
        in_pts, out_pts, trieste_space, trieste_model, labels
    )
    fig, _ = plot_evaluations(
        res, truths=true_vals, dim_labels=dim_labels, **kwargs
    )
    return fig


def _trieste_to_scipy_res(
    x, y, trieste_space, trieste_model, param_names=None
):
    """
    Create a SciPyOptimizeResult object from the given y_pts and trieste_model.

    trieste_space

    """
    n_dims = x.shape[1]
    bounds = _trieste_to_scipy_space(n_dims, trieste_space, param_names)
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


def _trieste_to_scipy_space(n_dims, trieste_space, param_names=None):
    bounds = Space(
        [
            Real(
                trieste_space.lower[i].numpy(), trieste_space.upper[i].numpy()
            )
            for i in range(n_dims)
        ]
    )

    if param_names is not None:
        bounds = Space(
            [
                Real(
                    trieste_space.lower[i].numpy(),
                    trieste_space.upper[i].numpy(),
                    name=param_names[i],
                )
                for i in range(n_dims)
            ]
        )
    return bounds
