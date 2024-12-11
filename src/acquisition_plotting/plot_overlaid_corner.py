"""Help plot overlaid corners"""

import warnings
from typing import Dict, List, Union

import corner
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from corner.core import _get_fig_axes

from .utils import _add_legend_to_grid

__all__ = ["plot_overlaid_corner"]

warnings.filterwarnings("ignore")

BINS1D = 30
CORNER_KWARGS = dict(
    smooth=0.99,
    smooth1d=0.5,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color="tab:orange",
    quantiles=None,
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
    bins=BINS1D,
)


def plot_overlaid_corner(
    samples_list: List[pd.DataFrame],
    sample_labels: List[str],
    axis_labels: List[str] = None,
    colors: List[str] = None,
    fname: str = "corner.png",
    truths: Union[Dict[str, float], List[float]] = None,
    annotate: str = "",
):
    """Plots multiple corners on top of each other"""
    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    min_len = min([len(s) for s in samples_list])
    # drop the 'log_likelhood' 'log_prior' columns from samples
    samples_list = [
        samples.drop(columns=["log_likelihood", "log_prior"])
        for samples in samples_list
    ]

    # make the samples the same len --> min_len
    samples_list = [
        (
            samples.sample(n=min_len, replace=False)
            if len(samples) > min_len
            else samples
        )
        for samples in samples_list
    ]

    if colors is None:
        colors = [f"C{i}" for i in range(n)]

    if axis_labels is None:
        axis_labels = samples_list[0].columns

    truths = list(truths.values()) if isinstance(truths, dict) else truths
    CORNER_KWARGS.update(
        labels=axis_labels,
        ranges=get_axes_ranges(samples_list, truths),
        truths=truths,
    )

    fig = corner.corner(
        samples_list[0].values,
        color=colors[0],
        **CORNER_KWARGS,
    )
    _, dims = samples_list[0].values.shape

    for idx in range(1, n):
        s = samples_list[idx].values
        if dims == 1:
            fig.gca().hist(
                s,
                bins=BINS1D,
                color=colors[idx],
                histtype="step",
                label=sample_labels[idx],
            )
            fig.legend()
        else:
            fig = corner.corner(
                s,
                fig=fig,
                color=colors[idx],
                **CORNER_KWARGS,
            )

    if dims > 1:
        axes, _ = _get_fig_axes(fig, dims)
        lgd_labels = {sample_labels[idx]: colors[idx] for idx in range(n)}
        _add_legend_to_grid(axes, lgd_labels)

    if annotate:
        axes, _ = _get_fig_axes(fig, dims)
        # annotate at bottom left corner of ax[0,-1]
        axes[0, -1].text(
            0.1,
            0.9,
            annotate,
            ha="left",
            va="bottom",
            transform=axes[0, -1].transAxes,
        )

    fig.savefig(fname)
    plt.close(fig)


def _get_data_ranges(data: pd.DataFrame) -> List[List[float]]:
    """Get the ranges of the data"""
    return [[data[col].min(), data[col].max()] for col in data.columns]


def get_axes_ranges(
    samples_list: List[pd.DataFrame],
    truths: List[float] = {},
    truth_thres=0.1,
) -> List[List[float]]:
    """Get the ranges of the data"""
    ranges = [_get_data_ranges(samples) for samples in samples_list]
    if truths:
        ranges.append(
            [
                [
                    t - truth_thres * t,
                    t + truth_thres * t,
                ]
                for t in truths
            ]
        )
    ranges = np.array(ranges)

    # set ax_range based on max and mins of all col
    ax_ranges = np.array(
        [
            [np.min(ranges[:, :, 0]), np.max(ranges[:, :, 1])]
            for _ in range(len(samples_list[0].columns))
        ]
    )

    return ax_ranges
