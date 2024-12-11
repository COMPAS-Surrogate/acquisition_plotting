import os

import numpy as np
import pandas as pd

from acquisition_plotting import plot_overlaid_corner


def _generate_mock_posterior(n, sigma=1.0, mu=0.0, p=1):
    params = {f"p{i}": np.random.normal(mu, sigma, n) for i in range(p)}
    return pd.DataFrame(
        {
            **params,
            "log_likelihood": np.random.normal(0, 1, n),
            "log_prior": np.random.normal(0, 1, n),
        }
    )


def test_overlaid_corner(tmpdir):
    n = 1000
    # rs = [
    #     _generate_mock_posterior(n, p=2),
    #     _generate_mock_posterior(n, mu=1, sigma=0.5, p=2),
    # ]
    # plot_overlaid_corner(
    #     rs,
    #     ["r1", "r2"],
    #     colors=["r", "b"],
    #     fname=f"{tmpdir}/corner.png",
    #     truths=[0.1],
    # )
    rs = [
        _generate_mock_posterior(n, p=2),
        _generate_mock_posterior(n, mu=1, sigma=0.5, p=2),
    ]
    plot_overlaid_corner(
        rs,
        sample_labels=["r1", "r2"],
        colors=["r", "b"],
        fname=f"{tmpdir}/corner2.png",
        truths=[0.1, 0.1],
        annotate="NumPts 10",
    )
