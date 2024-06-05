import os

from acquisition_plotting import plot_evaluations


def test_plot_2d(tmpdir, result2D):
    fig, _ = plot_evaluations(result2D, dim_labels=["x", "y"], truths=[3, 0])
    fig.savefig(f"{tmpdir}/2deval.png", bbox_inches="tight")
    assert os.path.exists(f"{tmpdir}/2deval.png")


def test_plot_Nd(tmpdir, resultND):
    fig, _ = plot_evaluations(resultND)
    fig.savefig(f"{tmpdir}/Ndeval.png", bbox_inches="tight")
    assert os.path.exists(f"{tmpdir}/Ndeval.png")
