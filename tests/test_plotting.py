import os

from acquisition_plotting import (
    plot_objective,
    plot_evaluations,
)


def test_plot_2d(tmpdir, result2D):
    plot_evaluations(
        result2D, plot_dims=["x", "y"]).savefig(f"{tmpdir}/2deval.png", bbox_inches="tight")
    plot_objective(result2D).savefig(f"{tmpdir}/2dobj.png", bbox_inches="tight")
    assert os.path.exists(f"{tmpdir}/2deval.png")
    assert os.path.exists(f"{tmpdir}/2dobj.png")


def test_plot_Nd(tmpdir, resultND):
    plot_evaluations(resultND).savefig(f"{tmpdir}/Ndeval.png", bbox_inches="tight")
    plot_objective(resultND, n_samples=50).savefig(f"{tmpdir}/Ndobj.png", bbox_inches="tight")
    assert os.path.exists(f"{tmpdir}/Ndeval.png")
    assert os.path.exists(f"{tmpdir}/Ndobj.png")
