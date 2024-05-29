import os

from acquisition_plotting import plot_evaluations, plot_objective


def test_plot_2d(tmpdir, result2D):
    fig, _ = plot_evaluations(result2D, dim_labels=["x", "y"], truths=[3, 0])
    fig.savefig(f"{tmpdir}/2deval.png", bbox_inches="tight")
    fig, _ = plot_objective(
        result2D,
        truths=[3, 0],
    )
    fig.savefig(f"{tmpdir}/2dobj.png", bbox_inches="tight")
    assert os.path.exists(f"{tmpdir}/2deval.png")
    assert os.path.exists(f"{tmpdir}/2dobj.png")


def test_plot_Nd(tmpdir, resultND):
    fig, _ = plot_evaluations(resultND)
    fig.savefig(f"{tmpdir}/Ndeval.png", bbox_inches="tight")
    fig, _ = plot_objective(resultND, n_samples=50, truths=[0, 0, 0])
    fig.savefig(f"{tmpdir}/Ndobj.png", bbox_inches="tight")
    assert os.path.exists(f"{tmpdir}/Ndeval.png")
    assert os.path.exists(f"{tmpdir}/Ndobj.png")
