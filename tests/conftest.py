import os

import matplotlib.pyplot as plt
import numpy as np
from pytest import fixture
from scipy.optimize import OptimizeResult, minimize
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C

from acquisition_plotting.space import Real, Space
from acquisition_plotting.utils import _evenly_sample

HERE = os.path.dirname(__file__)

DATA_SPACE_2D = Space([Real(-5.0, 4.0, name="x"), Real(-3.0, 7.0, name="y")])

N = 3
DATA_SPACE_ND = Space([Real(-5.0, 5.0, name=f"A[{i}]") for i in range(N)])


def generate_nd_data(nx=20, noise_sigma=0):
    samples = [_evenly_sample(DATA_SPACE_ND[i][1], nx) for i in range(N)]
    X = np.array(np.meshgrid(*[s[0] for s in samples])).reshape(N, -1)
    Z = multivariate_normal.pdf(X.T, mean=np.zeros(N), cov=np.eye(N))
    Z += noise_sigma * np.random.randn(*Z.shape)
    Z *= -1
    Z = Z.reshape(-1, 1)
    return X, Z


def generate_2d_data(nx=100, ny=100, noise_sigma=0):
    # Define the domain for the fit
    x, _ = _evenly_sample(DATA_SPACE_2D[0][1], nx)
    y, _ = _evenly_sample(DATA_SPACE_2D[1][1], ny)
    X, Y = np.meshgrid(x, y)

    # Define the 2D Gaussian function
    def gaussian(x, y, x0, y0, xalpha, yalpha, A):
        return A * np.exp(
            -(((x - x0) / xalpha) ** 2) - ((y - y0) / yalpha) ** 2
        )

    # Gaussian parameters: x0, y0, xalpha, yalpha, A
    gaussian_params = [
        (0, 2, 2.5, 5.4, 1.5),
        (-1, 4, 6, 2.5, 1.8),
        (-3, -0.5, 1, 2, 4),
        (3, 0.5, 2, 1, 5),
    ]

    # Generate the Z values for the Gaussian sum
    Z = np.zeros(X.shape)
    for params in gaussian_params:
        Z += gaussian(X, Y, *params)
    Z += noise_sigma * np.random.randn(*Z.shape)
    Z *= -1

    return np.array([X, Y]), Z


def minimize_gp(IN, OUT):
    # Reshape the input data
    X = np.array([IN[i].flatten() for i in range(np.shape(IN)[0])]).T
    y = OUT.flatten()

    # Define the kernel
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))

    # Initialize the Gaussian Process Regressor
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
    gp.fit(X, y)

    return gp


@fixture
def tmpdir():
    temp = os.path.join(HERE, "out_test")
    os.makedirs(temp, exist_ok=True)
    return temp


@fixture
def result2D():
    IN, OUT = generate_2d_data(15, 15, noise_sigma=0.0)
    gp = minimize_gp(IN, OUT)

    random_in = np.array(DATA_SPACE_2D.rvs(250))
    random_out = gp.predict(random_in, return_std=False)
    fig = plot_gp_and_train_data(IN, OUT, random_in, random_out)
    fig.savefig(os.path.join(HERE, "out_test", "trained_gp.png"))
    return make_scipy_optmize_result(random_in, random_out, DATA_SPACE_2D, gp)


@fixture
def resultND():
    IN, OUT = generate_nd_data(10, noise_sigma=0.0)
    gp = minimize_gp(IN, OUT)

    random_in = np.array(DATA_SPACE_ND.rvs(250))
    random_out = gp.predict(random_in, return_std=False)
    return make_scipy_optmize_result(random_in, random_out, DATA_SPACE_ND, gp)


def plot_gp_and_train_data(
    train_in: np.ndarray,
    train_out: np.ndarray,
    random_in: np.ndarray,
    random_out: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    contours = axes[0].contourf(
        train_in[0], train_in[1], train_out, cmap="plasma"
    )
    axes[0].set_title("True Function")
    axes[1].tricontourf(
        random_in[:, 0], random_in[:, 1], random_out, cmap="plasma"
    )
    axes[1].set_title("GP Prediction")
    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
    fig.colorbar(contours, ax=axes.ravel().tolist())
    return fig


def make_scipy_optmize_result(x, y, space, model) -> OptimizeResult:
    min_idx = np.argmin(y)
    return OptimizeResult(
        dict(
            fun=y[min_idx],
            x=x[min_idx],
            success=True,
            func_vals=y,
            x_iters=x,
            models=[model],
            space=space,
        )
    )
