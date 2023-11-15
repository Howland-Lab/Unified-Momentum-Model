import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate


# All the implementations
def integrate_x_cumtrapz(field: np.ndarray, dx: float) -> ArrayLike:
    return integrate.cumulative_trapezoid(field, dx=dx, axis=0, initial=0)


def integrate_y_cumtrapz(field: np.ndarray, dx: float) -> ArrayLike:
    return integrate.cumulative_trapezoid(field, dx=dx, axis=1, initial=0)


def derivative_x_4th_central(field: np.ndarray, dx: float) -> ArrayLike:
    out = np.zeros_like(field)
    out[2:-2, :] = (
        field[:-4, :] - 8 * field[1:-3, :] + 8 * field[3:-1, :] - field[4:, :]
    ) / (12 * dx)

    return out


def derivative_y_4th_central(field: np.ndarray, dx: float) -> ArrayLike:
    out = np.zeros_like(field)
    out[:, 2:-2] = (
        field[:, :-4] - 8 * field[:, 1:-3] + 8 * field[:, 3:-1] - field[:, 4:]
    ) / (12 * dx)

    return out


def derivative_x_2nd_central(field: np.ndarray, dx: float) -> ArrayLike:
    out = np.zeros_like(field)
    out[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dx)
    return out


def derivative_y_2nd_central(field: np.ndarray, dx: float) -> ArrayLike:
    out = np.zeros_like(field)
    out[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dx)
    return out


def derivative_x_1st_forward(field: np.ndarray, dx: float) -> ArrayLike:
    out = np.zeros_like(field)
    out[1:, :] = (field[1:, :] - field[:-1, :]) / (dx)
    return out


def derivative_x_2nd_forward(field: np.ndarray, dx: float) -> ArrayLike:
    out = np.zeros_like(field)
    out[2:, :] = (3 * field[2:, :] - 4 * field[1:-1, :] + field[:-2]) / (2 * dx)
    return out


# The chosen implementations
def derivative_x(field: np.ndarray, dx: float) -> ArrayLike:
    return derivative_x_2nd_central(field, dx)
    # return derivative_x_2nd_forward(field, dx)


def derivative_y(field: np.ndarray, dx: float) -> ArrayLike:
    return derivative_y_2nd_central(field, dx)


def integrate_x(field: np.ndarray, dx: float) -> ArrayLike:
    return integrate_x_cumtrapz(field, dx)


def integrate_y(field: np.ndarray, dx: float) -> ArrayLike:
    return integrate_y_cumtrapz(field, dx)
