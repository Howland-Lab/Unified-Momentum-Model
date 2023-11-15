from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike


class FixedPointIterationResult:
    def __init__(self, converged, niter, relax, max_resid, x=None):
        self.converged = converged
        self.niter = niter
        self.relax = relax
        self.max_resid = max_resid
        self.x = x


def fixedpointiteration(
    f: Callable[[ArrayLike, Any], np.ndarray],
    x0: np.ndarray,
    args=(),
    kwargs={},
    eps=0.00001,
    maxiter=100,
    relax=0,
) -> FixedPointIterationResult:
    """
    Performs fixed-point iteration on function f until residuals converge or max
    iterations is reached.

    Args:
        f (Callable): residual function of form f(x, *args) -> np.ndarray
        x0 (np.ndarray): Initial guess
        args (tuple): arguments to pass to residual function. Defaults to ().
        eps (float): Convergence tolerance. Defaults to 0.000001.
        maxiter (int): Maximum number of iterations. Defaults to 100.

    Raises:
        ValueError: Max iterations reached.

    Returns:
        np.ndarray: Solution to residual function.
    """
    for c in range(maxiter):
        residuals = f(x0, *args, **kwargs)

        x0 = [_x0 + (1 - relax) * _r for _x0, _r in zip(x0, residuals)]
        # x0 = x0 + (1 - relax) * residuals
        max_resid = np.nanmax(np.abs(residuals))

        if max_resid < eps:
            break
    else:
        return FixedPointIterationResult(False, c, relax, max_resid, x0)

    return FixedPointIterationResult(True, c, relax, max_resid, x0)


def adaptivefixedpointiteration(
    f: Callable[[np.ndarray, Any], np.ndarray],
    x0: np.ndarray,
    args=(),
    kwargs={},
    eps=0.00001,
    maxiter=100,
):
    for relax in [0.3, 0.9]:
        try:
            sol = fixedpointiteration(
                f,
                x0,
                args,
                kwargs,
                eps=eps,
                maxiter=maxiter,
                relax=relax,
            )
            converged = sol.converged
        except FloatingPointError:
            converged = False
        if sol.converged:
            return sol
    return FixedPointIterationResult(False, maxiter, np.nan, np.nan)
