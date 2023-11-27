from typing import Any, Callable, Protocol
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike


class FixedPointIterationCompatible(Protocol):
    def residual(self, *args, **kwargs):
        ...

    def initial_guess(self, *args, **kwargs):
        ...


@dataclass
class FixedPointIterationResult:
    converged: bool
    niter: int
    relax: float
    max_resid: float
    x: ArrayLike


def _fixedpointiteration(
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
        max_resid = [np.nanmax(np.abs(_r)) for _r in residuals]

        if all(_r < eps for _r in max_resid):
            converged = True
            break
    else:
        converged = False

    return FixedPointIterationResult(converged, c, relax, max_resid, x0)


def fixedpointiteration(
    max_iter: int = 100, tolerance: float = 1e-6, relaxation: float = 0.0
) -> Callable:
    """
    Class decorator which adds a __call__ method to the class which performs
    fixed-point iteration. The class must contain 2 mandatory methods and 1
    optional method:

    initial_guess(self, *args, **kwargs)
    residual(self, x, *args, **kwargs)
    post_process(self, result:FixedPointIterationResult) # Optional

    """

    def decorator(cls: FixedPointIterationCompatible) -> Callable:
        def call(self, *args, **kwargs):
            x0 = self.initial_guess(*args, **kwargs)
            result = _fixedpointiteration(
                self.residual,
                x0,
                args=args,
                kwargs=kwargs,
                eps=tolerance,
                maxiter=max_iter,
                relax=relaxation,
            )

            if hasattr(self, "post_process"):
                return self.post_process(result, *args, **kwargs)
            else:
                return result

        setattr(cls, "__call__", call)
        return cls

    return decorator


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
            sol = _fixedpointiteration(
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
