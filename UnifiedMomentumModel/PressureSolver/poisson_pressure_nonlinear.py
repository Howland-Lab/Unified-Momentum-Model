from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from . import PressureSolver
from ..Utilities import Calculus, Geometry
from .poisson_pressure_linear import AnalyticalLinearSolution


def construct_f(dP: float, solver: PressureSolver.PressureSolver):
    fx = np.zeros(solver.shape)
    mask = np.abs(solver.y) <= 1.0
    N = mask.sum()
    fx[solver.Nx // 2, mask] = -dP * (solver.dy / solver.dx) / (2 / (N - 1))

    fy = np.zeros(solver.shape)
    return fx, fy


class NonLinearPoisson:
    def __init__(self, geometry: Geometry.Geometry, dP=1.0):
        self.pressuresolver = PressureSolver.Convolution(geometry)
        self.Lx = geometry.Lx
        self.Ly = geometry.Ly
        self.dx = geometry.dx
        self.dy = geometry.dy
        self.dP = dP
        self.geometry = geometry

        self.sol_linear = AnalyticalLinearSolution(dP, geometry)

        self.Nx = geometry.Nx
        self.Ny = geometry.Ny
        self.x = geometry.x
        self.shape = geometry.shape

        self.gx = np.zeros(self.shape)
        self.gy = np.zeros(self.shape)
        self.sol_NL = None

    def solve(
        self, max_iter=10000, eps=0.00001, relax=0.0, callback=None
    ) -> Tuple[PressureSolver.PressureSolution, ...]:
        converged = False

        # Solve for non-linear pressure term iteratively.
        for i in range(max_iter):
            self.sol_NL = self.pressuresolver.solve(self.gx, self.gy)

            _gx = -self.wx * self.dwxdx - self.wy * self.dwxdy
            _gy = -self.wx * self.dwydx - self.wy * self.dwydy

            self.res_gx = _gx - self.gx
            self.res_gy = _gy - self.gy

            self.gx += (1 - relax) * self.res_gx
            self.gy += (1 - relax) * self.res_gy

            if callback is not None:
                callback(self, i)

            max_res = np.max(np.abs(self.res_gx))

            if np.isnan(max_res):
                break

            elif max_res < eps:
                converged = True
                break

        sol_tot = PressureSolver.PressureSolution(
            self.Lx,
            self.Ly,
            self.dx,
            self.dy,
            self.geometry.xmesh,
            self.geometry.ymesh,
            self.sol_linear.p + self.sol_NL.p,
            self.gx,  # Note. this does not include fx
            self.gy,
            converged,
        )

        return sol_tot, self.sol_linear, self.sol_NL

    @property
    def p(self) -> ArrayLike:
        return self.sol_linear.p + self.sol_NL.p

    @property
    def p_NL(self) -> ArrayLike:
        return self.sol_NL.p

    @property
    def dwxdx(self) -> ArrayLike:
        return self.sol_linear.dwxdx + self.sol_NL.dwxdx

    @property
    def wx(self) -> ArrayLike:
        return self.sol_linear.wx + self.sol_NL.wx

    @property
    def dwxdy(self) -> ArrayLike:
        return Calculus.derivative_y(self.wx, self.dy)

    @property
    def dwydx(self) -> ArrayLike:
        return self.sol_linear.dwydx + self.sol_NL.dwydx

    @property
    def wy(self) -> ArrayLike:
        return self.sol_linear.wy + self.sol_NL.wy

    @property
    def dwydy(self) -> ArrayLike:
        return self.sol_linear.dwydy + self.sol_NL.dwydy
