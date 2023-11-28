from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator

from ..Utilities import Calculus
from ..Utilities.FixedPointIteration import (
    FixedPointIterationResult,
    adaptivefixedpointiteration,
)
from ..Utilities.Geometry import EquidistantRectGridEven, Geometry
from .poisson_pressure_linear import AnalyticalLinearSolution
from .PressureSolver import Convolution, PressureSolution, PressureSolver


class LinearPoisson:
    def __init__(self, geometry: Optional[Geometry] = None):
        self.geom = geometry or EquidistantRectGridEven(60.0, 60.0, 0.1, 1.0)

    def __call__(self, dP):
        # Analytical solutions to pressure and velocity fields.
        p = (
            -1
            / (2 * np.pi)
            * dP
            * (
                np.arctan((1 - self.geom.ymesh) / self.geom.xmesh)
                + np.arctan((1 + self.geom.ymesh) / self.geom.xmesh)
            )
        )

        wx = -np.array(self.p)
        wx[(self.geom.xmesh > 0) & (np.abs(self.geom.ymesh) <= 1)] -= dP
        wy = (
            dP
            / (4 * np.pi)
            * np.log(
                (self.geom.xmesh**2 + (self.geom.ymesh + 1) ** 2)
                / (self.geom.xmesh**2 + (self.geom.ymesh - 1) ** 2)
            )
        )

        # Remove nans and infs
        p[np.isnan(self.p)] = 0
        wx[np.isnan(self.wx)] = 0
        wy[self.wy == -np.inf] = np.nanmin(self.wy[np.isfinite(self.wy)])
        wy[self.wy == np.inf] = np.nanmax(self.wy[np.isfinite(self.wy)])


@adaptivefixedpointiteration(max_iter=3, tolerance=0.00001, relaxations=[0, 0.1, 0.2])
class NonLinearPoisson:
    def __init__(
        self,
        accumulate=True,
        geometry: Optional[Geometry] = None,
        pressuresolver: Optional[PressureSolver] = None,
        analyticalsolver: Optional[AnalyticalLinearSolution] = None,
    ):
        self.accumulate = accumulate
        self.geom = geometry or EquidistantRectGridEven(60.0, 60.0, 0.1, 1.0)
        self.pressuresolver = pressuresolver or Convolution(self.geom)
        self.analyticalsolver = analyticalsolver or AnalyticalLinearSolution

    def pre_process(self, dP):
        self.sol_linear = self.analyticalsolver(dP, self.geom)
        self.pressure_fields = []

    def initial_guess(self, dP):
        gx = np.zeros(self.geom.shape)
        gy = np.zeros(self.geom.shape)
        return [gx, gy]

    def residual(self, x: List[ArrayLike], dP: float):
        gx, gy = x
        sol_NL = self.pressuresolver.solve(gx, gy)

        wx = self.sol_linear.wx + sol_NL.wx
        wy = self.sol_linear.wy + sol_NL.wy

        dwxdx = self.sol_linear.dwxdx + sol_NL.dwxdx
        dwxdy = Calculus.derivative_y(wx, self.geom.dy)
        dwydx = self.sol_linear.dwydx + sol_NL.dwydx
        dwydy = self.sol_linear.dwydy + sol_NL.dwydy

        _gx = -wx * dwxdx - wy * dwxdy
        _gy = -wx * dwydx - wy * dwydy

        res_gx = _gx - gx
        res_gy = _gy - gy

        return [res_gx, res_gy]

    def callback(self, x):
        gx, gy = x
        self.pressure_fields.append(self.pressuresolver.solve(gx, gy).p)

    def post_process(self, result: FixedPointIterationResult, dP: float):
        gx, gy = result.x
        if len(self.pressure_fields) == 0:
            self.pressure_fields.append(np.zeros(self.geom.shape))

        if self.accumulate:
            combined_field = np.min(self.pressure_fields, axis=0)
        else:
            combined_field = self.pressure_fields[-1]
        out = PressureSolution(
            self.geom.Lx,
            self.geom.Ly,
            self.geom.dx,
            self.geom.dy,
            self.geom.xmesh,
            self.geom.ymesh,
            combined_field,
            gx,
            gy,
            result.converged,
        )

        return out


@adaptivefixedpointiteration(max_iter=3, tolerance=0.00001, relaxations=[0, 0.1, 0.2])
class NonLinearPoissonCenterline(NonLinearPoisson):
    def post_process(self, result: FixedPointIterationResult, dP: float):
        sol = super().post_process(result, dP)

        x_ = sol.xmesh[:, 0]
        y_ = sol.ymesh[0, :]
        # Create interpolator. Convert radius-based calculations to diameter-based
        interpolator = RegularGridInterpolator(
            [x_ / 2, y_], sol.p, bounds_error=False, fill_value=0
        )

        out = interpolator((x_, 0), method="linear")
        return x_, out
