"""
Solves the 2D pressure field due to uniform actuator disk forcing, ignoring
advection forcing. Tests and compares the solution of various pressure solvers.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy import interpolate

from ..Utilities import Calculus
from ..Utilities.Geometry import EquidistantRectGridOdd, Geometry


class AnalyticalLinearSolution:
    def __init__(self, dP, geometry: Geometry):
        self.dP = dP
        self.geom = geometry
        self.xmesh = geometry.xmesh
        self.ymesh = geometry.ymesh

        # Analytical solutions to pressure and velocity fields.
        self.p = (
            -1
            / (2 * np.pi)
            * dP
            * (
                np.arctan((1 - self.geom.ymesh) / self.geom.xmesh)
                + np.arctan((1 + self.geom.ymesh) / self.geom.xmesh)
            )
        )

        self.wx = -np.array(self.p)
        self.wx[(self.geom.xmesh > 0) & (np.abs(self.geom.ymesh) <= 1)] -= dP
        self.wy = (
            dP
            / (4 * np.pi)
            * np.log(
                (self.geom.xmesh**2 + (self.geom.ymesh + 1) ** 2)
                / (self.geom.xmesh**2 + (self.geom.ymesh - 1) ** 2)
            )
        )

        # Remove nans and infs
        self.p[np.isnan(self.p)] = 0
        self.wx[np.isnan(self.wx)] = 0
        self.wy[self.wy == -np.inf] = np.nanmin(self.wy[np.isfinite(self.wy)])
        self.wy[self.wy == np.inf] = np.nanmax(self.wy[np.isfinite(self.wy)])

    @property
    def dpdx(self) -> ArrayLike:
        return Calculus.derivative_x(self.p, self.geom.dx)

    @property
    def dpdy(self) -> ArrayLike:
        return Calculus.derivative_y(self.p, self.geom.dy)

    @property
    def d2pdx2(self) -> ArrayLike:
        return Calculus.derivative_x(self.dpdx, self.geom.dx)

    @property
    def d2pdy2(self) -> ArrayLike:
        return Calculus.derivative_y(self.dpdy, self.geom.dy)

    @property
    def dwxdx(self) -> ArrayLike:
        return Calculus.derivative_x(self.wx, self.geom.dx)

    @property
    def dwxdy(self) -> ArrayLike:
        return Calculus.derivative_y(self.wx, self.geom.dy)

    @property
    def dwydx(self) -> ArrayLike:
        return Calculus.derivative_x(self.wy, self.geom.dx)
        return -self.dpdy + self.fy

    @property
    def dwydy(self) -> ArrayLike:
        return Calculus.derivative_y(self.wy, self.geom.dy)

    def centerline_pressure(self, x):
        """Returns the centerline pressure at given x locations"""
        x_ = self.xmesh[:, 0]
        y_ = self.ymesh[0, :]

        interpolator = interpolate.RegularGridInterpolator(
            [x_, y_], self.p, bounds_error=False, fill_value=0
        )

        # TO DO: upgrade to scipy 1.10.0+ to higher-order interpolation methods
        out = interpolator((x, 0), method="linear")

        return out


def construct_f(dP: float, geom: Geometry):
    fx = np.zeros(geom.shape)
    mask = np.abs(geom.y) <= 1.0
    N = mask.sum()
    fx[geom.Nx // 2, mask] = -dP * (geom.dy / geom.dx) / (2 / (N - 1))

    fy = np.zeros(geom.shape)
    return fx, fy
