from abc import ABC

import numpy as np
from numpy.typing import ArrayLike
from scipy import interpolate, signal, sparse
from tqdm import trange

from ..Utilities import Calculus
from ..Utilities.Geometry import Geometry


class PressureSolution:
    def __init__(
        self,
        geometry: Geometry,
        p: ArrayLike,
        fx: ArrayLike,
        fy: ArrayLike,
        converged: bool,
    ):
        self.geom = geometry
        self.p = p.copy()
        self.fx = fx.copy()
        self.fy = fy.copy()
        self.converged = converged

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
        return -self.dpdx + self.fx

    @property
    def wx(self) -> ArrayLike:
        return -self.p + Calculus.integrate_x(self.fx, dx=self.geom.dx)

    @property
    def dwxdy(self) -> ArrayLike:
        return -self.dpdy + Calculus.integrate_x(
            Calculus.derivative_y(self.fx, self.geom.dy), self.geom.dx
        )

    @property
    def dwydx(self) -> ArrayLike:
        return -self.dpdy + self.fy

    @property
    def wy(self) -> ArrayLike:
        return Calculus.integrate_x(self.dwydx, dx=self.geom.dx)

    @property
    def dwydy(self) -> ArrayLike:
        return Calculus.derivative_y(self.wy, self.geom.dy)

    def centerline_pressure(self, x):
        """Returns the centerline pressure at given x locations"""
        x_ = self.geom.xmesh[:, 0]
        y_ = self.geom.ymesh[0, :]

        interpolator = interpolate.RegularGridInterpolator(
            [x_, y_], self.p, bounds_error=False, fill_value=0
        )

        # TO DO: upgrade to scipy 1.10.0+ to higher-order interpolation methods
        out = interpolator((x, 0), method="linear")

        return out


class PressureSolver(ABC):
    def __init__(self, geometry: Geometry, *args, **kwargs):
        self.geom = geometry

    def solve(self, fx: ArrayLike, fy: ArrayLike) -> PressureSolution:
        ...


def explicit_pressure_residual(p, div_f, dx, dy):
    eps = np.zeros_like(p)

    eps[1:-1, 1:-1] = (
        0.5
        / (dx**2 + dy**2)
        * (
            -(dx**2) * dy**2 * div_f[1:-1, 1:-1]
            + dy**2 * (p[2:, 1:-1] + p[:-2, 1:-1])
            + dx**2 * (p[1:-1, 2:] + p[1:-1, :-2])
        )
        - p[1:-1, 1:-1]
    )
    return eps


class Iterative(PressureSolver):
    def solve(
        self,
        fx: ArrayLike,
        fy: ArrayLike,
        max_iter=10000,
        relax=0.0,
        eps=0.00001,
        progress=True,
    ) -> (ArrayLike, bool):
        div_f = Calculus.derivative_x(fx, self.geom.dx) + Calculus.derivative_y(
            fy, self.geom.dy
        )

        p = np.zeros(self.geom.shape)
        converged = False
        for i in trange(max_iter, disable=False if progress is True else True):
            res_p = explicit_pressure_residual(p, div_f, self.geom.dx, self.geom.dy)
            p += (1 - relax) * res_p
            if np.max(np.abs(res_p)) < eps:
                converged = True
                break

        return PressureSolution(
            self.geom,
            p,
            fx,
            fy,
            converged,
        )


class GreensFunctionIntegration(PressureSolver):
    """
    Solves the Poisson-pressure equation using a 2d integral over 2d space. This
    just doesn't work as it requires a 4d array which is too large to allocate.
    Needs a smarter method.
    """

    def solve(self, fx: ArrayLike, fy: ArrayLike) -> PressureSolution:
        xmesh, ymesh, ximesh, numesh = np.meshgrid(
            self.geom.x, self.geom.y, self.geom.x, self.geom.y, indexing="ij"
        )

        print(xmesh.shape)


class MatrixInversion(PressureSolver):
    """
    Solves the Poisson-pressure equation using a sparse matrix inversion
    """

    def solve(self, fx: ArrayLike, fy: ArrayLike) -> PressureSolution:
        div_f = Calculus.derivative_x(fx, self.geom.dx) + Calculus.derivative_y(
            fy, self.geom.dy
        )

        _Nx = self.geom.Nx - 2
        _Ny = self.geom.Ny - 2
        N = _Nx * _Ny

        diags = [
            -self.geom.dy**2 * np.ones(N - _Ny),
            -self.geom.dx**2 * np.ones(N - 1),
            2 * (self.geom.dx**2 + self.geom.dy**2) * np.ones(N),
            -self.geom.dx**2 * np.ones(N - 1),
            -self.geom.dy**2 * np.ones(N - _Ny),
        ]

        offsets = [-_Ny, -1, 0, 1, _Ny]
        A = sparse.diags(diags, offsets, format="csc")

        B = (-(self.geom.dx**2 * self.geom.dy**2) * div_f[1:-1, 1:-1]).ravel()
        p = sparse.linalg.spsolve(A, B)
        p = p.reshape(_Nx, _Ny)

        p_out = np.zeros(self.shape)
        p_out[1:-1, 1:-1] = p

        return PressureSolution(
            self.geom,
            p_out,
            fx,
            fy,
            True,
        )


class Convolution(PressureSolver):
    """
    Solves the Poisson-pressure equation by solving the Green's function
    solution using 2D convolutions.
    """

    def __init__(self, geometry: Geometry):
        super().__init__(geometry)

        self.x_kernel, self.y_kernel = self.construct_kernel(
            self.geom.dx, self.geom.dy, self.geom.Lx, self.geom.Ly
        )

    # @staticmethod
    def construct_kernel(
        self, dx: float, dy: float, kernel_width: float, kernel_height: float
    ):
        _Nx = kernel_width / dx
        _x = dx * np.arange(0, _Nx)
        x = np.concatenate([-np.flip(_x)[:-1], _x])

        _Ny = kernel_height / dy
        _y = dy * np.arange(0, _Ny)
        y = np.concatenate([-np.flip(_y)[:-1], _y])

        xmesh, ymesh = np.meshgrid(x, y, indexing="ij")

        with np.errstate(invalid="ignore"):
            x_kernel = xmesh / (xmesh**2 + ymesh**2)
            y_kernel = ymesh / (xmesh**2 + ymesh**2)

        x_kernel[np.isnan(x_kernel)] = 0.0
        y_kernel[np.isnan(y_kernel)] = 0.0

        return x_kernel * dx * dy, y_kernel * dx * dy

    def solve(
        self,
        fx: ArrayLike,
        fy: ArrayLike,
    ) -> PressureSolution:
        p_out = (
            1
            / (2 * np.pi)
            * (
                signal.fftconvolve(fx, self.x_kernel, mode="same")
                + signal.fftconvolve(fy, self.y_kernel, mode="same")
            )
        )

        return PressureSolution(
            self.geom,
            p_out,
            fx,
            fy,
            True,
        )
