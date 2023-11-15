import numpy as np
from numpy.typing import ArrayLike
from scipy import signal, sparse, interpolate
from tqdm import trange
from abc import ABC

from ..Utilities import Calculus, Geometry


class PressureSolution:
    def __init__(
        self,
        Lx: float,
        Ly: float,
        dx: float,
        dy: float,
        xmesh: ArrayLike,
        ymesh: ArrayLike,
        p: ArrayLike,
        fx: ArrayLike,
        fy: ArrayLike,
        converged: bool,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.dy = dy
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.p = p.copy()
        self.fx = fx.copy()
        self.fy = fy.copy()
        self.converged = converged

    @property
    def dpdx(self) -> ArrayLike:
        return Calculus.derivative_x(self.p, self.dx)

    @property
    def dpdy(self) -> ArrayLike:
        return Calculus.derivative_y(self.p, self.dy)

    @property
    def d2pdx2(self) -> ArrayLike:
        return Calculus.derivative_x(self.dpdx, self.dx)

    @property
    def d2pdy2(self) -> ArrayLike:
        return Calculus.derivative_y(self.dpdy, self.dy)

    @property
    def dwxdx(self) -> ArrayLike:
        return -self.dpdx + self.fx

    @property
    def wx(self) -> ArrayLike:
        return -self.p + Calculus.integrate_x(self.fx, dx=self.dx)

    @property
    def dwxdy(self) -> ArrayLike:
        return -self.dpdy + Calculus.integrate_x(
            Calculus.derivative_y(self.fx, self.dy), self.dx
        )

    @property
    def dwydx(self) -> ArrayLike:
        return -self.dpdy + self.fy

    @property
    def wy(self) -> ArrayLike:
        return Calculus.integrate_x(self.dwydx, dx=self.dx)

    @property
    def dwydy(self) -> ArrayLike:
        return Calculus.derivative_y(self.wy, self.dy)

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


class PressureSolver(ABC):
    def __init__(self, geometry: Geometry.Geometry, *args, **kwargs):
        self.Lx = geometry.Lx
        self.Ly = geometry.Ly
        self.dx = geometry.dx
        self.dy = geometry.dy

        self.shape = geometry.shape
        self.Nx, self.Ny = geometry.Nx, geometry.Ny

        self.x, self.y = geometry.x, geometry.y
        self.xmesh, self.ymesh = geometry.xmesh, geometry.ymesh

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
        div_f = Calculus.derivative_x(fx, self.dx) + Calculus.derivative_y(fy, self.dy)

        p = np.zeros(self.shape)
        converged = False
        for i in trange(max_iter, disable=False if progress is True else True):
            res_p = explicit_pressure_residual(p, div_f, self.dx, self.dy)
            p += (1 - relax) * res_p
            if np.max(np.abs(res_p)) < eps:
                converged = True
                break

        return PressureSolution(
            self.Lx,
            self.Ly,
            self.dx,
            self.dy,
            self.xmesh,
            self.ymesh,
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
            self.x, self.y, self.x, self.y, indexing="ij"
        )

        print(xmesh.shape)


class MatrixInversion(PressureSolver):
    """
    Solves the Poisson-pressure equation using a sparse matrix inversion
    """

    def solve(self, fx: ArrayLike, fy: ArrayLike) -> PressureSolution:
        div_f = Calculus.derivative_x(fx, self.dx) + Calculus.derivative_y(fy, self.dy)

        _Nx = self.Nx - 2
        _Ny = self.Ny - 2
        N = _Nx * _Ny

        diags = [
            -self.dy**2 * np.ones(N - _Ny),
            -self.dx**2 * np.ones(N - 1),
            2 * (self.dx**2 + self.dy**2) * np.ones(N),
            -self.dx**2 * np.ones(N - 1),
            -self.dy**2 * np.ones(N - _Ny),
        ]

        offsets = [-_Ny, -1, 0, 1, _Ny]
        A = sparse.diags(diags, offsets, format="csc")

        B = (-(self.dx**2 * self.dy**2) * div_f[1:-1, 1:-1]).ravel()
        p = sparse.linalg.spsolve(A, B)
        p = p.reshape(_Nx, _Ny)

        p_out = np.zeros(self.shape)
        p_out[1:-1, 1:-1] = p

        return PressureSolution(
            self.Lx,
            self.Ly,
            self.dx,
            self.dy,
            self.xmesh,
            self.ymesh,
            p_out,
            fx,
            fy,
            True,
        )


class GreensFunctionIntegration(PressureSolver):
    """
    Solves the Poisson-pressure equation using a 2d integral over 2d space. This
    just doesn't work as it requires a 4d array which is too large to allocate.
    Needs a smarter method.
    """

    def solve(self, fx: ArrayLike, fy: ArrayLike) -> PressureSolution:
        xmesh, ymesh, ximesh, numesh = np.meshgrid(
            self.x, self.y, self.x, self.y, indexing="ij"
        )

        print(xmesh.shape)


class Convolution(PressureSolver):
    """
    Solves the Poisson-pressure equation by solving the Green's function
    solution using 2D convolutions.
    """

    def __init__(self, geometry: Geometry.Geometry):
        super().__init__(geometry)

        self.x_kernel, self.y_kernel = self.construct_kernel(
            self.dx, self.dy, self.Lx, self.Ly
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
            self.Lx,
            self.Ly,
            self.dx,
            self.dy,
            self.xmesh,
            self.ymesh,
            p_out,
            fx,
            fy,
            True,
        )
