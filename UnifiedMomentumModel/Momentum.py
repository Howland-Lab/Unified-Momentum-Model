from abc import ABCMeta
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .Pressure import PressureTable
from .Utilities.FixedPointIteration import fixedpointiteration, adaptivefixedpointiteration


@dataclass
class MomentumSolution:
    """Stores the results of the Unified Momentum model solution."""

    Ctprime: float
    yaw: float
    an: Union[float, npt.ArrayLike]
    u4: Union[float, npt.ArrayLike]
    v4: Union[float, npt.ArrayLike]
    x0: Union[float, npt.ArrayLike]
    dp: Union[float, npt.ArrayLike]
    dp_NL: Optional[Union[float, npt.ArrayLike]] = 0.0
    niter: Optional[int] = 1
    converged: Optional[bool] = True
    beta: Optional[float] = 0.0

    @property
    def Ct(self):
        """Returns the thrust coefficient Ct."""
        return self.Ctprime * (1 - self.an) ** 2 * np.cos(self.yaw) ** 2

    @property
    def Cp(self):
        """Returns the power coefficient Cp."""
        return self.Ctprime * ((1 - self.an) * np.cos(self.yaw)) ** 3


class MomentumBase(metaclass=ABCMeta):
    pass


class LimitedHeck(MomentumBase):
    """
    Solves the limiting case when v_4 << u_4. (Eq. 2.19, 2.20). Also takes Numpy
    array arguments.
    """

    def __call__(self, Ctprime: float, yaw: float, **kwargs) -> MomentumSolution:
        """
        Args:
            Ctprime (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).

        Returns:
            Tuple[float, float, float]: induction and outlet velocities.
        """

        a = Ctprime * np.cos(yaw) ** 2 / (4 + Ctprime * np.cos(yaw) ** 2)
        u4 = (4 - Ctprime * np.cos(yaw) ** 2) / (4 + Ctprime * np.cos(yaw) ** 2)
        v4 = (
            -(4 * Ctprime * np.sin(yaw) * np.cos(yaw) ** 2)
            / (4 + Ctprime * np.cos(yaw) ** 2) ** 2
        )
        dp = np.zeros_like(a)
        x0 = np.inf * np.ones_like(a)
        return MomentumSolution(Ctprime, yaw, a, u4, v4, x0, dp)


@fixedpointiteration(max_iter=500, tolerance=0.00001, relaxation=0.1)
class Heck(MomentumBase):
    """
    Solves the iterative momentum equation for an actuator disk model.
    """

    def __init__(self, v4_correction: float = 1.0):
        """
        Initialize the HeckModel instance.

        Args:
            v4_correction (float, optional): The premultiplier of v4 in the Heck
            model. A correction factor applied to v4, with a default value of
            1.0, indicating no correction. Lu (2023) suggests an empirical correction
            of 1.5.

        Example:
            >>> model = HeckModel(v4_correction=1.5)
        """
        self.v4_correction = v4_correction

    def initial_guess(self, Ctprime, yaw):
        sol = LimitedHeck()(Ctprime, yaw)
        return sol.an, sol.u4, sol.v4

    def residual(self, x: np.ndarray, Ctprime: float, yaw: float) -> np.ndarray:
        """
        Residual function of yawed-actuator disk model in Eq. 2.15.

        Args:
            x (np.ndarray): (a, u4, v4)
            Ctprime (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).

        Returns:
            np.ndarray: residuals of induction and outlet velocities.
        """

        a, u4, v4 = x
        e_a = 1 - np.sqrt(1 - u4**2 - v4**2) / (np.sqrt(Ctprime) * np.cos(yaw)) - a

        e_u4 = (1 - 0.5 * Ctprime * (1 - a) * np.cos(yaw) ** 2) - u4

        e_v4 = (
            -self.v4_correction
            * 0.25
            * Ctprime
            * (1 - a) ** 2
            * np.sin(yaw)
            * np.cos(yaw) ** 2
            - v4
        )

        return np.array([e_a, e_u4, e_v4])

    def post_process(self, result, Ctprime: float, yaw: float):
        if result.converged:
            a, u4, v4 = result.x
        else:
            a, u4, v4 = np.nan * np.zeros_like([Ctprime, Ctprime, Ctprime])
        dp = np.zeros_like(a)
        x0 = np.inf * np.ones_like(a)
        return MomentumSolution(
            Ctprime,
            yaw,
            a,
            u4,
            v4,
            x0,
            dp,
            niter=result.niter,
            converged=result.converged,
        )


@fixedpointiteration(max_iter=500, relaxation=0.25, tolerance=0.00001)
class UnifiedMomentum(MomentumBase):
    def __init__(self, beta=0.1403, cached=True, v4_correction=1., **kwargs):
        self.beta = beta
        self.v4_correction = v4_correction

        if cached and PressureTable.CACHE_FN.exists():
            # load cache
            self.nonlinear_interpolator = PressureTable.load_cache()
        else:
            # otherwise, generate and save
            dps, xs, ps = PressureTable.generate_pressure_table(**kwargs)
            if cached:
                PressureTable.save_cache(dps, xs, ps)
            self.nonlinear_interpolator = PressureTable.make_interpolator(dps, xs, ps)

    def initial_guess(self, Ctprime, yaw):
        """Returns the initial guess for the solution variables."""
        sol = LimitedHeck()(Ctprime, yaw)

        x0 = 1000 * np.ones_like(Ctprime)
        dp = np.zeros_like(Ctprime)

        return np.vstack([sol.an, sol.u4, sol.v4, x0, dp])

    def residual(self, x: np.ndarray, Ctprime: float, yaw: float) -> Tuple[float, ...]:
        """
        Returns the residuals of the Unified Momentum Model for the fixed point
        iteration. The equations referred to in this function are from the
        associated paper.
        """
        an, u4, v4, x0, dp = x
        if type(Ctprime) is float and Ctprime == 0:
            return 0 - an, 1 - u4, 0 - v4, 100 - x0, 0 - dp

        p_g = self._nonlinear_pressure(Ctprime, yaw, an, x0)

        # Eq. 4 - Near wake length in residual form.
        e_x0 = (
            np.cos(yaw)
            / (2 * self.beta)
            * (1 + u4)
            / np.abs(1 - u4)
            * np.sqrt((1 - an) * np.cos(yaw) / (1 + u4))
        ) - x0

        # Eq. 1 - Rotor-normal induction in residual form.
        e_an = (
            1
            - np.sqrt(
                -dp / (0.5 * Ctprime * np.cos(yaw) ** 2)
                + (1 - u4**2 - v4**2) / (Ctprime * np.cos(yaw) ** 2)
            )
        ) - an

        # Eq. 2 - Streamwise outlet velocity in residual form.
        e_u4 = (
            -(1 / 4) * Ctprime * (1 - an) * np.cos(yaw) ** 2
            + (1 / 2)
            + (1 / 2)
            * np.sqrt(
                (1 / 2 * Ctprime * (1 - an) * np.cos(yaw) ** 2 - 1) ** 2 - (4 * dp)
            )
        ) - u4

        # Eq. 3 - Lateral outlet velocity in residual form.
        e_v4 = -self.v4_correction * (1 / 4) * Ctprime * (1 - an) ** 2 * np.sin(yaw) * np.cos(yaw) ** 2 - v4

        # Eq. 5 - Outlet pressure drop in residual form.
        e_dp = (
            (
                -(1 / (2 * np.pi))
                * Ctprime
                * (1 - an) ** 2
                * np.cos(yaw) ** 2
                * np.arctan(1 / (2 * x0))
            )
            + p_g
        ) - dp

        return e_an, e_u4, e_v4, e_x0, e_dp

    def _nonlinear_pressure(self, Ctprime, yaw, an, x0):
        CT = Ctprime * (1 - an) ** 2 * np.cos(yaw) ** 2

        p_g = self.nonlinear_interpolator((CT / 2, x0))
        return p_g

    def post_process(self, result, Ctprime, yaw):
        a, u4, v4, x0, dp = result.x

        p_g = self._nonlinear_pressure(Ctprime, yaw, a, x0)
        return MomentumSolution(
            Ctprime,
            yaw,
            a,
            u4,
            v4,
            x0,
            dp,
            dp_NL=p_g,
            niter=result.niter,
            converged=result.converged,
            beta=self.beta,
        )


@adaptivefixedpointiteration(max_iter=10000, relaxations=[0.4, 0.6], tolerance=0.00001)
class ThrustBasedUnified(UnifiedMomentum):
    def __init__(self, beta=0.1403, cached=True):
        super().__init__(beta=beta, cached=cached)

    def initial_guess(self, Ct, yaw):
        an = 0.5 * Ct
        u4 = 1 - Ct
        v4 = np.zeros_like(Ct)
        dp = np.zeros_like(Ct)
        x0 = 100 * np.ones_like(Ct)
        Ctprime = np.sign(Ct)

        return np.vstack([an, u4, v4, x0, dp, Ctprime])

    def residual(self, x, Ct, yaw):
        an, u4, v4, x0, dp, Ctprime = x

        e_an, e_u4, e_v4, e_x0, e_dp = super().residual(
            [an, u4, v4, x0, dp], Ctprime, yaw
        )

        # Eq. 6 - thrust coefficient equation in residual form.
        e_Ctprime = Ct / ((1 - an) ** 2 * np.cos(yaw) ** 2) - Ctprime
        return np.array([e_an, e_u4, e_v4, e_x0, e_dp, e_Ctprime])

    def post_process(self, result, Ct, yaw):
        a, u4, v4, x0, dp, Ctprime = result.x
        p_g = self._nonlinear_pressure(Ctprime, yaw, a, x0)
        return MomentumSolution(
            Ctprime,
            yaw,
            a,
            u4,
            v4,
            x0,
            dp,
            dp_NL=p_g,
            niter=result.niter,
            converged=result.converged,
            beta=self.beta,
        )
