from abc import ABCMeta
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from .Pressure import PressureTable
from .Utilities.FixedPointIteration import fixedpointiteration, adaptivefixedpointiteration
from .Utilities.Geometry import calc_eff_yaw, eff_yaw_inv_rotation

@dataclass
class MomentumSolution:
    """Stores the results of a momentum model solution."""
    Ctprime: Union[float, npt.ArrayLike]
    yaw: Union[float, npt.ArrayLike]
    an: Union[float, npt.ArrayLike]
    u4: Union[float, npt.ArrayLike]
    v4: Union[float, npt.ArrayLike]
    x0: Union[float, npt.ArrayLike]
    dp: Union[float, npt.ArrayLike]
    # optional keyword paramters
    tilt: Union[float, npt.ArrayLike] = 0.0
    w4: Union[float, npt.ArrayLike] = 0.0
    dp_NL: Optional[Union[float, npt.ArrayLike]] = 0.0
    niter: Optional[int] = 1
    converged: Optional[bool] = True
    beta: Optional[float] = 0.0

    @property
    def Ct(self):
        """Returns the thrust coefficient Ct."""
        eff_yaw = calc_eff_yaw(self.yaw, self.tilt)
        return self.Ctprime * (1 - self.an) ** 2 * np.cos(eff_yaw) ** 2

    @property
    def Cp(self):
        """Returns the power coefficient Cp."""
        eff_yaw = calc_eff_yaw(self.yaw, self.tilt)
        return self.Ctprime * ((1 - self.an) * np.cos(eff_yaw)) ** 3


class MomentumBase(metaclass=ABCMeta):
    pass


class LimitedHeck(MomentumBase):
    """
    Solves the limiting case of the Heck momentum model when v_4 << u_4. See Heck et al 2023: Eqs. 2.19 - 2.20.

    __init__:
        - Args: None
        - Returns: LimitedHeck object
        - Example:
            >>> model = LimitedHeck()

    __call__:
        - Args:
            - Ctprime (float or npt.ArrayLike): Local rotor thrust coefficient.
            - yaw (float or npt.ArrayLike): Rotor yaw angle (radians). Postitive yaw positive is a CCW rotation viewed from above & v4 < 0.
            - tilt (float or npt.ArrayLike): Rotor tilt angle(radians). Positive tilt is an upward facing rotor & w4 > 0.
        - Returns: MomentumSolution calculated by LimitedHeck.
        - Example:
            >>> solution = model(1, yaw = 0, tilt = 0)
    """

    def __call__(self, Ctprime: float, yaw: float = 0, tilt: float = 0, **kwargs) -> MomentumSolution:
        """
        Solves the limiting case of the Heck momentum model when v_4 << u_4.
        See above class documentation on __call__ for more details.
        """
        eff_yaw = calc_eff_yaw(yaw, tilt)
        a = Ctprime * np.cos(eff_yaw) ** 2 / (4 + Ctprime * np.cos(eff_yaw) ** 2)
        u4 = (4 - Ctprime * np.cos(eff_yaw) ** 2) / (4 + Ctprime * np.cos(eff_yaw) ** 2)
        v4 = (
            -(4 * Ctprime * np.sin(eff_yaw) * np.cos(eff_yaw) ** 2)
            / (4 + Ctprime * np.cos(eff_yaw) ** 2) ** 2
        )
        w4 = np.zeros_like(v4)
        dp = np.zeros_like(a)
        x0 = np.inf * np.ones_like(a)
        u4, v4, w4 = eff_yaw_inv_rotation(u4, v4, w4, eff_yaw, yaw, tilt)
        return MomentumSolution(Ctprime, yaw, a, u4, v4, x0, dp, tilt = tilt, w4 = w4)


@fixedpointiteration(max_iter=500, tolerance=0.00001, relaxation=0.1)
class Heck(MomentumBase):
    """
    Solves the Heck momentum model for an actuator disk. See Heck et al, 2023. Uses an iterative solver.

    __init__:
        - Args:
            - v4_correction (float, optional): The premultiplier of v4 in the Heck model.
                A correction factor applied to v4, with a default value of 1.0, indicating no correction.
                Lu (2023) suggests an empirical correction of 1.5.
        - Returns: Heck object
        - Example:
            >>> model = Heck(v4_correction=1.5)

    __call__:
        - Args:
            - Ctprime (float or npt.ArrayLike): Local rotor thrust coefficient.
            - yaw (float or npt.ArrayLike): Rotor yaw angle (radians). Postitive yaw positive is a CCW rotation viewed from above & v4 < 0.
            - tilt (float or npt.ArrayLike): Rotor tilt angle(radians). Positive tilt is an upward facing rotor & w4 > 0.
        - Returns: MomentumSolution calculated by Heck.
        - Example:
            >>> solution = model([0.5, 1.0, 1.5], yaw = 0, tilt = 0)

    child class:
        - Requires any new setpoints to be keyword arguments to work with current pre_process function.
        - User can define a new pre_process class, but is required to define an effective yaw (self.eff_yaw)
            that combines the misalignment due to yaw and tilt into an effective angle.
            See functions calc_eff_yaw and eff_yaw_inv_rotation in Geometry for more information.
    """

    def __init__(self, v4_correction: float = 1.0):
        """
        Initialize the Heck instance.
        See above class documentation on __init__ for more details.
        """
        self.v4_correction = v4_correction

    def pre_process(self, Ctprime, yaw = 0, tilt = 0, **kwargs):
        # switch reference frame to a "yaw-only" frame where y' is aligned with the lateral wake
        self.eff_yaw = calc_eff_yaw(yaw, tilt)
        return

    def initial_guess(self, Ctprime, *args, **kwargs):
        sol = LimitedHeck()(Ctprime, self.eff_yaw)
        return sol.an, sol.u4, sol.v4

    def residual(self, x: np.ndarray, Ctprime: float, *args: float, **kwargs: float) -> np.ndarray:
        """
        Residual function of yawed-actuator disk model in Heck et al, 2023. See Eq. 2.15.

        Args:
            x (np.ndarray): (a, u4, v4)
            Ctprime (float): Rotor thrust coefficient.

        Returns:
            np.ndarray: residuals of induction and outlet velocities.
        """
        a, u4, v4 = x
        e_a = 1 - np.sqrt(1 - u4**2 - v4**2) / (np.sqrt(Ctprime) * np.cos(self.eff_yaw)) - a

        e_u4 = (1 - 0.5 * Ctprime * (1 - a) * np.cos(self.eff_yaw) ** 2) - u4

        e_v4 = (
            -self.v4_correction
            * 0.25
            * Ctprime
            * (1 - a) ** 2
            * np.sin(self.eff_yaw)
            * np.cos(self.eff_yaw) ** 2
            - v4
        )

        return np.array([e_a, e_u4, e_v4])

    def post_process(self, result, Ctprime: float, yaw: float = 0, tilt: float = 0, **kwargs):
        if result.converged:
            a, u4, v4 = result.x
            w4 = np.zeros_like(v4)
            # rotate back into ground frame from "yaw-only" frame
            u4, v4, w4 = eff_yaw_inv_rotation(u4, v4, w4, self.eff_yaw, yaw, tilt)
        else:
            a, u4, v4, w4 = np.nan * np.zeros_like([Ctprime, Ctprime, Ctprime, Ctprime])
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
            tilt = tilt,
            w4 = w4,
            niter=result.niter,
            converged=result.converged,
        )


@fixedpointiteration(max_iter=500, relaxation=0.25, tolerance=0.00001)
class UnifiedMomentum(MomentumBase):
    """
    Solves the UnifiedMomentum momentum model for an actuator disk. See Liew et al, 2024. Uses an iterative solver.

    __init__:
        - Args:
            - beta (float, optional)
            - cached (boolean, optional)
            - v4_correction (float, optional): The premultiplier of v4 in the Heck model.
                A correction factor applied to v4, with a default value of 1.0, indicating no correction.
                Lu (2023) suggests an empirical correction of 1.5.
        - Returns: UnifiedMomentum object
        - Example:
            >>> model = UnifiedMomentum(v4_correction=1.5)

    __call__:
        - Args:
            - Ctprime (float or npt.ArrayLike): Local rotor thrust coefficient.
            - yaw (float or npt.ArrayLike): Rotor yaw angle (radians). Postitive yaw positive is a CCW rotation viewed from above & v4 < 0.
            - tilt (float or npt.ArrayLike): Rotor tilt angle(radians). Positive tilt is an upward facing rotor & w4 > 0.
        - Returns: MomentumSolution calculated by UnifiedMomentum.
        - Example:
            >>> solution = model([0.5, 1.0, 1.5], yaw = 0, tilt = 0)

    child class:
        - Requires any new setpoints to be keyword arguments to work with current pre_process function.
        - User can define a new pre_process class, but is required to define an effective yaw (self.eff_yaw)
            that combines the misalignment due to yaw and tilt into an effective angle.
            See functions calc_eff_yaw and eff_yaw_inv_rotation in Geometry for more information.
    """
    def __init__(self, beta=0.1403, cached=True, v4_correction=1.0, **kwargs):
        """
        Initialize the UnifiedMomentum instance.
        See above class documentation on __init__ for more details.
        """
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

    def pre_process(self, Ctprime, yaw = 0, tilt = 0, **kwargs):
        # switch reference frame to a "yaw-only" frame where y' is aligned with the lateral wake
        self.eff_yaw = calc_eff_yaw(yaw, tilt)
        return

    def initial_guess(self, Ctprime, *args, **kwargs):
        """Returns the initial guess for the solution variables."""
        sol = LimitedHeck()(Ctprime, self.eff_yaw)
        x0 = 1000 * np.ones_like(Ctprime)
        dp = np.zeros_like(Ctprime)

        return sol.an, sol.u4, sol.v4, x0, dp

    def residual(self, x: np.ndarray, Ctprime: float, *args: float, e_x0 = None, **kwargs: float) -> Tuple[float, ...]:
        """
        Returns the residuals of the Unified Momentum Model for the fixed point
        iteration. The equations referred to in this function are from the
        associated paper Liew et al 2024.
        """
        an, u4, v4, x0, dp = x
        if type(Ctprime) is float and Ctprime == 0:
            return 0 - an, 1 - u4, 0 - v4, 100 - x0, 0 - dp

        p_g = self._nonlinear_pressure(Ctprime, self.eff_yaw, an, x0)

        # Eq. 4 - Near wake length in residual form.
        if e_x0 is None:
            e_x0 = (
                np.cos(self.eff_yaw)
                / (2 * self.beta)
                * (1 + u4)
                / np.abs(1 - u4)
                * np.sqrt((1 - an) * np.cos(self.eff_yaw) / (1 + u4))
            ) - x0

        # Eq. 1 - Rotor-normal induction in residual form.
        e_an = (
            1
            - np.sqrt(
                -dp / (0.5 * Ctprime * np.cos(self.eff_yaw) ** 2)
                + (1 - u4**2 - v4**2) / (Ctprime * np.cos(self.eff_yaw) ** 2)
            )
        ) - an

        # Eq. 2 - Streamwise outlet velocity in residual form.
        e_u4 = (
            -(1 / 4) * Ctprime * (1 - an) * np.cos(self.eff_yaw) ** 2
            + (1 / 2)
            + (1 / 2)
            * np.sqrt(
                (1 / 2 * Ctprime * (1 - an) * np.cos(self.eff_yaw) ** 2 - 1) ** 2 - (4 * dp)
            )
        ) - u4

        # Eq. 3 - Lateral outlet velocity in residual form.
        e_v4 = (
            -self.v4_correction
            * (1 / 4)
            * Ctprime
            * (1 - an) ** 2
            * np.sin(self.eff_yaw)
            * np.cos(self.eff_yaw) ** 2
            - v4
        )

        # Eq. 5 - Outlet pressure drop in residual form.
        e_dp = (
            (
                -(1 / (2 * np.pi))
                * Ctprime
                * (1 - an) ** 2
                * np.cos(self.eff_yaw) ** 2
                * np.arctan(1 / (2 * x0))
            )
            + p_g
        ) - dp

        return e_an, e_u4, e_v4, e_x0, e_dp

    def _nonlinear_pressure(self, Ctprime, eff_yaw, an, x0):
        CT = Ctprime * (1 - an) ** 2 * np.cos(eff_yaw) ** 2
        p_g = self.nonlinear_interpolator((CT / 2, x0))
        return p_g

    def post_process(self, result, Ctprime, yaw = 0, tilt = 0, **kwargs):
        a, u4, v4, x0, dp = result.x
        w4 = np.zeros_like(v4)
        p_g = self._nonlinear_pressure(Ctprime, self.eff_yaw, a, x0)
        # rotate back into ground frame from "yaw-only" frame
        u4, v4, w4 = eff_yaw_inv_rotation(u4, v4, w4, self.eff_yaw, yaw, tilt)
        return MomentumSolution(
            Ctprime,
            yaw,
            a,
            u4,
            v4,
            x0,
            dp,
            dp_NL=p_g,
            tilt = tilt,
            w4 = w4,
            niter=result.niter,
            converged=result.converged,
            beta=self.beta,
        )


@adaptivefixedpointiteration(max_iter=10000, relaxations=[0.4, 0.6], tolerance=0.00001)
class ThrustBasedUnified(UnifiedMomentum):
    """
    Solves the ThrustBasedUnified momentum model for an actuator disk. See Liew et al, 2024. Uses an iterative solver.
    Has one extra equation compared to the UnifiedMomentum solver that allows CT as an input rather than CT'.

    __init__:
        - Args:
            - beta (float, optional)
            - cached (boolean, optional)
        - Returns: UnifiedMomentum object
        - Example:
            >>> model = ThrustBasedUnified()

    __call__:
        - Args:
            - Ct (float or npt.ArrayLike): Global rotor thrust coefficient.
            - yaw (float or npt.ArrayLike): Rotor yaw angle (radians). Postitive yaw positive is a CCW rotation viewed from above & v4 < 0.
            - tilt (float or npt.ArrayLike): Rotor tilt angle(radians). Positive tilt is an upward facing rotor & w4 > 0.
        - Returns: MomentumSolution calculated by ThrustBasedUnified.
        - Example:
            >>> solution = model([0.5, 1.0, 1.5], yaw = 0, tilt = 0)

    child class:
        - Requires any new setpoints to be keyword arguments to work with current pre_process function.
        - User can define a new pre_process class, but is required to define an effective yaw (self.eff_yaw)
            that combines the misalignment due to yaw and tilt into an effective angle.
            See functions calc_eff_yaw and eff_yaw_inv_rotation in Geometry for more information.
    """
    def __init__(self, beta=0.1403, cached=True):
        super().__init__(beta=beta, cached=cached)

    def pre_process(self, *args, **kwargs):
        super().pre_process(*args, **kwargs)
        return

    def initial_guess(self, Ct, *args, **kwargs):
        an = 0.5 * Ct
        u4 = 1 - Ct
        v4 = np.zeros_like(Ct)
        dp = np.zeros_like(Ct)
        x0 = 100 * np.ones_like(Ct)
        Ctprime = np.sign(Ct)

        return an, u4, v4, x0, dp, Ctprime

    def residual(self, x, Ct, *args, **kwargs):
        an, u4, v4, x0, dp, Ctprime = x

        e_an, e_u4, e_v4, e_x0, e_dp = super().residual(
            [an, u4, v4, x0, dp], Ctprime, self.eff_yaw
        )

        # Eq. 6 - thrust coefficient equation in residual form.
        e_Ctprime = Ct / ((1 - an) ** 2 * np.cos(self.eff_yaw) ** 2) - Ctprime
        return np.array([e_an, e_u4, e_v4, e_x0, e_dp, e_Ctprime])

    def post_process(self, result, Ct, yaw = 0, tilt = 0, **kwargs):
        a, u4, v4, x0, dp, Ctprime = result.x
        w4 = np.zeros_like(v4)
        p_g = self._nonlinear_pressure(Ctprime, self.eff_yaw, a, x0)
        # rotate back into ground frame from "yaw-only" frame
        u4, v4, w4 = eff_yaw_inv_rotation(u4, v4, w4, self.eff_yaw, yaw, tilt)
        return MomentumSolution(
            Ctprime,
            yaw,
            a,
            u4,
            v4,
            x0,
            dp,
            dp_NL=p_g,
            tilt = tilt,
            w4 = w4,
            niter=result.niter,
            converged=result.converged,
            beta=self.beta,
        )
