from abc import ABCMeta
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from scipy.interpolate import LinearNDInterpolator
from pathlib import Path
import functools
import polars as pl
from scipy.optimize import root, fsolve

import numpy as np
import numpy.typing as npt

from .Pressure import PressureTable
from .Utilities.FixedPointIteration import fixedpointiteration, adaptivefixedpointiteration
from .Utilities.Geometry import calc_eff_yaw, eff_yaw_inv_rotation
from .Utilities.Caching import cache_polars

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
    # optional keyword parameters
    tilt: Union[float, npt.ArrayLike] = 0.0
    w4: Union[float, npt.ArrayLike] = 0.0
    dp_NL: Optional[Union[float, npt.ArrayLike]] = 0.0
    niter: Optional[int] = 1
    converged: Optional[bool] = True
    beta_s: Optional[float] = 0.0 # shear layer growth parameter for Unified Momentum Model

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
    
@dataclass
class BlockageSolution(MomentumSolution):
    dp: Union[float, npt.ArrayLike] = 0.0 # dp = (p1 - p4) / (rho * Uinf^2)
    dpw: Union[float, npt.ArrayLike] = 0.0 # dpw = (p4 - p4w) / (rho * Uinf^2)
    us: Union[float, npt.ArrayLike] = 0.0
    A4: Union[float, npt.ArrayLike] = 0.0
    beta: Union[float, npt.ArrayLike] = 0.0 # blockage ratio


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
        # get values depending on convergence
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
            - beta_s (float, optional)
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
    def __init__(self, beta_s=0.1403, cached=True, v4_correction=1.0, **kwargs):
        """
        Initialize the UnifiedMomentum instance.
        See above class documentation on __init__ for more details.
        """
        self.beta_s = beta_s
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

    def residual(self, x: np.ndarray, Ctprime: float, *args: float, **kwargs: float) -> Tuple[float, ...]:
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
        e_x0 = (
            np.cos(self.eff_yaw)
            / (2 * self.beta_s)
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
            beta_s=self.beta_s,
        )


@adaptivefixedpointiteration(max_iter=10000, relaxations=[0.4, 0.6], tolerance=0.00001)
class ThrustBasedUnified(UnifiedMomentum):
    """
    Solves the ThrustBasedUnified momentum model for an actuator disk. See Liew et al, 2024. Uses an iterative solver.
    Has one extra equation compared to the UnifiedMomentum solver that allows CT as an input rather than CT'.

    __init__:
        - Args:
            - beta_s (float, optional)
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
    def __init__(self, beta_s=0.1403, cached=True):
        super().__init__(beta_s=beta_s, cached=cached)

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
            beta_s=self.beta_s,
        )

""" Unified Blockage Model (Upfal et al. 2026) """
class UnifiedBlockage(MomentumBase):
    def residual(self, x, Ctprime, yaw, beta):
        '''
        Inputs:
            x: Input vector [an, dPstar, us, A4, u4, v4]
            Ctprime : Local thrust coefficient
            yaw : Yaw angle in radians
            beta : Blockage ratio

        Returns:
            Residuals of the system of equations
        '''
        an, dPstar, us, A4, u4, v4 = x

        umm_model = UnifiedMomentum()
        umm_sol = umm_model(Ctprime, yaw)
        dp_UMM = umm_sol.dp
        dpw = - (1 - beta) * dp_UMM # dpw = p4 - p4w / (rho Uinf^2)

        r1 = 1 - np.sqrt(np.clip(
            ((1 - u4**2 - v4**2) / (Ctprime*(np.cos(yaw)**2) + 1e-6)) + 
            ((dPstar + dpw) / (0.5*Ctprime*(np.cos(yaw)**2))), 0, 1e6)
            ) - an
        r2 = (1 - an)*np.cos(yaw)/A4 - u4
        r3 = - (1/4)*Ctprime*((1-an)**2)*np.sin(yaw)*(np.cos(yaw)**2) - v4
        r4 = 1 + (beta * A4 * (1 - u4))/(1 - beta * A4) - us
        r5 = (
            (0.5 * Ctprime * ((1 - an)**2) * (np.cos(yaw)**3) + (1/beta)*(us**2 - dPstar - 1)) /
            (dpw - u4**2 + us**2)
        ) - A4
        r6 =  0.5*us**2 - 0.5 - dPstar

        return [r1, r2, r3, r4, r5, r6]
    

    def _initial_guess_umm(self, Ctprime, yaw,  beta):
        umm_model = UnifiedMomentum()
        umm_sol = umm_model(Ctprime, yaw)
        A4 = (1 - umm_sol.an) * np.cos(yaw) / (umm_sol.u4 + 1e-1) + 1
        us = 1 + (beta * A4 * (1 - umm_sol.u4))/(1 - beta * A4)
        dp = (us**2 - 1)/2
        initial_guess = [umm_sol.an, dp, us, A4, umm_sol.u4, umm_sol.v4]
        return initial_guess
    

    def _solve(self, Ctprime, yaw, beta, initial_guess):
        sol = fsolve(
            self.residual,
            initial_guess,
            args=(Ctprime, yaw, beta),
            xtol=1e-6,  
            full_output=True,
        )
        return sol

    def __call__(self, Ctprime, yaw, beta):
        '''
        Parameters:
        - Ctprime : Local thrust coefficient
        - yaw : Yaw angle in radians
        - beta : Blockage ratio

        Returns:
        - sol: BlockageSolution

        '''
        if beta > 0.4:
            beta_it1 = 0.15    
            initial_guess_it1 = self._initial_guess_umm(Ctprime, yaw, beta)
            sol_it1 = self._solve(Ctprime, yaw, beta_it1, initial_guess_it1)
            beta_it2 = 0.4
            initial_guess_it2 = sol_it1[0]
            sol_it2 = self._solve(Ctprime, yaw, beta_it2, initial_guess_it2)
            initial_guess = sol_it2[0]

        elif beta > 0.15:
            beta_it1 = 0.15
            initial_guess_it1 = self._initial_guess_umm(Ctprime, yaw, beta)
            sol_it1 = self._solve(Ctprime, yaw, beta_it1, initial_guess_it1)
            initial_guess = sol_it1[0]

        else:
            initial_guess = self._initial_guess_umm(Ctprime, yaw, beta)

        sol = self._solve(Ctprime, yaw, beta, initial_guess)

        umm_sol = UnifiedMomentum()(Ctprime, yaw)

        return BlockageSolution(
            Ctprime=Ctprime,
            yaw=yaw,
            an=sol[0][0],
            u4=sol[0][4],
            v4=sol[0][5],
            x0=umm_sol.x0,
            dp=sol[0][1],
            dpw = - (1 - beta) * umm_sol.dp,
            us=sol[0][2],
            A4=sol[0][3],
            converged=sol[2] == 1,
            beta=beta
            )


""" Thrust-based formulation of Unified Blockage Model (Upfal et al. 2026) """
blockage_solution_table = Path(__file__).parent / "Utilities/blockage_solution_table.csv"
@cache_polars(blockage_solution_table)
def generate_table():
    print("Generating table for ThrustBasedBlocked initial guess...")
    betas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    ctprimes = np.concat([[-2, -1], np.linspace(0.1, 30, 20), [40, 50, 60]])
    yaw_angles = np.radians(np.linspace(-45, 45, 10))
    results = []
    for beta in betas:
        for ctprime in ctprimes:
            for yaw in yaw_angles:
                ubm = UnifiedBlockage()
                sol = ubm(ctprime, yaw=yaw, beta=beta)
                _df = pl.DataFrame({
                    "beta": [beta],
                    "ctprime": [ctprime],
                    "yaw": [yaw],
                    "ct": [sol.Ct],
                    "an": [sol.an],
                    "dPstar": [sol.dp],
                    "us": [sol.us],
                    "A4": [sol.A4],
                    "u4": [sol.u4],
                    "v4": [sol.v4],
                    'success': [sol.converged]
                })
                if sol.converged:
                    results.append(_df)
    return pl.concat(results)

class ThrustBasedBlockage(UnifiedBlockage):
    def _initial_guess_umm(self, Ct, yaw, beta):
        umm_model = ThrustBasedUnified()
        umm_sol = umm_model(Ct, yaw)
        A4 = (1 - umm_sol.an) * np.cos(yaw) / (umm_sol.u4 + 1e-6)
        us = (
            (1/beta - umm_sol.u4 * A4) /
            (1/beta - A4)
        )
        dp = (us**2 - 1)/2
        return [umm_sol.an, dp, us, A4, umm_sol.Ctprime, umm_sol.u4, umm_sol.v4]
    
    def initial_guess_from_table(self, Ct, yaw, beta):
        df = generate_table()
        axes = df.select(['ct', 'yaw', 'beta']).to_numpy()
        an_data = df['an'].to_numpy()
        dPstar_data = df['dPstar'].to_numpy()
        us_data = df['us'].to_numpy()
        A4_data = df['A4'].to_numpy()
        ctprime_data = df['ctprime'].to_numpy()
        u4_data = df['u4'].to_numpy()
        v4_data = df['v4'].to_numpy()

        an_interp = LinearNDInterpolator(axes, an_data)
        dPstar_interp = LinearNDInterpolator(axes, dPstar_data)
        us_interp = LinearNDInterpolator(axes, us_data)
        A4_interp = LinearNDInterpolator(axes, A4_data)
        ctprime_interp = LinearNDInterpolator(axes, ctprime_data)
        u4_interp = LinearNDInterpolator(axes, u4_data)
        v4_interp = LinearNDInterpolator(axes, v4_data)

        an_0 = an_interp(Ct, yaw, beta)
        dPstar_0 = dPstar_interp(Ct, yaw, beta)
        us_0 = us_interp(Ct, yaw, beta)
        A4_0 = A4_interp(Ct, yaw, beta)
        ctprime_0 = ctprime_interp(Ct, yaw, beta)
        u4_0 = u4_interp(Ct, yaw, beta)
        v4_0 = v4_interp(Ct, yaw, beta)
        return np.array([an_0, dPstar_0, us_0, A4_0, ctprime_0, u4_0, v4_0])

    def residual(self, x, Ct, yaw, beta):
        '''
        Inputs:
            x: Input vector [an, dPstar, us, A4, Ctprime, u4, v4]
            Ct : Thrust coefficient
            yaw : Yaw angle in radians
            beta : Blockage ratio

        Returns:
            Residuals of the system of equations
        '''
        an, dPstar, us, A4, Ctprime, u4, v4 = x

        Ctprime = np.clip(Ctprime, 1e-3, 20)
        u4 = np.clip(u4, 0.0001, 1.01)
        an = np.clip(an, 0.0001, 0.999)

        r1, r2, r3, r4, r5, r6 = super().residual(
            [an, dPstar, us, A4, u4, v4], Ctprime, yaw, beta
        )
        r7 = (Ct / ((1-an)**2 * np.cos(yaw)**2 + 1e-6)) - Ctprime
        
        return [r1, r2, r3, r4, r5, r6, r7]

    def __call__(self, Ct, yaw, beta):
        '''

        Inputs:
        - Ct : Thrust coefficient
        - yaw : Yaw angle in radians
        - beta : Blockage ratio

        Returns:
        - sol: BlockageSolution
        '''

        # Use the Unified Momentum Model solution-based guess as an initial 
        # guess for low blockage ratios, and use a table-based initial guess 
        # for higher blockage ratios.

        if beta > 0.01:
            initial_guess = self.initial_guess_from_table(Ct, yaw, beta)
        else:
            initial_guess = self._initial_guess_umm(Ct, yaw, beta)

        
        sol = fsolve(
            lambda x: self.residual(x, Ct, yaw, beta),
            initial_guess,
            xtol=1e-6,
            full_output=True
        )

        umm_sol = UnifiedMomentum()(sol[0][4], yaw)

        return BlockageSolution(
            Ctprime=sol[0][4],
            yaw=yaw,
            an=sol[0][0],
            u4=sol[0][5],
            v4=sol[0][6],
            x0=umm_sol.x0,
            dp=sol[0][1],
            dpw = - (1 - beta) * umm_sol.dp,
            us=sol[0][2],
            A4=sol[0][3],
            converged=sol[2],
            beta=beta
        )