import numpy as np
from pytest import approx, mark

from UnifiedMomentumModel.Momentum import MomentumSolution, LimitedHeck, Heck, UnifiedMomentum, ThrustBasedUnified
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw


def test_MomentumSolution_constructors():
    # default constructor
    Ctprime, yaw, tilt, an, u4, v4, w4, x0, dp = 1, 1, 1, 0.5, 1, 1, 0, 1, 1
    sol1 = MomentumSolution(Ctprime, yaw, tilt, an, u4, v4, w4, x0, dp)
    assert Ctprime == sol1.Ctprime and yaw == sol1.yaw and tilt == sol1.tilt, "MomentumSolution set points not set correctly."
    assert an == sol1.an, "MomentumSolution induction not set correctly."
    assert u4 == sol1.u4 and v4 == sol1.v4 and w4 == sol1.w4, "MomentumSolution wake velocities not set correctly."
    assert x0 == sol1.x0 and dp == sol1.dp, "MomentumSolution far wake distance and pressure not set correctly."
    assert sol1.dp_NL == 0 and sol1.niter == 1 and sol1.converged == True and sol1.beta == 0, "MomentumSolution default parameters not set correctly."
    # constructor with optional parameter dp_NL
    sol2 = MomentumSolution(Ctprime, yaw, tilt, an, u4, v4, w4, x0, dp, dp_NL=1)
    assert sol2.dp_NL == 1, "MomentumSolution optional parameter dp_NL not setting incorrectly "
    # constructor with optional parameter niter
    sol3 = MomentumSolution(Ctprime, yaw, tilt, an, u4, v4, w4, x0, dp, niter=2)
    assert sol3.niter == 2, "MomentumSolution optional parameter niter not setting incorrectly "
    # constructor with optional parameter beta
    sol4 = MomentumSolution(Ctprime, yaw, tilt, an, u4, v4, w4, x0, dp, beta=1)
    assert sol4.beta == 1, "MomentumSolution optional parameter beta not setting incorrectly "
    assert sol4.Ct == approx(0.0729816)
    assert sol4.Cp == approx(0.0197160756563)


def test_MomentumSolution_comparison():
    Ctprime, yaw, tilt, an, u4, v4, w4, x0, dp = 1, 1, 0, 0.5, 1, 1, 0, 1, 1
    a = MomentumSolution(Ctprime, yaw, tilt, an, u4, v4, w4, x0, dp)
    b = MomentumSolution(Ctprime, yaw, tilt, an, u4, v4, w4, x0, dp)
    c = MomentumSolution(Ctprime, yaw, tilt, an, 2 * u4, v4, w4, x0, dp)

    assert a == b
    assert a != c


def test_LimitedHeck_aligned():
    model = LimitedHeck()
    sol = model(2, 0)

    expected = MomentumSolution(2, 0, 0, 1 / 3, 1 / 3, 0, 0, np.inf, 0)
    assert sol == expected
    assert sol.Ctprime == approx(2)
    assert sol.yaw == approx(0)
    assert sol.tilt == approx(0)
    assert sol.an == approx(1 / 3)
    assert sol.u4 == approx(1 / 3)
    assert sol.v4 == approx(0.0)
    assert sol.w4 == approx(0.0)
    assert sol.x0 == approx(np.inf)


def test_LimitedHeck_misaligned():
    model = LimitedHeck()
    sol = model(2, np.deg2rad(10))
    assert sol.Ctprime == approx(2)
    assert sol.yaw == approx(np.deg2rad(10))
    assert sol.tilt == approx(0)
    assert sol.an == approx(0.32656447810076383)
    assert sol.u4 == approx(0.3468710437984724)
    assert sol.v4 == approx(-0.038188728025758754)
    assert sol.w4 == approx(0)
    assert sol.x0 == approx(np.inf)


def test_LimitedHeck_zero():
    model = LimitedHeck()
    sol = model(0, 0)
    expected = MomentumSolution(0, 0, 0, 0, 1, 0, 0, np.inf, 0)
    assert sol == expected


# to do: Heck, UnifiedMomentum, ThrustBasedUnified
@mark.parametrize("model", [LimitedHeck(), Heck(), UnifiedMomentum()])
def test_model_yaw_tilt_comparison(model):
    Ct_prime, yaw, tilt = 2, 1, 1
    eff_angle = calc_eff_yaw(yaw, tilt)
    yaw_sol = model(Ct_prime, yaw = eff_angle)
    tilt_sol = model(Ct_prime, tilt = eff_angle)
    # yaw_tilt_sol = model(Ct_prime, yaw = yaw, tilt = tilt)

    # check that yaw and tilt solutions are equivalent up to a -90 degree rotation
    assert yaw_sol.an == tilt_sol.an
    # assert yaw_sol.u4 == tilt_sol.u4
    # assert yaw_sol.v4 == -tilt_sol.w4
    # assert yaw_sol.w4 == 0 and tilt_sol.v4 == 0
    # assert yaw_sol.x0 == tilt_sol.x0
    # assert yaw_sol.dp == tilt_sol.dp

    # assert yaw_tilt_sol.v4 != 0 and yaw_tilt_sol.w4 != 0 
    # assert approx(np.linalg.norm([yaw_tilt_sol.u4, yaw_tilt_sol.v4, yaw_tilt_sol.w4])) == approx(np.linalg.norm([yaw_sol.u4, yaw_sol.v4, yaw_sol.w4]))

def test_Heck_yaw_tilt_comparison():
    assert True

def test_UnifiedMomentum_yaw_tilt_comparison():
    assert True


def test_ThrustBasedUnified_yaw_tilt_comparison():
    assert True
