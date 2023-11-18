from UnifiedMomentumModel.Momentum import MomentumSolution, LimitedHeck
import numpy as np


def test_MomentumSolution():
    MomentumSolution(1, 1, 0.5, 1, 1, 1, 1)
    MomentumSolution(1, 1, 0.5, 1, 1, 1, 1, dp_NL=1)
    MomentumSolution(1, 1, 0.5, 1, 1, 1, 1, niter=1)
    sol = MomentumSolution(1, 1, 0.5, 1, 1, 1, 1, beta=1)

    an, u4, v4, dp = sol.solution

    assert an == sol.an
    assert u4 == sol.u4
    assert v4 == sol.v4
    assert dp == sol.dp
    assert np.allclose(sol.Ct, 0.0729816)
    assert np.allclose(sol.Cp, 0.0197160756563)


def test_MomentumSolution_comparison():
    a = MomentumSolution(1, 1, 0.5, 1, 1, 1, 1)
    b = MomentumSolution(1, 1, 0.5, 1, 1, 1, 1)
    c = MomentumSolution(1, 1, 0.5, 2, 1, 1, 1)

    assert a == b
    assert a != c


def test_LimitedHeck_aligned():
    model = LimitedHeck()
    sol = model.solve(2, 0)

    expected = MomentumSolution(2, 0, 1 / 3, 1 / 3, 0, np.inf, 0)
    assert sol == expected
    assert np.allclose(sol.Ctprime, 2)
    assert np.allclose(sol.yaw, 0)
    assert np.allclose(sol.an, 1 / 3)
    assert np.allclose(sol.u4, 1 / 3)
    assert np.allclose(sol.v4, 0.0)
    assert np.allclose(sol.x0, np.inf)


def test_LimitedHeck_misaligned():
    model = LimitedHeck()
    sol = model.solve(2, np.deg2rad(10))
    assert np.allclose(sol.Ctprime, 2)
    assert np.allclose(sol.yaw, np.deg2rad(10))
    assert np.allclose(sol.an, 0.32656447810076383)
    assert np.allclose(sol.u4, 0.3468710437984724)
    assert np.allclose(sol.v4, -0.038188728025758754)
    assert np.allclose(sol.x0, np.inf)


def test_LimitedHeck_zero():
    model = LimitedHeck()
    sol = model.solve(0, 0)
    expected = MomentumSolution(0, 0, 0, 1, 0, np.inf, 0)
    assert sol == expected


# to do: Heck, UnifiedMomentum, ThrustBasedUnified
