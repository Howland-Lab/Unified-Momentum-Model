import numpy as np
from pytest import approx

from UnifiedMomentumModel.Momentum import LimitedHeck, MomentumSolution


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
    assert sol.Ct == approx(0.0729816)
    assert sol.Cp == approx(0.0197160756563)


def test_MomentumSolution_comparison():
    a = MomentumSolution(1, 1, 0.5, 1, 1, 1, 1)
    b = MomentumSolution(1, 1, 0.5, 1, 1, 1, 1)
    c = MomentumSolution(1, 1, 0.5, 2, 1, 1, 1)

    assert a == b
    assert a != c


def test_LimitedHeck_aligned():
    model = LimitedHeck()
    sol = model(2, 0)

    expected = MomentumSolution(2, 0, 1 / 3, 1 / 3, 0, np.inf, 0)
    assert sol == expected
    assert sol.Ctprime == approx(2)
    assert sol.yaw == approx(0)
    assert sol.an == approx(1 / 3)
    assert sol.u4 == approx(1 / 3)
    assert sol.v4 == approx(0.0)
    assert sol.x0 == approx(np.inf)


def test_LimitedHeck_misaligned():
    model = LimitedHeck()
    sol = model(2, np.deg2rad(10))
    assert sol.Ctprime == approx(2)
    assert sol.yaw == approx(np.deg2rad(10))
    assert sol.an == approx(0.32656447810076383)
    assert sol.u4 == approx(0.3468710437984724)
    assert sol.v4 == approx(-0.038188728025758754)
    assert sol.x0 == approx(np.inf)


def test_LimitedHeck_zero():
    model = LimitedHeck()
    sol = model(0, 0)
    expected = MomentumSolution(0, 0, 0, 1, 0, np.inf, 0)
    assert sol == expected


# to do: Heck, UnifiedMomentum, ThrustBasedUnified
