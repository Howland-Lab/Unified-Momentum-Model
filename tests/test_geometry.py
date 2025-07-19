import numpy as np
from pytest import approx
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw, eff_yaw_inv_rotation

def test_calc_eff_yaw():
    yaw, tilt = 1, 1
    assert calc_eff_yaw(0, 0) == 0, "No tilt and no yaw case incorrectly has an effective yaw."
    assert calc_eff_yaw(yaw, 0) == approx(calc_eff_yaw(0, tilt)), "Equivalent magnitude yaw and tilt have the same effective angle."
    assert calc_eff_yaw(-yaw, 0) != approx(calc_eff_yaw(0, tilt)), "Signs aren't being correctly incorporated into effective yaw."
    assert approx(np.cos(calc_eff_yaw(yaw, tilt))) == approx(np.cos(yaw) * np.cos(tilt)), "Effective angle doesn't match spherical law of cosines."

def test_eff_yaw_inv_rotation():
    # Note that these numbers are nonsense and this is to test the geometric rotations, not actual physics
    yaw, tilt = 1, 1 # positive yaw -> v4 < 0 & positive tilt -> w4 > 0
    init_u4, init_v4, init_w4 = 1, -1, 0

    wake_u, wake_v, wake_w = eff_yaw_inv_rotation(init_u4, init_v4, calc_eff_yaw(0, 0), 0, 0)
    assert wake_u == init_u4 and wake_v == init_v4 and wake_w == init_w4, "Wake should remain unaffected rotation with no yaw or tilt"

    wake_u, wake_v, wake_w = eff_yaw_inv_rotation(init_u4, init_v4, calc_eff_yaw(yaw, 0), yaw, 0)
    assert wake_u == init_u4 and wake_v == init_v4 and wake_w == init_w4, "Wake should remain unaffected rotation with no tilt"

    wake_u, wake_v, wake_w = eff_yaw_inv_rotation(init_u4, init_v4, calc_eff_yaw(0, tilt), 0, tilt)
    assert wake_u == init_u4, "X-vector should remain unaffected by rotation with tilt"
    assert approx(wake_v) == approx(0), "v4 should be zero in case with only tilt"
    assert approx(wake_w) == -init_v4, "w4 should be non-zero and equal to -v4 (initial value) in case with only positive tilt"

    wake_u, wake_v, wake_w = eff_yaw_inv_rotation(init_u4, init_v4, calc_eff_yaw(0, -tilt), 0, -tilt)
    assert approx(wake_v) == approx(0), "v4 should be zero in case with only tilt"
    assert approx(wake_w) == init_v4, "w4 should be non-zero and equal to v4 (initial value) in case with only negative tilt"

    wake_u, wake_v, wake_w  = eff_yaw_inv_rotation(init_u4, init_v4, calc_eff_yaw(yaw, tilt), yaw, tilt)
    assert wake_u == init_u4, "X-vector should remain unaffected by rotation with yaw and tilt"
    assert np.linalg.norm([init_u4, init_v4, init_w4]) == np.linalg.norm([wake_u, wake_v, wake_w]), "Vector in y-z plane should have the same magniutude before and after translation"
    print(wake_v, wake_w)
    assert wake_w != 0, "w4 should be non-zero post rotation with tilt"
    assert approx(np.arctan2(wake_w, wake_v) - np.pi) == approx(np.arctan2(-np.sin(tilt) * np.cos(yaw), np.sin(yaw))), "Angle between v4 and w4 isn't equal to angle between normal vector and y-axis."
