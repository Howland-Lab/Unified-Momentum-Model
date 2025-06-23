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
    init_u4, init_v4 = 1, 1
    init_wake_vels = [init_u4, init_v4, 0]  # velocity vector in the "yaw-only" frame of reference
    yaw, tilt = 1, 1
    assert (eff_yaw_inv_rotation(init_u4, init_v4, calc_eff_yaw(0, 0), 0, 0) == init_wake_vels).all(), "Wake velocities unchanged by rotation with no yaw and no tilt."
    assert (eff_yaw_inv_rotation(init_u4, init_v4, calc_eff_yaw(yaw, 0), yaw, 0) == init_wake_vels).all(), "Wake velocities unchanged by rotation with non-zero yaw but no tilt."

    wake_vels = eff_yaw_inv_rotation(init_u4, init_v4, calc_eff_yaw(0, tilt), 0, tilt)
    assert wake_vels[0] == init_u4, "X-vector should remain unaffected by rotation with tilt"
    assert approx(wake_vels[1]) == approx(0), "v4 should be zero in case with only tilt"
    assert approx(wake_vels[2]) == -init_v4, "w4 should be non-zero and equal to -v4 (initial value) in case with only positive tilt"

    wake_vels = eff_yaw_inv_rotation(init_u4, init_v4, calc_eff_yaw(0, -tilt), 0, -tilt)
    assert approx(wake_vels[1]) == approx(0), "v4 should be zero in case with only tilt"
    assert approx(wake_vels[2]) == init_v4, "w4 should be non-zero and equal to v4 (initial value) in case with only negative tilt"


    wake_vels = eff_yaw_inv_rotation(init_u4, init_v4, calc_eff_yaw(yaw, tilt), yaw, tilt)
    assert wake_vels[0] == init_u4, "X-vector should remain unaffected by rotation with yaw and tilt"
    assert np.linalg.norm(init_wake_vels) == np.linalg.norm(wake_vels), "Vector in y-z plane should have the same magniutude before and after translation"
    assert wake_vels[0] != 0, "w4 should be non-zero post rotation with tilt"
    assert approx(np.arctan2(wake_vels[2], wake_vels[1])) == approx(np.arctan2(-np.sin(tilt) * np.cos(yaw), np.sin(yaw))), "Angle between v4 and u4 isn't equal to angle between normal vector and y-axis."
