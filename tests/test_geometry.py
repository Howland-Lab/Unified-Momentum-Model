import numpy as np
from pytest import approx
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw, eff_yaw_inv_rotation, eff_yaw_rotation

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

    wake_u, wake_v, wake_w = eff_yaw_inv_rotation(init_u4, init_v4, init_w4, calc_eff_yaw(0, 0), 0, 0)
    assert wake_u == init_u4 and wake_v == init_v4 and wake_w == init_w4, "Wake should remain unaffected rotation with no yaw or tilt"

    wake_u, wake_v, wake_w = eff_yaw_inv_rotation(init_u4, init_v4, init_w4, calc_eff_yaw(yaw, 0), yaw, 0)
    assert wake_u == init_u4 and wake_v == init_v4 and wake_w == init_w4, "Wake should remain unaffected rotation with no tilt"

    wake_u, wake_v, wake_w = eff_yaw_inv_rotation(init_u4, init_v4, init_w4, calc_eff_yaw(0, tilt), 0, tilt)
    assert wake_u == init_u4, "X-vector should remain unaffected by rotation with tilt"
    assert approx(wake_v) == approx(0), "v4 should be zero in case with only tilt"
    assert approx(wake_w) == -init_v4, "w4 should be non-zero and equal to -v4 (initial value) in case with only positive tilt"

    wake_u, wake_v, wake_w = eff_yaw_inv_rotation(init_u4, init_v4, init_w4, calc_eff_yaw(0, -tilt), 0, -tilt)
    assert approx(wake_v) == approx(0), "v4 should be zero in case with only tilt"
    assert approx(wake_w) == init_v4, "w4 should be non-zero and equal to v4 (initial value) in case with only negative tilt"

    wake_u, wake_v, wake_w  = eff_yaw_inv_rotation(init_u4, init_v4, init_w4, calc_eff_yaw(yaw, tilt), yaw, tilt)
    assert wake_u == init_u4, "X-vector should remain unaffected by rotation with yaw and tilt"
    assert np.linalg.norm([init_u4, init_v4, init_w4]) == np.linalg.norm([wake_u, wake_v, wake_w]), "Vector in y-z plane should have the same magniutude before and after translation"

    assert wake_w != 0, "w4 should be non-zero post rotation with tilt"
    assert approx(np.arctan2(wake_w, wake_v) - np.pi) == approx(np.arctan2(-np.sin(tilt) * np.cos(yaw), np.sin(yaw))), "Angle between v4 and w4 isn't equal to angle between normal vector and y-axis."

def test_eff_yaw_rotation():
    # Note that these numbers are nonsense and this is to test the geometric rotations, not actual physics
    yaw, tilt = 1, 1 # positive yaw -> v4 < 0 & positive tilt -> w4 > 0
    init_u4, init_v4, init_w4 = 1, 0.5, 0.75

    wake_u, wake_v, wake_w = eff_yaw_rotation(init_u4, init_v4, init_w4, calc_eff_yaw(0, 0), 0, 0)
    assert wake_u == init_u4 and wake_v == init_v4 and wake_w == init_w4, "Wake should remain unaffected rotation with no yaw or tilt"

    wake_u, wake_v, wake_w = eff_yaw_rotation(init_u4, init_v4, init_w4, calc_eff_yaw(yaw, 0), yaw, 0)
    assert wake_u == init_u4 and wake_v == init_v4 and wake_w == init_w4, "Wake should remain unaffected rotation with no tilt"

    # test that rotor velocity is just in x and y in yaw-only frame
    for (yaw , tilt) in ((yaw, 0), (0, tilt), (yaw, tilt)):
        u4n, v4n, w4n = np.cos(tilt) * np.cos(yaw), np.sin(yaw), -np.sin(tilt) * np.cos(yaw)
        vw4n = np.sqrt(v4n**2 + w4n**2)
        u4np, v4np, w4np = eff_yaw_rotation(u4n, v4n, w4n, calc_eff_yaw(yaw, tilt), yaw, tilt)
        assert u4np == approx(u4n), "rotor u4 should remain unchanged when in yaw-only frame"
        assert v4np == approx(vw4n), "rotor v4 should have magnitude of combined v4 and w4 in yaw-only frame"
        assert approx(w4np) == approx(0), "rotor w4 should be zero in yaw-only frame"

    # ensure that rotation into yaw frame and rotation back into ground frame give the original velocities
    for (yaw, tilt) in ((yaw, 0), (0, tilt), (yaw, tilt)):
        eff_u4, eff_v4, eff_w4 = eff_yaw_rotation(init_u4, init_v4, init_w4, calc_eff_yaw(yaw, tilt), yaw, tilt)
        assert eff_u4 == approx(init_u4)
        if tilt != 0:
            assert eff_v4 != approx(init_v4)
            assert eff_w4 != approx(init_w4)
        else:
            assert eff_v4 == approx(init_v4)
            assert eff_w4 == approx(init_w4)
        final_u4, final_v4, final_w4 = eff_yaw_inv_rotation(eff_u4, eff_v4, eff_w4, calc_eff_yaw(yaw, tilt), yaw, tilt)
        assert final_u4 == approx(init_u4)
        assert final_v4 == approx(init_v4)
        assert final_w4 == approx(init_w4)
