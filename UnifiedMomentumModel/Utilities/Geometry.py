from dataclasses import dataclass

import numpy as np


@dataclass
class Geometry:
    Lx: float
    Ly: float
    dx: float
    dy: float


@dataclass
class EquidistantRectGridOdd(Geometry):
    def __post_init__(self):
        self.x = np.arange(-self.Lx / 2, self.Lx / 2, self.dx)
        self.y = np.arange(-self.Ly / 2, self.Ly / 2, self.dy)

        self.Nx, self.Ny = len(self.x), len(self.y)
        self.shape = (self.Nx, self.Ny)

        self.xmesh, self.ymesh = np.meshgrid(self.x, self.y, indexing="ij")


@dataclass
class EquidistantRectGridEven(Geometry):
    def __post_init__(self):
        self.x = np.arange(-self.Lx / 2 + self.dx / 2, self.Lx / 2, self.dx)
        self.y = np.arange(-self.Ly / 2 + self.dy / 2, self.Ly / 2, self.dy)

        self.Nx, self.Ny = len(self.x), len(self.y)
        self.shape = (self.Nx, self.Ny)

        self.xmesh, self.ymesh = np.meshgrid(self.x, self.y, indexing="ij")

def calc_eff_yaw(yaw, tilt):
    """
    Returns the effective angle, combining yaw and tilt (in radians).
    
    Consider the rotation matrices:
    R_z(yaw) = [cos(yaw) -sin(yaw) 0; sin(yaw) cos(yaw) 0; 0 0 1]
    R_y(tilt) = [cos(tilt) 0 sin(tilt); 0 1 0; -sin(tilt) - cos(tilt)]

    and apply them to [1, 0, 0] to get the direction of the rotor normal of a yawed and tilted turbine
    n = R_y(tilt) * R_z(yaw) * n = [cos(yaw)cos(tilt), sin(yaw), -sin(tilt)cos(yaw)].

    Using the spherical law of cosines, let:
    cos(theta) = cos(yaw)cos(tilt) and
    sin(theta) = sqrt(cos^2(yaw) * sin^2(tilt) + sin^2(yaw))

    We then want to consider a frame that just has thrust in the x and y directions to be able to
    use the UMM as derived in Liew et al 2024. We let this new frame of reference be defined by x', y' and z'.
    We need to find the rotor normal n' in this new frame of reference where n' falls along y'. This is a rotation
    in the y-z plane around the x-axis.

    Recall: cos(a) = (A . B) / (|A||B|) and sin(a) = |A x B| / (|A||B|) where a is the angle between A and B.

    If A = [1, 0] and B = [sin(yaw), -sin(tilt)cos(yaw)], then:
    cos(a) = sin(yaw) / sin(theta)
    sin(a) = -sin(tilt)cos(yaw) / sin(theta)

    We can then apply the rotation matrix around the x axis as follows:
    R_x(a) = [1 0 0; 0 cos(a) sin(a); 0 -sin(a) cos(a)]
    n' = R_x(a) * n = [cos(theta), sin(theta), 0]

    In the UMM with just yaw, we have n = [cos(yaw), sin(yaw), 0]. Therefore, we solve the UMM in this rotated "yaw-only" frame,
    using eff_yaw = theta.

    Note: when tilt = 0, theta = eff_yaw = yaw and no rotation is needed.
    """
    eff_yaw = np.arccos(np.cos(yaw) * np.cos(tilt))
    eff_yaw = np.where(yaw == 0, np.abs(tilt), eff_yaw) # tilt sign reintroduced in rotation back to ground frame
    eff_yaw = np.where(tilt == 0, yaw, eff_yaw)
    return eff_yaw

def eff_yaw_inv_rotation(eff_u, eff_v, eff_w, eff_yaw, yaw, tilt):
    """
    Changes frame of reference back to the ground frame from the yaw-only frame created by aligning the y' axis
    with the rotor normal. We can use the inverse rotation matrix as applied above in the calc_eff_yaw function.

    R_x^(-1)(a) = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)] where the cos(a) and sin(a) derivation is described above.

    We can then apply this rotation matrix to the wake velocities to rotate them back into the ground frame.
    """
    # if tilt = 0, then no rotation took place to enter the "yaw-only" frame, so v4 = 1 * eff_v and w4 = 0 * eff_v
    cos_a = np.ones_like(eff_yaw)
    sin_a = np.zeros_like(eff_yaw)
    # if tilt != 0, then apply rotation matrix [cos_a -sin_a; sin_a cos_a] * [v_eff; 0] = [cos_a * veff ; -sin_a * veff]
    non_zero_tilt = (tilt != 0) & (eff_yaw != 0) # really, really small tilt can lead to zero eff yaw
    sin_eff = np.sin(eff_yaw)
    cos_a = np.divide(np.sin(yaw), sin_eff, where = non_zero_tilt, out = cos_a)
    sin_a = np.divide(-(np.sin(tilt) * np.cos(yaw)), sin_eff, where = non_zero_tilt, out = sin_a)
    # calculate wake velocities in the ground frame
    u = eff_u 
    v = cos_a * eff_v - sin_a * eff_w
    w = sin_a * eff_v + cos_a * eff_w
    return u, v, w

if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    figdir = Path("fig")
    figdir.mkdir(parents=True, exist_ok=True)

    Lx, Ly = 30, 30
    dx, dy = 1, 1

    geometry_classes = {
        "EquidistantRectGridOdd": EquidistantRectGridOdd,
        "EquidistantRectGridEven": EquidistantRectGridEven,
    }

    fig, axes = plt.subplots(len(geometry_classes), 1, sharex=True, sharey=True)
    for ax, (name, geom_class) in zip(axes, geometry_classes.items()):
        print(name)
        geometry = geom_class(Lx, Ly, dx, dy)
        extent = (
            geometry.xmesh.min(),
            geometry.xmesh.max(),
            geometry.ymesh.min(),
            geometry.ymesh.max(),
        )
        print(extent)

        ax.plot(geometry.xmesh.ravel(), geometry.ymesh.ravel(), ".", label=name)
        ax.legend()
        # ax.axis("equal")
        ax.axvline(0, lw=1, ls="-", c="k")
        ax.axhline(1, lw=1, ls="-", c="k")
        ax.axhline(-1, lw=1, ls="-", c="k")

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.savefig(figdir / "geometry.png", dpi=300, bbox_inches="tight")
