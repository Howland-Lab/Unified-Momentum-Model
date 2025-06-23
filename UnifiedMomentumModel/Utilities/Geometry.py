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
    Returns the effective angle, combining yaw and tilt. This effectively changes the frame of
    reference to one where the lateral wake velocity is aligned with the y' axis. This is
    equivalent to just having yaw in the ground frame.
    """
    eff_yaw = yaw
    if tilt != 0:
        eff_yaw =  np.arccos(np.cos(yaw) * np.cos(tilt))
    return eff_yaw

def eff_yaw_inv_rotation(eff_u, eff_v, eff_yaw, yaw, tilt):
    """
    Changes frame of reference back to the ground frame from the yaw-only frame created by
    aligning the y' axis with the lateral wake velocity.
    """
    vec = np.array([eff_u, eff_v, 0.0], dtype = float)
    if tilt != 0:
        cos_a = np.sin(yaw) / np.sin(eff_yaw)
        sin_a = - (np.sin(tilt) * np.cos(yaw)) / np.sin(eff_yaw)
        rot_mat = np.array([[1, 0, 0], [0, cos_a, -1 * sin_a], [0, sin_a, cos_a]])
        vec = rot_mat @ vec
    return vec

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
