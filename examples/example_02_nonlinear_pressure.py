from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import polars as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
from UnifiedMomentumModel.PressureSolver.ADPressureField import (
    AccumulatedNonlinearADPressureField,
)


# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

fig_fn = FIGDIR / "example_02_combined_pressure.png"


def main():
    # dp_mesh, x_mesh = np.meshgrid(dps, xs, indexing="ij")

    centerline_pressure = AccumulatedNonlinearADPressureField()
    dps, xs = centerline_pressure.interpolator.grid
    combined_field = centerline_pressure.interpolator.values

    levels = np.arange(-0.3, 0.001, 0.025)

    plt.figure(figsize=(6, 3))
    CF = plt.contourf(xs, dps, combined_field, levels=levels, cmap="viridis_r")
    CS = plt.contour(xs, dps, combined_field, levels=levels, colors="k")
    plt.clabel(CS, inline=True, fontsize=10, fmt="%1.3f")

    cbar = plt.colorbar(CF)
    cbar.set_label(label=r"$p^{NL}/\rho$ [-]")

    plt.ylim(0.3, 1)

    plt.xlabel("$x_0$ [D]")
    plt.ylabel(r"$(p_2-p_3)/\rho$ [-]")

    plt.savefig(fig_fn, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
