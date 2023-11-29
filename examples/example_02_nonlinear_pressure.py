from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from UnifiedMomentumModel.Pressure.PressureTable import generate_pressure_table

# Use Latex Fonts
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

fig_fn = FIGDIR / "example_02_combined_pressure.png"


def main():
    dps, xs, ps = generate_pressure_table(progress=True)
    
    print(np.sum(ps))

    levels = np.arange(-0.3, 0.001, 0.025)

    plt.figure(figsize=(6, 3))
    CF = plt.contourf(xs, dps, ps, levels=levels, cmap="viridis_r")
    CS = plt.contour(xs, dps, ps, levels=levels, colors="k")
    plt.clabel(CS, inline=True, fontsize=10, fmt="%1.3f")

    cbar = plt.colorbar(CF)
    cbar.set_label(label=r"$p^{NL}/\rho$ [-]")

    plt.ylim(0.3, 1)

    plt.xlabel("$x_0$ [D]")
    plt.ylabel(r"$(p_2-p_3)/\rho$ [-]")

    plt.savefig(fig_fn, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
