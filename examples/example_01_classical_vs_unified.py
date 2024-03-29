from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from UnifiedMomentumModel import Momentum

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)


momentum_theories = {
    "Limited Heck": Momentum.LimitedHeck(),
    "Heck": Momentum.Heck(),
    "Unified Momentum": Momentum.UnifiedMomentum(),
    "Unified Momentum (linear)": Momentum.UnifiedMomentum(cached=False, max_iter=0),
}


def main():
    yaw = np.deg2rad(-0)
    Ctprime = np.linspace(-1, 100, 500)
    out = {}
    for key, model in momentum_theories.items():
        sol = model(Ctprime, yaw)
        print(key, sol.niter)
        out[key] = sol

    fig, axes = plt.subplots(4, 1, sharex=True)
    for key, sol in out.items():
        axes[0].plot(sol.Ctprime, sol.an, label=key)
        axes[1].plot(sol.Ctprime, sol.u4, label=key)
        axes[2].plot(sol.Ctprime, sol.v4, label=key)
        axes[3].plot(sol.Ctprime, sol.dp, label=key)
    axes[0].legend()

    axes[-1].set_xlabel("$C_T'$")
    axes[0].set_ylabel("$a_n$")
    axes[1].set_ylabel("$u_4$")
    axes[2].set_ylabel("$v_4$")
    axes[3].set_ylabel("$p_4-p_1$")

    plt.xlim(0, 12)
    plt.savefig(
        FIGDIR / "example_001_momentum_aligned.png", dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
