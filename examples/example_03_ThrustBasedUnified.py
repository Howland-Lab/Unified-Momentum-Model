from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from UnifiedMomentumModel import Momentum

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)


def main():
    yaw = np.deg2rad(-0)
    Ctprime = np.linspace(-1, 2, 500)
    model = Momentum.ThrustBasedUnified()

    sol = model(Ctprime, yaw)

    print(sol)


if __name__ == "__main__":
    main()
