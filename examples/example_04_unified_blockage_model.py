from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from UnifiedMomentumModel.Momentum import UnifiedBlockage, UnifiedMomentum, ThrustBasedBlockage, ThrustBasedUnified

""" Unified Blockage Model from Upfal et al. 2026 """

FIGDIR = Path("fig")
FIGDIR.mkdir(exist_ok=True, parents=True)

def run_Ctprime_formulation():
    yaw_angles = np.deg2rad(np.linspace(-45, 45, 20))
    blockage_ratios = [0.005, 0.3]
    Ctprime = 2.0

    fig, ax = plt.subplots()

    # Run UMM
    umm = UnifiedMomentum()
    unblocked_an = [umm(Ctprime=Ctprime, yaw=yaw).an for yaw in yaw_angles]
    ax.plot(np.rad2deg(yaw_angles), unblocked_an, label="UMM (beta = 0)", linestyle="--", color="black")

    # Run UBM
    print("Running UBM with Ct' formulation...")
    ubm = UnifiedBlockage()
    for beta in blockage_ratios:
        an_sols = [ubm(Ctprime=Ctprime, yaw=yaw, beta=beta).an for yaw in yaw_angles]
        ax.plot(np.rad2deg(yaw_angles), an_sols, label="UBM (beta = {:.3f})".format(beta))

    ax.set_xlabel("Yaw angle (degrees)")
    ax.set_ylabel("Induction factor")
    ax.set_ylim(0.1, 0.35)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(
        FIGDIR / "example_04_blockage_model_Ctprime.png", dpi=300, bbox_inches="tight"
    )

def run_Ct_formulation():
    yaw_angles = np.deg2rad(np.linspace(-45, 45, 20))
    blockage_ratios = [0.005, 0.3]
    Ct = 0.8

    fig, ax = plt.subplots()

    # Run UMM
    umm = ThrustBasedUnified()
    unblocked_an = [umm(Ct, yaw).an for yaw in yaw_angles]
    ax.plot(np.rad2deg(yaw_angles), unblocked_an, label="UMM (beta = 0)", linestyle="--", color="black")

    # Run UBM
    print("Running UBM with Ct formulation...")
    ubm = ThrustBasedBlockage()
    for beta in blockage_ratios:
        an_sols = []
        yaws = []
        for yaw in yaw_angles:
            sol = ubm(Ct, yaw, beta)
            an_sols.append(sol.an)
            yaws.append(yaw)
        ax.plot(np.rad2deg(yaws), an_sols, label="UBM (beta = {:.3f})".format(beta))

    ax.set_xlabel("Yaw angle (degrees)")
    ax.set_ylabel("Induction factor")
    ax.set_ylim(0.1, 0.35)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(
        FIGDIR / "example_04_blockage_model_Ct.png", dpi=300, bbox_inches="tight"
    )

if __name__ == "__main__":
    run_Ctprime_formulation()
    run_Ct_formulation()