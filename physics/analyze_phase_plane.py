from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ground_truth_dynamics.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)


# Vehicle parameter used by the simulator
LR = 1.6

# Import the exact peak slip angle from the simulator
SIMULATOR_DIR = PROJECT_ROOT / "simulator"
sys.path.insert(0, str(SIMULATOR_DIR))

from vehicle_model import ALPHA_R_PEAK


def rear_saturation_boundary(
    beta,
    speed,
    lr,
    alpha_r_peak
):
    """
    Rear-tire saturation boundary in the beta-r phase plane.

    Derived from:

        alpha_r = atan2(vy - lr*r, vx)

    and

        vy = vx*tan(beta)

    with

        |alpha_r| = alpha_r_peak
    """

    vx = speed
    vy = vx * np.tan(beta)

    r_positive = (
        vy - vx * np.tan(alpha_r_peak)
    ) / lr

    r_negative = (
        vy + vx * np.tan(alpha_r_peak)
    ) / lr

    return r_negative, r_positive


def main():

    print("=" * 70)
    print("BETA-R PHASE-PLANE ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    surfaces = [
        "dry",
        "wet",
        "ice_snow",
    ]

    for surface in surfaces:

        surface_df = df[
            df["mu_surface"] == surface
        ].copy()

        plt.figure(figsize=(10, 7))

        # ------------------------------------------------------
        # Plot stable and unstable samples
        # ------------------------------------------------------

        stable = surface_df[
            ~surface_df["instability_flag"]
        ]

        unstable = surface_df[
            surface_df["instability_flag"]
        ]

        plt.scatter(
            stable["beta"],
            stable["yaw_rate"],
            s=3,
            alpha=0.25,
            label="Stable"
        )

        plt.scatter(
            unstable["beta"],
            unstable["yaw_rate"],
            s=4,
            alpha=0.5,
            label="Instability"
        )

        # ------------------------------------------------------
        # Boundary for each speed
        # ------------------------------------------------------

        beta_min = surface_df["beta"].min()
        beta_max = surface_df["beta"].max()

        beta_grid = np.linspace(
            beta_min,
            beta_max,
            1000
        )

        speeds = sorted(
            surface_df[
                "target_speed_kmh"
            ].unique()
        )

        for speed_kmh in speeds:

            speed_ms = speed_kmh / 3.6

            r_negative, r_positive = (
                rear_saturation_boundary(
                    beta_grid,
                    speed_ms,
                    LR,
                    ALPHA_R_PEAK
                )
            )

            plt.plot(
                beta_grid,
                r_negative,
                linewidth=1.2,
                label=f"{speed_kmh} km/h boundary"
            )

            plt.plot(
                beta_grid,
                r_positive,
                linewidth=1.2
            )

        # ------------------------------------------------------
        # Formatting
        # ------------------------------------------------------

        plt.axhline(
            0,
            linewidth=0.7
        )

        plt.axvline(
            0,
            linewidth=0.7
        )

        plt.xlabel(
            "Sideslip angle β [rad]"
        )

        plt.ylabel(
            "Yaw rate r [rad/s]"
        )

        plt.title(
            f"β–r Phase Plane — {surface}"
        )

        plt.grid(
            True,
            alpha=0.25
        )

        plt.legend(
            fontsize=8
        )

        plt.tight_layout()

        output_path = (
            FIGURE_DIR
            / f"beta_r_phase_plane_{surface}.png"
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

    # ----------------------------------------------------------
    # Combined phase plane
    # ----------------------------------------------------------

    plt.figure(figsize=(10, 7))

    for surface in surfaces:

        surface_df = df[
            df["mu_surface"] == surface
        ]

        unstable = surface_df[
            surface_df["instability_flag"]
        ]

        plt.scatter(
            unstable["beta"],
            unstable["yaw_rate"],
            s=4,
            alpha=0.35,
            label=surface
        )

    plt.xlabel(
        "Sideslip angle β [rad]"
    )

    plt.ylabel(
        "Yaw rate r [rad/s]"
    )

    plt.title(
        "β–r Phase Plane — Instability Samples"
    )

    plt.grid(
        True,
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "beta_r_phase_plane_instability.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )

    print("\n" + "=" * 70)
    print("PHASE-PLANE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()