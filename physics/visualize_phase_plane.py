from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stability_boundary import rear_saturation_boundary


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


LR = 1.6
ALPHA_R_PEAK = 0.140


def main():

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(DATA_PATH)

    # Select one representative run
    run_id = df["Run_ID"].iloc[0]

    run = df[df["Run_ID"] == run_id].copy()

    speed = (
        run["target_speed_kmh"].iloc[0]
        / 3.6
    )

    beta = run["beta"].to_numpy()
    yaw_rate = run["yaw_rate"].to_numpy()

    beta_grid = np.linspace(
        beta.min() - 0.05,
        beta.max() + 0.05,
        500
    )

    r_lower, r_upper = rear_saturation_boundary(
        beta_grid,
        speed,
        LR,
        ALPHA_R_PEAK
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        beta_grid,
        r_upper,
        label="Rear saturation boundary"
    )

    plt.plot(
        beta_grid,
        r_lower
    )

    plt.scatter(
        beta,
        yaw_rate,
        c=run["stability_margin"],
        s=8
    )

    plt.xlabel("Sideslip angle β [rad]")
    plt.ylabel("Yaw rate r [rad/s]")

    plt.title(
        f"β–r Phase Plane — Run {run_id}"
    )

    plt.grid(True)
    plt.colorbar(
        label="Stability margin"
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "beta_r_phase_plane.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.show()

    print(
        f"Saved figure to: {output_path}"
    )


if __name__ == "__main__":
    main()