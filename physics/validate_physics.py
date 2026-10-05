from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_DIR = PROJECT_ROOT / "simulator"
sys.path.insert(0, str(SIMULATOR_DIR))

from stability_margin import (
    compute_stability_margin,
    compute_instability_flag,
)

from vehicle_model import ALPHA_R_PEAK


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ground_truth_dynamics.csv"
)


def main():

    print("=" * 70)
    print("PHYSICS VALIDATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Confirm the exact physical constant being used
    # ------------------------------------------------------------

    print(
        "\nALPHA_R_PEAK used for validation:",
        repr(ALPHA_R_PEAK)
    )

    df = pd.read_csv(DATA_PATH)

    # ------------------------------------------------------------
    # 1. Recalculate stability margin
    # ------------------------------------------------------------

    calculated_margin = compute_stability_margin(
        df["alpha_r"].to_numpy(),
        ALPHA_R_PEAK,
    )

    stored_margin = df["stability_margin"].to_numpy()

    margin_error = calculated_margin - stored_margin

    print("\n1. STABILITY MARGIN CHECK")
    print("-" * 70)

    print(
        "Maximum absolute error:",
        np.max(np.abs(margin_error)),
    )

    print(
        "Mean absolute error:",
        np.mean(np.abs(margin_error)),
    )

    print(
        "Margins match:",
        np.allclose(
            calculated_margin,
            stored_margin,
            atol=1e-12,
            rtol=1e-12,
        ),
    )

    # ------------------------------------------------------------
    # 2. Recalculate instability flags
    # ------------------------------------------------------------

    calculated_flags = np.zeros(
        len(df),
        dtype=bool,
    )

    for run_id, indices in df.groupby("Run_ID").groups.items():

        indices = np.asarray(indices)

        run_margin = calculated_margin[indices]

        run_flags = compute_instability_flag(
            run_margin,
            consecutive_samples=3,
        )

        calculated_flags[indices] = run_flags

    stored_flags = (
        df["instability_flag"]
        .astype(bool)
        .to_numpy()
    )

    flag_matches = calculated_flags == stored_flags

    print("\n2. INSTABILITY FLAG CHECK")
    print("-" * 70)

    print(
        "Matching samples:",
        flag_matches.sum(),
        "/",
        len(flag_matches),
    )

    print(
        "Mismatch count:",
        (~flag_matches).sum(),
    )

    print(
        "Flags match:",
        np.array_equal(
            calculated_flags,
            stored_flags,
        ),
    )

    # ------------------------------------------------------------
    # 3. Distribution
    # ------------------------------------------------------------

    print("\n3. STABILITY MARGIN DISTRIBUTION")
    print("-" * 70)

    print(
        pd.Series(
            calculated_margin
        ).describe()
    )

    print(
        "\nPositive-margin samples:",
        np.sum(calculated_margin > 0),
    )

    print(
        "Zero-margin samples:",
        np.sum(
            np.isclose(
                calculated_margin,
                0,
                atol=1e-12,
            )
        ),
    )

    print(
        "Negative-margin samples:",
        np.sum(calculated_margin < 0),
    )

    # ------------------------------------------------------------
    # 4. Road-surface analysis
    # ------------------------------------------------------------

    print("\n4. VALIDATION BY ROAD SURFACE")
    print("-" * 70)

    surface_results = []

    for surface, group in df.groupby("mu_surface"):

        indices = group.index.to_numpy()

        margin = calculated_margin[indices]

        surface_results.append(
            {
                "surface": surface,
                "samples": len(group),
                "min_margin": margin.min(),
                "mean_margin": margin.mean(),
                "negative_margin_pct":
                    100 * np.mean(margin < 0),
            }
        )

    surface_df = pd.DataFrame(surface_results)

    print(
        surface_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # 5. State range
    # ------------------------------------------------------------

    print("\n5. STATE RANGE CHECK")
    print("-" * 70)

    print(
        "Beta range:",
        df["beta"].min(),
        "to",
        df["beta"].max(),
        "rad",
    )

    print(
        "Yaw-rate range:",
        df["yaw_rate"].min(),
        "to",
        df["yaw_rate"].max(),
        "rad/s",
    )

    print(
        "Rear slip angle range:",
        df["alpha_r"].min(),
        "to",
        df["alpha_r"].max(),
        "rad",
    )

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    print("\n" + "=" * 70)

    margins_ok = np.allclose(
        calculated_margin,
        stored_margin,
        atol=1e-12,
        rtol=1e-12,
    )

    flags_ok = np.array_equal(
        calculated_flags,
        stored_flags,
    )

    if margins_ok and flags_ok:
        print("PHYSICS VALIDATION PASSED")
    else:
        print("PHYSICS VALIDATION REQUIRES INVESTIGATION")

    print("=" * 70)


if __name__ == "__main__":
    main()