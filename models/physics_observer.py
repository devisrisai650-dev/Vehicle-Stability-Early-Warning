from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==============================================================
# Project paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SENSOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "retrofit_sensor_stream.csv"
)

GT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ground_truth_dynamics.csv"
)

META_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "run_metadata.csv"
)

TEST_SEQUENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_sequences.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)


# ==============================================================
# Vehicle parameters
# ==============================================================

LR = 1.6


# ==============================================================
# Import exact physics constant
# ==============================================================

SIMULATOR_DIR = (
    PROJECT_ROOT
    / "simulator"
)

sys.path.insert(
    0,
    str(SIMULATOR_DIR)
)

from vehicle_model import ALPHA_R_PEAK


# ==============================================================
# Forecast horizons
# ==============================================================

HORIZON_NAMES = [
    "0.5s",
    "1.0s",
    "2.0s",
]


# ==============================================================
# Estimate lateral velocity
# ==============================================================

def estimate_vy_for_run(run_df):
    """
    Estimate lateral velocity using:

        vy_dot = ay - vx*r

    Integration is performed independently for each run,
    starting from vy = 0.

    Trapezoidal integration is used between consecutive
    20-Hz measurements.
    """

    run_df = (
        run_df
        .sort_values("Time")
        .reset_index(drop=True)
        .copy()
    )

    time = run_df["Time"].to_numpy(
        dtype=float
    )

    yaw_rate = run_df[
        "yaw_rate_meas"
    ].to_numpy(
        dtype=float
    )

    lateral_accel = run_df[
        "lateral_accel_meas"
    ].to_numpy(
        dtype=float
    )

    speed = run_df[
        "speed_meas"
    ].to_numpy(
        dtype=float
    )

    # Convert measured speed into m/s if necessary.
    # Sensor stream is already stored in m/s.
    vx = speed

    # Kinematic lateral-velocity derivative
    vy_dot = (
        lateral_accel
        - vx * yaw_rate
    )

    vy = np.zeros(
        len(run_df),
        dtype=float
    )

    for i in range(1, len(run_df)):

        dt = time[i] - time[i - 1]

        vy[i] = (
            vy[i - 1]
            + 0.5
            * (
                vy_dot[i - 1]
                + vy_dot[i]
            )
            * dt
        )

    return vy


# ==============================================================
# Estimate beta and rear slip
# ==============================================================

def estimate_physics_states(run_df):
    """
    Estimate beta, rear slip angle and stability margin
    using only retrofit sensor measurements.
    """

    run_df = (
        run_df
        .sort_values("Time")
        .reset_index(drop=True)
        .copy()
    )

    vy = estimate_vy_for_run(
        run_df
    )

    vx = run_df[
        "speed_meas"
    ].to_numpy(
        dtype=float
    )

    yaw_rate = run_df[
        "yaw_rate_meas"
    ].to_numpy(
        dtype=float
    )

    vx_safe = np.maximum(
        vx,
        1.0
    )

    # Sideslip estimate
    beta_est = np.arctan2(
        vy,
        vx_safe
    )

    # Rear slip estimate
    alpha_r_est = np.arctan2(
        vy - LR * yaw_rate,
        vx_safe
    )

    # Physics-derived stability margin
    margin_est = (
        ALPHA_R_PEAK
        - np.abs(alpha_r_est)
    ) / ALPHA_R_PEAK

    result = pd.DataFrame(
        {
            "Run_ID":
                run_df["Run_ID"].to_numpy(),

            "Time":
                run_df["Time"].to_numpy(),

            "beta_est":
                beta_est,

            "alpha_r_est":
                alpha_r_est,

            "margin_est":
                margin_est,
        }
    )

    return result


# ==============================================================
# Build estimates for complete sensor dataset
# ==============================================================

def build_observer_estimates(sensor):
    """
    Run the observer independently for every Run_ID.
    """

    estimates = []

    for run_id, run_df in sensor.groupby(
        "Run_ID",
        sort=True
    ):

        run_estimate = estimate_physics_states(
            run_df
        )

        estimates.append(
            run_estimate
        )

    return pd.concat(
        estimates,
        ignore_index=True
    )


# ==============================================================
# Main
# ==============================================================

def main():

    print("=" * 70)
    print("SENSOR-ONLY PHYSICS OBSERVER BASELINE")
    print("=" * 70)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------

    sensor = pd.read_csv(
        SENSOR_PATH
    )

    gt = pd.read_csv(
        GT_PATH
    )

    test_sequences = np.load(
        TEST_SEQUENCE_PATH
    )

    test_run_ids = test_sequences[
        "Run_ID"
    ]

    test_times = test_sequences[
        "Time"
    ]

    y_test = test_sequences[
        "y"
    ]

    print("\nINPUTS")
    print("-" * 70)

    print(
        "Sensor shape:",
        sensor.shape
    )

    print(
        "Test target shape:",
        y_test.shape
    )

    print(
        "Exact ALPHA_R_PEAK:",
        repr(ALPHA_R_PEAK)
    )

    # ----------------------------------------------------------
    # Compute physics estimates
    # ----------------------------------------------------------

    print(
        "\nComputing sensor-only physics estimates..."
    )

    estimates = build_observer_estimates(
        sensor
    )

    # ----------------------------------------------------------
    # Create lookup for estimated current margin
    # ----------------------------------------------------------

    estimates["key"] = list(
        zip(
            estimates["Run_ID"].astype(int),
            estimates["Time"].astype(float)
        )
    )

    estimate_lookup = dict(
        zip(
            estimates["key"],
            estimates["margin_est"]
        )
    )

    # ----------------------------------------------------------
    # Obtain current estimated margin for test samples
    # ----------------------------------------------------------

    current_margin_est = []

    for run_id, time in zip(
        test_run_ids,
        test_times
    ):

        key = (
            int(run_id),
            float(time)
        )

        if key not in estimate_lookup:

            raise KeyError(
                "Missing observer estimate for "
                f"Run_ID={run_id}, Time={time}"
            )

        current_margin_est.append(
            estimate_lookup[key]
        )

    current_margin_est = np.asarray(
        current_margin_est,
        dtype=np.float32
    )

    # ----------------------------------------------------------
    # Persistence of physics-derived estimate
    # ----------------------------------------------------------

    y_pred = np.repeat(
        current_margin_est[:, None],
        3,
        axis=1
    )

    # ----------------------------------------------------------
    # Regression metrics
    # ----------------------------------------------------------

    results = []

    for i, horizon in enumerate(
        HORIZON_NAMES
    ):

        y_true = y_test[:, i]

        prediction = y_pred[:, i]

        mae = mean_absolute_error(
            y_true,
            prediction
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                prediction
            )
        )

        r2 = r2_score(
            y_true,
            prediction
        )

        results.append(
            {
                "horizon": horizon,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # ----------------------------------------------------------
    # Current observer quality
    # ----------------------------------------------------------

    # Compare estimated current margin with actual current
    # ground-truth margin at the same test timestamps.
    gt_lookup = dict(
        zip(
            zip(
                gt["Run_ID"].astype(int),
                gt["Time"].astype(float)
            ),
            gt["stability_margin"].astype(float)
        )
    )

    current_margin_true = np.asarray(
        [
            gt_lookup[
                (
                    int(run_id),
                    float(time)
                )
            ]
            for run_id, time in zip(
                test_run_ids,
                test_times
            )
        ],
        dtype=np.float32
    )

    observer_current_mae = (
        mean_absolute_error(
            current_margin_true,
            current_margin_est
        )
    )

    observer_current_rmse = np.sqrt(
        mean_squared_error(
            current_margin_true,
            current_margin_est
        )
    )

    observer_current_r2 = r2_score(
        current_margin_true,
        current_margin_est
    )

    # ----------------------------------------------------------
    # Display results
    # ----------------------------------------------------------

    print("\nCURRENT MARGIN ESTIMATION")
    print("-" * 70)

    print(
        "MAE:",
        observer_current_mae
    )

    print(
        "RMSE:",
        observer_current_rmse
    )

    print(
        "R2:",
        observer_current_r2
    )

    print("\nPHYSICS OBSERVER FORECAST RESULTS")
    print("-" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Save metrics
    # ----------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "physics_observer_metrics.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved:",
        output_path
    )

    # ----------------------------------------------------------
    # Save observer estimates
    # ----------------------------------------------------------

    estimate_output = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "physics_observer_estimates.csv"
    )

    estimates.drop(
        columns=["key"]
    ).to_csv(
        estimate_output,
        index=False
    )

    print(
        "Saved:",
        estimate_output
    )

    print("\n" + "=" * 70)
    print("PHYSICS OBSERVER BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()