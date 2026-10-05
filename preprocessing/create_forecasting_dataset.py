from pathlib import Path

import numpy as np
import pandas as pd


# ==============================================================
# Paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SENSOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "retrofit_sensor_stream.csv"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ground_truth_dynamics.csv"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "run_metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ==============================================================
# Forecasting configuration
# ==============================================================

SAMPLE_RATE_HZ = 20
DT = 1.0 / SAMPLE_RATE_HZ

HISTORY_SECONDS = 1.0
HISTORY_STEPS = int(
    HISTORY_SECONDS * SAMPLE_RATE_HZ
)

HORIZONS_SECONDS = [
    0.5,
    1.0,
    2.0,
]

HORIZON_STEPS = [
    int(h * SAMPLE_RATE_HZ)
    for h in HORIZONS_SECONDS
]


# Only real retrofit sensor channels are used as X.
FEATURE_COLUMNS = [
    "yaw_rate_meas",
    "lateral_accel_meas",
    "speed_meas",
    "steering_angle_meas",
]

TARGET_COLUMN = "stability_margin"


# ==============================================================
# Train / validation / test configuration
# ==============================================================

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

RANDOM_SEED = 42


# ==============================================================
# Run-level split
# ==============================================================

def split_runs(metadata):
    """
    Split complete runs into train/validation/test.

    No individual run may appear in more than one split.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    runs = metadata["Run_ID"].unique().copy()

    rng.shuffle(runs)

    n_runs = len(runs)

    n_train = int(
        TRAIN_FRACTION * n_runs
    )

    n_val = int(
        VAL_FRACTION * n_runs
    )

    train_runs = runs[:n_train]

    val_runs = runs[
        n_train:n_train + n_val
    ]

    test_runs = runs[
        n_train + n_val:
    ]

    return (
        np.sort(train_runs),
        np.sort(val_runs),
        np.sort(test_runs),
    )


# ==============================================================
# Sequence creation
# ==============================================================

def create_sequences_for_runs(
    sensor_df,
    target_df,
    run_ids,
):
    """
    Create forecasting sequences only within complete runs.

    X shape:
        [samples, history_steps, features]

    y shape:
        [samples, 3]

    where the three targets are the stability margin at:
        +0.5 s
        +1.0 s
        +2.0 s
    """

    X_sequences = []
    y_sequences = []

    sample_run_ids = []
    sample_times = []

    for run_id in run_ids:

        sensor_run = (
            sensor_df[
                sensor_df["Run_ID"] == run_id
            ]
            .sort_values("Time")
            .reset_index(drop=True)
        )

        target_run = (
            target_df[
                target_df["Run_ID"] == run_id
            ]
            .sort_values("Time")
            .reset_index(drop=True)
        )

        # ------------------------------------------------------
        # Verify alignment
        # ------------------------------------------------------

        if not np.array_equal(
            sensor_run["Time"].to_numpy(),
            target_run["Time"].to_numpy(),
        ):
            raise ValueError(
                f"Time grids do not match for Run_ID={run_id}"
            )

        sensor_values = (
            sensor_run[
                FEATURE_COLUMNS
            ]
            .to_numpy(dtype=np.float32)
        )

        target_values = (
            target_run[
                TARGET_COLUMN
            ]
            .to_numpy(dtype=np.float32)
        )

        times = (
            sensor_run["Time"]
            .to_numpy()
        )

        # ------------------------------------------------------
        # Last valid current-time index
        # ------------------------------------------------------

        maximum_horizon = max(
            HORIZON_STEPS
        )

        start_index = HISTORY_STEPS - 1

        end_index = (
            len(sensor_run)
            - maximum_horizon
        )

        # Need enough past samples and enough future samples.
        for t in range(
            start_index,
            end_index
        ):

            history_start = (
                t - HISTORY_STEPS + 1
            )

            history_end = t + 1

            future_indices = [
                t + horizon
                for horizon in HORIZON_STEPS
            ]

            X_window = sensor_values[
                history_start:history_end
            ]

            y_future = target_values[
                future_indices
            ]

            X_sequences.append(
                X_window
            )

            y_sequences.append(
                y_future
            )

            sample_run_ids.append(
                run_id
            )

            sample_times.append(
                times[t]
            )

    X = np.stack(
        X_sequences
    )

    y = np.stack(
        y_sequences
    )

    return (
        X,
        y,
        np.asarray(sample_run_ids),
        np.asarray(sample_times),
    )


# ==============================================================
# Save dataset
# ==============================================================

def save_split(
    name,
    X,
    y,
    run_ids,
    times,
):
    """
    Save one split as a compressed NumPy archive.
    """

    output_path = (
        OUTPUT_DIR
        / f"{name}_sequences.npz"
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        Run_ID=run_ids,
        Time=times,
    )

    print(
        f"Saved {name}: {output_path}"
    )

    print(
        f"  X shape: {X.shape}"
    )

    print(
        f"  y shape: {y.shape}"
    )


# ==============================================================
# Main
# ==============================================================

def main():

    print("=" * 70)
    print("FORECASTING DATASET CREATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------

    sensor = pd.read_csv(
        SENSOR_PATH
    )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print("\nINPUT DATA")
    print("-" * 70)

    print(
        "Sensor shape:",
        sensor.shape
    )

    print(
        "Ground truth shape:",
        ground_truth.shape
    )

    print(
        "Metadata shape:",
        metadata.shape
    )

    # ----------------------------------------------------------
    # Check expected columns
    # ----------------------------------------------------------

    for column in FEATURE_COLUMNS:

        if column not in sensor.columns:
            raise ValueError(
                f"Missing sensor feature: {column}"
            )

    if TARGET_COLUMN not in ground_truth.columns:

        raise ValueError(
            f"Missing target: {TARGET_COLUMN}"
        )

    # ----------------------------------------------------------
    # Check run split
    # ----------------------------------------------------------

    train_runs, val_runs, test_runs = (
        split_runs(metadata)
    )

    print("\nRUN SPLIT")
    print("-" * 70)

    print(
        "Train runs:",
        len(train_runs)
    )

    print(
        "Validation runs:",
        len(val_runs)
    )

    print(
        "Test runs:",
        len(test_runs)
    )

    # ----------------------------------------------------------
    # Verify no run overlap
    # ----------------------------------------------------------

    train_set = set(train_runs)
    val_set = set(val_runs)
    test_set = set(test_runs)

    assert train_set.isdisjoint(
        val_set
    )

    assert train_set.isdisjoint(
        test_set
    )

    assert val_set.isdisjoint(
        test_set
    )

    print(
        "Run overlap check: PASSED"
    )

    # ----------------------------------------------------------
    # Create sequences
    # ----------------------------------------------------------

    print("\nCREATING TRAIN SEQUENCES")
    print("-" * 70)

    X_train, y_train, train_ids, train_times = (
        create_sequences_for_runs(
            sensor,
            ground_truth,
            train_runs,
        )
    )

    print("\nCREATING VALIDATION SEQUENCES")
    print("-" * 70)

    X_val, y_val, val_ids, val_times = (
        create_sequences_for_runs(
            sensor,
            ground_truth,
            val_runs,
        )
    )

    print("\nCREATING TEST SEQUENCES")
    print("-" * 70)

    X_test, y_test, test_ids, test_times = (
        create_sequences_for_runs(
            sensor,
            ground_truth,
            test_runs,
        )
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    save_split(
        "train",
        X_train,
        y_train,
        train_ids,
        train_times,
    )

    save_split(
        "validation",
        X_val,
        y_val,
        val_ids,
        val_times,
    )

    save_split(
        "test",
        X_test,
        y_test,
        test_ids,
        test_times,
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("FORECASTING DATASET COMPLETE")
    print("=" * 70)

    print(
        f"History: {HISTORY_SECONDS} s "
        f"({HISTORY_STEPS} samples)"
    )

    print(
        "Horizons:",
        HORIZONS_SECONDS
    )

    print(
        "Features:",
        FEATURE_COLUMNS
    )

    print(
        "Target:",
        TARGET_COLUMN
    )

    print(
        "\nTrain samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_val)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    print("=" * 70)


if __name__ == "__main__":
    main()