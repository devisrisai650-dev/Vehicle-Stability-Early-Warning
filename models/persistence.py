from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==============================================================
# Paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_SEQUENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_sequences.npz"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ground_truth_dynamics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)


HORIZONS = [
    "0.5s",
    "1.0s",
    "2.0s",
]


# ==============================================================
# Load current margin
# ==============================================================

def build_margin_lookup(gt):
    """
    Create a lookup:

        (Run_ID, Time) -> current stability margin
    """

    lookup = {}

    for row in gt.itertuples(index=False):

        key = (
            int(row.Run_ID),
            float(row.Time),
        )

        lookup[key] = float(
            row.stability_margin
        )

    return lookup


# ==============================================================
# Main
# ==============================================================

def main():

    print("=" * 70)
    print("PERSISTENCE BASELINE")
    print("=" * 70)

    # ----------------------------------------------------------
    # Load test sequences
    # ----------------------------------------------------------

    test_data = np.load(
        TEST_SEQUENCE_PATH
    )

    X_test = test_data["X"]

    y_test = test_data["y"]

    test_run_ids = test_data["Run_ID"]

    test_times = test_data["Time"]

    print("\nTEST DATA")
    print("-" * 70)

    print(
        "X shape:",
        X_test.shape
    )

    print(
        "y shape:",
        y_test.shape
    )

    # ----------------------------------------------------------
    # Load ground truth
    # ----------------------------------------------------------

    gt = pd.read_csv(
        GROUND_TRUTH_PATH
    )

    margin_lookup = build_margin_lookup(
        gt
    )

    # ----------------------------------------------------------
    # Obtain current margin M(t)
    # ----------------------------------------------------------

    current_margin = []

    for run_id, time in zip(
        test_run_ids,
        test_times
    ):

        key = (
            int(run_id),
            float(time),
        )

        if key not in margin_lookup:

            raise KeyError(
                f"Missing ground-truth margin "
                f"for Run_ID={run_id}, Time={time}"
            )

        current_margin.append(
            margin_lookup[key]
        )

    current_margin = np.asarray(
        current_margin,
        dtype=np.float32
    )

    print(
        "\nCurrent-margin shape:",
        current_margin.shape
    )

    # ----------------------------------------------------------
    # Persistence prediction
    # ----------------------------------------------------------

    y_pred = np.repeat(
        current_margin[:, None],
        3,
        axis=1
    )

    print(
        "Prediction shape:",
        y_pred.shape
    )

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    results = []

    for i, horizon in enumerate(HORIZONS):

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
    # Display
    # ----------------------------------------------------------

    print("\nPERSISTENCE RESULTS")
    print("-" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "persistence_metrics.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nSaved:",
        output_path
    )

    print("\n" + "=" * 70)
    print("PERSISTENCE BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()