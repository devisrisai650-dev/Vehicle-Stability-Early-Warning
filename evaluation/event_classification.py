from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ==============================================================
# Paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TABLE_DIR = RESULTS_DIR / "tables"


TEST_PATH = (
    DATA_DIR
    / "test_sequences.npz"
)


# ==============================================================
# Configuration
# ==============================================================

HORIZON_NAMES = [
    "0.5s",
    "1.0s",
    "2.0s",
]

# The physical instability threshold is zero stability margin.
WARNING_THRESHOLD = 0.0


# ==============================================================
# Load test targets
# ==============================================================

def load_test_data():

    data = np.load(TEST_PATH)

    y_test = data["y"]

    run_ids = data["Run_ID"]

    times = data["Time"]

    return (
        y_test,
        run_ids,
        times,
    )


# ==============================================================
# Classification metrics
# ==============================================================

def evaluate_predictions(
    y_true,
    y_pred,
    model_name,
):

    rows = []

    for i, horizon in enumerate(
        HORIZON_NAMES
    ):

        actual = (
            y_true[:, i]
            < WARNING_THRESHOLD
        )

        predicted = (
            y_pred[:, i]
            < WARNING_THRESHOLD
        )

        precision = precision_score(
            actual,
            predicted,
            zero_division=0,
        )

        recall = recall_score(
            actual,
            predicted,
            zero_division=0,
        )

        f1 = f1_score(
            actual,
            predicted,
            zero_division=0,
        )

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[False, True],
        ).ravel()

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        rows.append(
            {
                "model": model_name,
                "horizon": horizon,

                "precision": precision,
                "recall": recall,
                "F1": f1,

                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,

                "false_positive_rate":
                    false_positive_rate,
            }
        )

    return pd.DataFrame(rows)


# ==============================================================
# Main
# ==============================================================

def main():

    print("=" * 70)
    print("EVENT-LEVEL WARNING CLASSIFICATION")
    print("=" * 70)

    y_test, run_ids, times = load_test_data()

    print("\nTEST TARGETS")
    print("-" * 70)

    print(
        "y_test shape:",
        y_test.shape
    )

    print(
        "Unique test runs:",
        len(np.unique(run_ids))
    )

    # ----------------------------------------------------------
    # IMPORTANT
    # ----------------------------------------------------------
    #
    # At this point we need prediction files from the models.
    #
    # The files should contain:
    #
    #    [samples, 3]
    #
    # matching:
    #
    #    y_test[:, 0] -> 0.5s
    #    y_test[:, 1] -> 1.0s
    #    y_test[:, 2] -> 2.0s
    #
    # ----------------------------------------------------------

    prediction_dir = (
        DATA_DIR
        / "predictions"
    )

    prediction_files = {
        "XGBoost":
            prediction_dir
            / "xgboost_test_predictions.npy",

        "LSTM":
            prediction_dir
            / "lstm_test_predictions.npy",

        "GRU":
            prediction_dir
            / "gru_test_predictions.npy",
    }

    all_results = []

    for model_name, path in (
        prediction_files.items()
    ):

        if not path.exists():

            print(
                f"\nPrediction file missing "
                f"for {model_name}:"
            )

            print(path)

            continue

        y_pred = np.load(path)

        print(
            f"\n{model_name}"
        )

        print(
            "Prediction shape:",
            y_pred.shape
        )

        results = evaluate_predictions(
            y_test,
            y_pred,
            model_name,
        )

        print(
            results.to_string(
                index=False
            )
        )

        all_results.append(
            results
        )

    if not all_results:

        raise FileNotFoundError(
            "No prediction files were found."
        )

    results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        TABLE_DIR
        / "event_classification_metrics.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved:",
        output_path
    )

    print("\n" + "=" * 70)
    print("EVENT CLASSIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()