# ================================================================
# EXTENDED FINAL EVALUATION
# Vehicle Stability Early Warning
#
# Adds:
#   Forecasting:
#       MAE, MSE, RMSE, R2
#
#   Classification:
#       Accuracy, Precision, Recall, F1
#       Specificity, FPR, NPV
#       Balanced Accuracy, MCC
#       AUROC, AUPRC
#       Brier Score, ECE
#
#   Event-level:
#       Detection rate
#       Miss rate
#       Lead-time statistics
#       Warning episodes
#       False-warning episodes
#
#   Deployment:
#       Parameter count
#       Model size
#       Inference latency
#       P95 latency
#       Predictions/sec
#
# IMPORTANT:
# This script uses the EXISTING test predictions and the
# validation-calibrated thresholds already produced by the project.
#
# ================================================================

import os
import time
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore")


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = r"D:\files\vehicle stability\vehicle-stability-early-warning"

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

PRED_DIR = os.path.join(
    DATA_DIR,
    "predictions"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "models"
)

TABLE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "tables"
)

FIG_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures"
)

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ================================================================
# FILES
# ================================================================

TEST_SEQUENCE_FILE = os.path.join(
    DATA_DIR,
    "test_sequences.npz"
)

THRESHOLD_FILE = os.path.join(
    DATA_DIR,
    "selected_warning_thresholds.npz"
)

THRESHOLD_CSV = os.path.join(
    TABLE_DIR,
    "selected_warning_thresholds.csv"
)

PERSISTENCE_METRICS = os.path.join(
    TABLE_DIR,
    "persistence_metrics.csv"
)

XGB_TEST_METRICS = os.path.join(
    TABLE_DIR,
    "xgboost_test_metrics.csv"
)

LSTM_TEST_METRICS = os.path.join(
    TABLE_DIR,
    "lstm_test_metrics.csv"
)

GRU_TEST_METRICS = os.path.join(
    TABLE_DIR,
    "gru_test_metrics.csv"
)


# ================================================================
# MODELS
# ================================================================

MODELS = {
    "XGBoost": {
        "prediction_file": os.path.join(
            PRED_DIR,
            "xgboost_test_predictions.npy"
        ),
        "model_files": [
            os.path.join(MODEL_DIR, "xgboost_0.5s.json"),
            os.path.join(MODEL_DIR, "xgboost_1.0s.json"),
            os.path.join(MODEL_DIR, "xgboost_2.0s.json"),
        ],
    },

    "LSTM": {
        "prediction_file": os.path.join(
            PRED_DIR,
            "lstm_test_predictions.npy"
        ),
        "model_files": [
            os.path.join(MODEL_DIR, "lstm_forecaster.keras"),
            os.path.join(MODEL_DIR, "lstm_best.keras"),
        ],
    },

    "GRU": {
        "prediction_file": os.path.join(
            PRED_DIR,
            "gru_test_predictions.npy"
        ),
        "model_files": [
            os.path.join(MODEL_DIR, "gru_forecaster.keras"),
            os.path.join(MODEL_DIR, "gru_best.keras"),
        ],
    },
}


HORIZONS = ["0.5s", "1.0s", "2.0s"]


# ================================================================
# IMPORTANT CONFIGURATION
# ================================================================

# Your warning threshold calibration treats the prediction score
# as a stability-margin-like quantity where LOWER values indicate
# greater instability risk.
#
# Therefore:
#
#   warning = prediction <= calibrated_threshold
#
# and for AUROC/AUPRC:
#
#   risk_score = -prediction
#
# If your target definition is the opposite, change this to:
#
#   EVENT_DIRECTION = "positive"
#
EVENT_DIRECTION = "negative"


# ECE bins
ECE_BINS = 10


# Number of repeated inference runs for latency measurement.
LATENCY_REPEATS = 100


# ================================================================
# UTILITY
# ================================================================

def print_header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# ================================================================
# LOAD TEST DATA
# ================================================================

def load_test_data():

    print_header("LOADING TEST SEQUENCES")

    if not os.path.exists(TEST_SEQUENCE_FILE):
        raise FileNotFoundError(
            f"Test sequence file not found:\n{TEST_SEQUENCE_FILE}"
        )

    data = np.load(
        TEST_SEQUENCE_FILE,
        allow_pickle=True
    )

    print("Available keys:")

    for key in data.files:
        print(
            f"  {key}: shape={data[key].shape}"
        )

    if "X" not in data:
        raise KeyError("X not found in test_sequences.npz")

    if "y" not in data:
        raise KeyError("y not found in test_sequences.npz")

    X_test = data["X"]
    y_test = data["y"]

    if "Run_ID" not in data:
        raise KeyError(
            "Run_ID not found in test_sequences.npz"
        )

    if "Time" not in data:
        raise KeyError(
            "Time not found in test_sequences.npz"
        )

    run_id = data["Run_ID"]
    times = data["Time"]

    print()
    print(f"X_test : {X_test.shape}")
    print(f"y_test : {y_test.shape}")
    print(f"Run_ID : {run_id.shape}")
    print(f"Time   : {times.shape}")

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test have different lengths."
        )

    if len(run_id) != len(X_test):
        raise ValueError(
            "Run_ID length does not match test samples."
        )

    if len(times) != len(X_test):
        raise ValueError(
            "Time length does not match test samples."
        )

    return X_test, y_test, run_id, times


# ================================================================
# LOAD PREDICTIONS
# ================================================================

def load_predictions():

    print_header("LOADING TEST PREDICTIONS")

    predictions = {}

    for model_name, info in MODELS.items():

        path = info["prediction_file"]

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{model_name} prediction file not found:\n{path}"
            )

        pred = np.load(path)

        print(f"{model_name}:")
        print(f"  File : {path}")
        print(f"  Shape: {pred.shape}")

        if pred.ndim != 2:
            raise ValueError(
                f"{model_name} prediction array must be 2-D."
            )

        if pred.shape[1] != 3:
            raise ValueError(
                f"{model_name} predictions should have 3 horizons."
            )

        predictions[model_name] = pred

    return predictions


# ================================================================
# LOAD THRESHOLDS
# ================================================================

def load_thresholds():

    print_header(
        "LOADING VALIDATION-CALIBRATED THRESHOLDS"
    )

    if not os.path.exists(THRESHOLD_FILE):
        raise FileNotFoundError(
            f"Threshold file not found:\n{THRESHOLD_FILE}"
        )

    data = np.load(
        THRESHOLD_FILE,
        allow_pickle=True
    )

    thresholds = {}

    for model in MODELS:

        thresholds[model] = {}

        for horizon in HORIZONS:

            key = f"{model}_{horizon}"

            if key not in data:
                raise KeyError(
                    f"Missing threshold: {key}"
                )

            thresholds[model][horizon] = float(
                data[key]
            )

    print("CALIBRATED THRESHOLDS")
    print("-" * 70)

    for model in MODELS:

        print(
            f"{model:<10}: "
            + ", ".join(
                f"{h}={thresholds[model][h]:.2f}"
                for h in HORIZONS
            )
        )

    return thresholds


# ================================================================
# EVENT LABEL
# ================================================================

def create_binary_labels(y):

    """
    Converts the continuous target into an instability label.

    Existing project convention:
        negative target -> unstable
        non-negative -> stable

    If the project's target convention is opposite,
    change EVENT_DIRECTION above.
    """

    if EVENT_DIRECTION == "negative":

        labels = (
            y < 0
        ).astype(int)

    elif EVENT_DIRECTION == "positive":

        labels = (
            y > 0
        ).astype(int)

    else:
        raise ValueError(
            "EVENT_DIRECTION must be 'negative' or 'positive'"
        )

    return labels


# ================================================================
# WARNING PREDICTIONS
# ================================================================

def create_warning_labels(
    predictions,
    threshold
):

    if EVENT_DIRECTION == "negative":

        warnings_binary = (
            predictions <= threshold
        ).astype(int)

    else:

        warnings_binary = (
            predictions >= threshold
        ).astype(int)

    return warnings_binary


# ================================================================
# ECE
# ================================================================

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10
):

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:

            mask = (
                (probabilities >= bin_edges[i])
                &
                (probabilities <= bin_edges[i + 1])
            )

        else:

            mask = (
                (probabilities >= bin_edges[i])
                &
                (probabilities < bin_edges[i + 1])
            )

        if not np.any(mask):
            continue

        confidence = np.mean(
            probabilities[mask]
        )

        accuracy = np.mean(
            y_true[mask]
        )

        fraction = (
            np.sum(mask)
            /
            len(y_true)
        )

        ece += (
            fraction
            *
            abs(
                accuracy
                -
                confidence
            )
        )

    return float(ece)


# ================================================================
# CONFUSION METRICS
# ================================================================

def classification_metrics(
    y_true,
    y_pred,
    risk_score
):

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total > 0 else np.nan
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else np.nan
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0 else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0 else np.nan
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    try:
        mcc = matthews_corrcoef(
            y_true,
            y_pred
        )
    except Exception:
        mcc = np.nan

    # AUROC/AUPRC require both classes
    if len(np.unique(y_true)) == 2:

        auroc = roc_auc_score(
            y_true,
            risk_score
        )

        auprc = average_precision_score(
            y_true,
            risk_score
        )

    else:

        auroc = np.nan
        auprc = np.nan

    # Convert risk score into a probability-like value.
    #
    # This is NOT used for thresholding.
    # It is only a monotonic normalization for calibration metrics.
    #
    # Sigmoid transform.
    centered = risk_score - np.median(risk_score)

    scale = np.std(centered)

    if scale < 1e-12:
        probabilities = np.full(
            len(centered),
            0.5
        )
    else:
        probabilities = 1.0 / (
            1.0
            +
            np.exp(
                -np.clip(
                    centered / scale,
                    -30,
                    30
                )
            )
        )

    try:

        brier = brier_score_loss(
            y_true,
            probabilities
        )

    except Exception:

        brier = np.nan

    ece = expected_calibration_error(
        y_true,
        probabilities,
        ECE_BINS
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "npv": npv,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
        "ece": ece,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


# ================================================================
# FORECASTING METRICS
# ================================================================

def forecasting_metrics(
    y_true,
    prediction
):

    mae = mean_absolute_error(
        y_true,
        prediction
    )

    mse = mean_squared_error(
        y_true,
        prediction
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        prediction
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# ================================================================
# MODEL SIZE
# ================================================================

def get_model_size(model_name):

    paths = MODELS[model_name]["model_files"]

    existing = [
        p for p in paths
        if os.path.exists(p)
    ]

    if not existing:
        return np.nan

    # For XGBoost use all horizon models.
    # For neural networks use the main forecaster.
    if model_name == "XGBoost":

        total_bytes = sum(
            os.path.getsize(p)
            for p in existing
        )

    else:

        # Prefer forecaster rather than best checkpoint
        preferred = [
            p for p in existing
            if "forecaster" in os.path.basename(p)
        ]

        if preferred:

            total_bytes = os.path.getsize(
                preferred[0]
            )

        else:

            total_bytes = os.path.getsize(
                existing[0]
            )

    return total_bytes / (1024 ** 2)


# ================================================================
# PARAMETER COUNT
# ================================================================

def get_parameter_count(model_name):

    if model_name == "XGBoost":

        try:

            import xgboost as xgb

            total = 0

            for h in HORIZONS:

                path = os.path.join(
                    MODEL_DIR,
                    f"xgboost_{h}.json"
                )

                if not os.path.exists(path):
                    continue

                model = xgb.XGBRegressor()

                model.load_model(path)

                booster = model.get_booster()

                # Approximate number of tree nodes
                trees = booster.get_dump()

                total += sum(
                    tree.count("\n")
                    for tree in trees
                )

            return float(total)

        except Exception:

            return np.nan

    if model_name in ["LSTM", "GRU"]:

        try:

            import tensorflow as tf

            candidates = [
                os.path.join(
                    MODEL_DIR,
                    f"{model_name.lower()}_forecaster.keras"
                ),
                os.path.join(
                    MODEL_DIR,
                    f"{model_name.lower()}_best.keras"
                )
            ]

            for path in candidates:

                if os.path.exists(path):

                    model = tf.keras.models.load_model(
                        path,
                        compile=False
                    )

                    return int(
                        model.count_params()
                    )

        except Exception:

            return np.nan

    return np.nan


# ================================================================
# INFERENCE LATENCY
# ================================================================

def measure_neural_latency(
    model_name,
    X_test
):

    try:

        import tensorflow as tf

        candidates = [
            os.path.join(
                MODEL_DIR,
                f"{model_name.lower()}_forecaster.keras"
            ),
            os.path.join(
                MODEL_DIR,
                f"{model_name.lower()}_best.keras"
            )
        ]

        model_path = None

        for p in candidates:

            if os.path.exists(p):

                model_path = p
                break

        if model_path is None:
            return np.nan, np.nan, np.nan

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        # Small batch for latency measurement
        n = min(
            32,
            len(X_test)
        )

        sample = X_test[:n].astype(
            np.float32
        )

        # Warm-up
        model.predict(
            sample,
            verbose=0
        )

        times = []

        for _ in range(
            LATENCY_REPEATS
        ):

            start = time.perf_counter()

            model.predict(
                sample,
                verbose=0
            )

            end = time.perf_counter()

            times.append(
                end - start
            )

        times = np.asarray(times)

        mean_batch = np.mean(times)

        p95_batch = np.percentile(
            times,
            95
        )

        mean_per_sample = (
            mean_batch / n
        )

        return (
            mean_per_sample * 1000.0,
            p95_batch / n * 1000.0,
            1.0 / mean_per_sample
        )

    except Exception as e:

        print(
            f"Latency measurement failed for "
            f"{model_name}: {e}"
        )

        return (
            np.nan,
            np.nan,
            np.nan
        )


def measure_xgboost_latency(
    X_test
):

    try:

        import xgboost as xgb

        X_flat = X_test.reshape(
            len(X_test),
            -1
        )

        model_path = os.path.join(
            MODEL_DIR,
            "xgboost_1.0s.json"
        )

        if not os.path.exists(model_path):
            return np.nan, np.nan, np.nan

        model = xgb.XGBRegressor()

        model.load_model(
            model_path
        )

        n = min(
            32,
            len(X_flat)
        )

        sample = X_flat[:n]

        # Warm-up
        model.predict(
            sample
        )

        times = []

        for _ in range(
            LATENCY_REPEATS
        ):

            start = time.perf_counter()

            model.predict(
                sample
            )

            end = time.perf_counter()

            times.append(
                end - start
            )

        times = np.asarray(times)

        mean_batch = np.mean(times)

        p95_batch = np.percentile(
            times,
            95
        )

        mean_per_sample = (
            mean_batch / n
        )

        return (
            mean_per_sample * 1000.0,
            p95_batch / n * 1000.0,
            1.0 / mean_per_sample
        )

    except Exception as e:

        print(
            f"XGBoost latency measurement failed: {e}"
        )

        return (
            np.nan,
            np.nan,
            np.nan
        )


# ================================================================
# EVENT EPISODES
# ================================================================

def count_warning_episodes(
    warnings_binary,
    run_ids
):

    total_episodes = 0
    false_episodes = 0

    for run in np.unique(run_ids):

        idx = np.where(
            run_ids == run
        )[0]

        values = warnings_binary[idx]

        if len(values) == 0:
            continue

        # Find contiguous warning episodes
        in_episode = False

        for i, value in enumerate(values):

            if value == 1 and not in_episode:

                total_episodes += 1

                in_episode = True

            elif value == 0:

                in_episode = False

    return total_episodes


# ================================================================
# EVENT DETECTION
# ================================================================

def event_level_metrics(
    y_true,
    warning_binary,
    run_ids,
    times
):

    """
    Event-level calculation.

    An event is a contiguous positive region in the ground-truth
    instability label within a run.

    An event is considered detected when at least one warning occurs
    before or at event onset.

    Lead time is measured from the FIRST warning associated with
    the event to event onset.

    """

    labels = create_binary_labels(
        y_true
    )

    events = []

    unique_runs = np.unique(
        run_ids
    )

    for run in unique_runs:

        idx = np.where(
            run_ids == run
        )[0]

        if len(idx) == 0:
            continue

        run_labels = labels[idx]
        run_times = times[idx]

        inside = False
        start = None

        for i in range(
            len(run_labels)
        ):

            if (
                run_labels[i] == 1
                and not inside
            ):

                start = i
                inside = True

            elif (
                run_labels[i] == 0
                and inside
            ):

                end = i - 1

                events.append(
                    {
                        "run": run,
                        "start_idx": idx[start],
                        "end_idx": idx[end],
                        "event_time": run_times[start]
                    }
                )

                inside = False

        if inside:

            end = len(run_labels) - 1

            events.append(
                {
                    "run": run,
                    "start_idx": idx[start],
                    "end_idx": idx[end],
                    "event_time": run_times[start]
                }
            )

    lead_times = []
    detected = 0

    for event in events:

        run = event["run"]

        event_time = event["event_time"]

        run_mask = (
            run_ids == run
        )

        warning_indices = np.where(
            run_mask
            &
            (warning_binary == 1)
        )[0]

        # Warnings before or at onset
        valid = warning_indices[
            times[warning_indices]
            <= event_time
        ]

        if len(valid) > 0:

            # Last warning before event onset gives
            # the actual warning lead immediately preceding event.
            warning_time = times[
                valid[-1]
            ]

            lead = (
                event_time
                -
                warning_time
            )

            lead_times.append(
                max(
                    0.0,
                    float(lead)
                )
            )

            detected += 1

    total_events = len(events)

    if total_events == 0:

        return {
            "total_events": 0,
            "detected_events": 0,
            "missed_events": 0,
            "event_detection_rate": np.nan,
            "event_miss_rate": np.nan,
            "mean_lead_time_s": np.nan,
            "median_lead_time_s": np.nan,
            "std_lead_time_s": np.nan,
            "lead_time_q25_s": np.nan,
            "lead_time_q75_s": np.nan,
        }

    missed = (
        total_events
        -
        detected
    )

    if len(lead_times) > 0:

        lead = np.asarray(
            lead_times
        )

        mean_lead = np.mean(lead)
        median_lead = np.median(lead)
        std_lead = np.std(lead)
        q25 = np.percentile(
            lead,
            25
        )
        q75 = np.percentile(
            lead,
            75
        )

    else:

        mean_lead = np.nan
        median_lead = np.nan
        std_lead = np.nan
        q25 = np.nan
        q75 = np.nan

    return {
        "total_events": total_events,
        "detected_events": detected,
        "missed_events": missed,
        "event_detection_rate":
            detected / total_events,
        "event_miss_rate":
            missed / total_events,
        "mean_lead_time_s": mean_lead,
        "median_lead_time_s": median_lead,
        "std_lead_time_s": std_lead,
        "lead_time_q25_s": q25,
        "lead_time_q75_s": q75,
    }


# ================================================================
# MAIN EVALUATION
# ================================================================

def main():

    print_header(
        "EXTENDED FINAL VEHICLE STABILITY EVALUATION"
    )

    X_test, y_test, run_ids, times = (
        load_test_data()
    )

    predictions = load_predictions()

    thresholds = load_thresholds()

    all_rows = []

    print_header(
        "RUNNING EXTENDED METRICS"
    )

    for model_name in MODELS:

        print()
        print(
            "=" * 70
        )
        print(
            model_name
        )
        print(
            "=" * 70
        )

        pred = predictions[
            model_name
        ]

        for h_idx, horizon in enumerate(
            HORIZONS
        ):

            print()
            print(
                f"Evaluating {horizon}..."
            )

            y_true = y_test[
                :, h_idx
            ]

            y_pred = pred[
                :, h_idx
            ]

            threshold = thresholds[
                model_name
            ][
                horizon
            ]

            # ----------------------------------------------------
            # Forecasting
            # ----------------------------------------------------

            forecast = forecasting_metrics(
                y_true,
                y_pred
            )

            # ----------------------------------------------------
            # Binary labels
            # ----------------------------------------------------

            y_binary = create_binary_labels(
                y_true
            )

            warning_binary = create_warning_labels(
                y_pred,
                threshold
            )

            # ----------------------------------------------------
            # Risk score
            # ----------------------------------------------------

            if EVENT_DIRECTION == "negative":

                risk_score = -y_pred

            else:

                risk_score = y_pred

            # ----------------------------------------------------
            # Classification
            # ----------------------------------------------------

            cls = classification_metrics(
                y_binary,
                warning_binary,
                risk_score
            )

            # ----------------------------------------------------
            # Event-level
            # ----------------------------------------------------

            event = event_level_metrics(
                y_true,
                warning_binary,
                run_ids,
                times
            )

            # ----------------------------------------------------
            # Warning episodes
            # ----------------------------------------------------

            total_warning_episodes = (
                count_warning_episodes(
                    warning_binary,
                    run_ids
                )
            )

            # ----------------------------------------------------
            # False warning episodes
            # ----------------------------------------------------

            # A warning episode is false if it does not overlap
            # any ground-truth instability sample.
            false_warning_episodes = 0

            for run in np.unique(
                run_ids
            ):

                idx = np.where(
                    run_ids == run
                )[0]

                labels_run = y_binary[idx]
                warnings_run = (
                    warning_binary[idx]
                )

                in_warning = False
                episode_has_event = False

                for i in range(
                    len(idx)
                ):

                    if (
                        warnings_run[i] == 1
                        and not in_warning
                    ):

                        in_warning = True
                        episode_has_event = (
                            labels_run[i] == 1
                        )

                    elif (
                        warnings_run[i] == 1
                        and in_warning
                    ):

                        if labels_run[i] == 1:
                            episode_has_event = True

                    elif (
                        warnings_run[i] == 0
                        and in_warning
                    ):

                        if not episode_has_event:
                            false_warning_episodes += 1

                        in_warning = False
                        episode_has_event = False

                if in_warning:

                    if not episode_has_event:
                        false_warning_episodes += 1

            # ----------------------------------------------------
            # Store
            # ----------------------------------------------------

            row = {

                "model":
                    model_name,

                "horizon":
                    horizon,

                "threshold":
                    threshold,

                # Forecast
                "MAE":
                    forecast["MAE"],

                "MSE":
                    forecast["MSE"],

                "RMSE":
                    forecast["RMSE"],

                "R2":
                    forecast["R2"],

                # Classification
                "accuracy":
                    cls["accuracy"],

                "precision":
                    cls["precision"],

                "recall":
                    cls["recall"],

                "F1":
                    cls["f1"],

                "specificity":
                    cls["specificity"],

                "false_positive_rate":
                    cls["false_positive_rate"],

                "NPV":
                    cls["npv"],

                "balanced_accuracy":
                    cls["balanced_accuracy"],

                "MCC":
                    cls["mcc"],

                "AUROC":
                    cls["auroc"],

                "AUPRC":
                    cls["auprc"],

                "Brier_score":
                    cls["brier_score"],

                "ECE":
                    cls["ece"],

                "true_positives":
                    cls["true_positives"],

                "false_positives":
                    cls["false_positives"],

                "true_negatives":
                    cls["true_negatives"],

                "false_negatives":
                    cls["false_negatives"],

                # Event
                "total_events":
                    event["total_events"],

                "detected_events":
                    event["detected_events"],

                "missed_events":
                    event["missed_events"],

                "event_detection_rate":
                    event[
                        "event_detection_rate"
                    ],

                "event_miss_rate":
                    event[
                        "event_miss_rate"
                    ],

                "mean_lead_time_s":
                    event[
                        "mean_lead_time_s"
                    ],

                "median_lead_time_s":
                    event[
                        "median_lead_time_s"
                    ],

                "std_lead_time_s":
                    event[
                        "std_lead_time_s"
                    ],

                "lead_time_q25_s":
                    event[
                        "lead_time_q25_s"
                    ],

                "lead_time_q75_s":
                    event[
                        "lead_time_q75_s"
                    ],

                # Warning burden
                "total_warning_episodes":
                    total_warning_episodes,

                "false_warning_episodes":
                    false_warning_episodes,

                # Base rate
                "event_base_rate":
                    np.mean(y_binary),
            }

            all_rows.append(
                row
            )

            print(
                f"  MAE       : {forecast['MAE']:.6f}"
            )

            print(
                f"  RMSE      : {forecast['RMSE']:.6f}"
            )

            print(
                f"  R2        : {forecast['R2']:.6f}"
            )

            print(
                f"  Accuracy  : {cls['accuracy']:.6f}"
            )

            print(
                f"  Precision : {cls['precision']:.6f}"
            )

            print(
                f"  Recall    : {cls['recall']:.6f}"
            )

            print(
                f"  F1        : {cls['f1']:.6f}"
            )

            print(
                f"  AUROC     : {cls['auroc']:.6f}"
            )

            print(
                f"  AUPRC     : {cls['auprc']:.6f}"
            )

            print(
                f"  FPR       : {cls['false_positive_rate']:.6f}"
            )

            print(
                f"  Event DR  : "
                f"{event['event_detection_rate']:.6f}"
            )

            print(
                f"  Mean lead : "
                f"{event['mean_lead_time_s']:.6f} s"
            )

    # ============================================================
    # SAVE EXTENDED RESULTS
    # ============================================================

    df = pd.DataFrame(
        all_rows
    )

    output_file = os.path.join(
        TABLE_DIR,
        "extended_final_model_comparison.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print_header(
        "EXTENDED RESULTS SAVED"
    )

    print(output_file)

    # ============================================================
    # DEPLOYMENT METRICS
    # ============================================================

    print_header(
        "DEPLOYMENT / COMPUTATIONAL METRICS"
    )

    deployment_rows = []

    for model_name in MODELS:

        print()
        print(
            f"Measuring {model_name}..."
        )

        size_mb = get_model_size(
            model_name
        )

        params = get_parameter_count(
            model_name
        )

        if model_name == "XGBoost":

            latency_ms, p95_ms, throughput = (
                measure_xgboost_latency(
                    X_test
                )
            )

        else:

            latency_ms, p95_ms, throughput = (
                measure_neural_latency(
                    model_name,
                    X_test
                )
            )

        deployment_rows.append(
            {
                "model":
                    model_name,

                "parameter_or_tree_count":
                    params,

                "model_size_MB":
                    size_mb,

                "mean_inference_latency_ms_per_sample":
                    latency_ms,

                "p95_inference_latency_ms_per_sample":
                    p95_ms,

                "approx_predictions_per_second":
                    throughput,
            }
        )

        print(
            f"  Parameter/tree count : {params}"
        )

        print(
            f"  Model size           : "
            f"{size_mb:.4f} MB"
        )

        print(
            f"  Mean latency         : "
            f"{latency_ms:.6f} ms/sample"
        )

        print(
            f"  P95 latency          : "
            f"{p95_ms:.6f} ms/sample"
        )

        print(
            f"  Throughput           : "
            f"{throughput:.2f} samples/s"
        )

    deployment_df = pd.DataFrame(
        deployment_rows
    )

    deployment_file = os.path.join(
        TABLE_DIR,
        "deployment_efficiency_comparison.csv"
    )

    deployment_df.to_csv(
        deployment_file,
        index=False
    )

    print()
    print(
        f"Saved:\n{deployment_file}"
    )

    # ============================================================
    # METRIC-SPECIFIC SUMMARY
    # ============================================================

    print_header(
        "METRIC-SPECIFIC SUMMARY"
    )

    summary_rows = []

    for horizon in HORIZONS:

        subset = df[
            df["horizon"] == horizon
        ]

        if len(subset) == 0:
            continue

        summary_rows.append(
            {
                "horizon":
                    horizon,

                "lowest_MAE_model":
                    subset.loc[
                        subset["MAE"].idxmin(),
                        "model"
                    ],

                "lowest_MAE":
                    subset["MAE"].min(),

                "lowest_RMSE_model":
                    subset.loc[
                        subset["RMSE"].idxmin(),
                        "model"
                    ],

                "lowest_RMSE":
                    subset["RMSE"].min(),

                "highest_R2_model":
                    subset.loc[
                        subset["R2"].idxmax(),
                        "model"
                    ],

                "highest_R2":
                    subset["R2"].max(),

                "highest_F1_model":
                    subset.loc[
                        subset["F1"].idxmax(),
                        "model"
                    ],

                "highest_F1":
                    subset["F1"].max(),

                "highest_AUROC_model":
                    subset.loc[
                        subset["AUROC"].idxmax(),
                        "model"
                    ],

                "highest_AUROC":
                    subset["AUROC"].max(),

                "highest_AUPRC_model":
                    subset.loc[
                        subset["AUPRC"].idxmax(),
                        "model"
                    ],

                "highest_AUPRC":
                    subset["AUPRC"].max(),

                "highest_detection_model":
                    subset.loc[
                        subset[
                            "event_detection_rate"
                        ].idxmax(),
                        "model"
                    ],

                "highest_detection_rate":
                    subset[
                        "event_detection_rate"
                    ].max(),

                "lowest_FPR_model":
                    subset.loc[
                        subset[
                            "false_positive_rate"
                        ].idxmin(),
                        "model"
                    ],

                "lowest_FPR":
                    subset[
                        "false_positive_rate"
                    ].min(),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = os.path.join(
        TABLE_DIR,
        "extended_metric_specific_comparisons.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved:\n{summary_file}"
    )

    # ============================================================
    # FINAL TEXT REPORT
    # ============================================================

    report_file = os.path.join(
        TABLE_DIR,
        "extended_final_evaluation_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "EXTENDED FINAL VEHICLE STABILITY "
            "EARLY-WARNING EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "Forecasting + Classification + "
            "Event-Level + Deployment Metrics\n\n"
        )

        f.write(
            df.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
            + "=" * 70
            + "\n"
        )

        f.write(
            "DEPLOYMENT METRICS\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            deployment_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\n"
            + "=" * 70
            + "\n"
        )

        f.write(
            "METRIC-SPECIFIC SUMMARY\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            summary_df.to_string(
                index=False
            )
        )

    print(
        f"\nSaved report:\n{report_file}"
    )

    # ============================================================
    # FIGURES
    # ============================================================

    try:

        import matplotlib.pyplot as plt

        # --------------------------------------------------------
        # F1
        # --------------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        for model in MODELS:

            sub = df[
                df["model"] == model
            ]

            plt.plot(
                sub["horizon"],
                sub["F1"],
                marker="o",
                label=model
            )

        plt.xlabel(
            "Forecast Horizon"
        )

        plt.ylabel(
            "F1 Score"
        )

        plt.title(
            "F1 Score vs Forecast Horizon"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIG_DIR,
                "classification_f1_vs_horizon.png"
            ),
            dpi=300
        )

        plt.close()

        # --------------------------------------------------------
        # AUROC
        # --------------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        for model in MODELS:

            sub = df[
                df["model"] == model
            ]

            plt.plot(
                sub["horizon"],
                sub["AUROC"],
                marker="o",
                label=model
            )

        plt.xlabel(
            "Forecast Horizon"
        )

        plt.ylabel(
            "AUROC"
        )

        plt.title(
            "AUROC vs Forecast Horizon"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIG_DIR,
                "classification_auroc_vs_horizon.png"
            ),
            dpi=300
        )

        plt.close()

        # --------------------------------------------------------
        # AUPRC
        # --------------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        for model in MODELS:

            sub = df[
                df["model"] == model
            ]

            plt.plot(
                sub["horizon"],
                sub["AUPRC"],
                marker="o",
                label=model
            )

        plt.xlabel(
            "Forecast Horizon"
        )

        plt.ylabel(
            "AUPRC"
        )

        plt.title(
            "AUPRC vs Forecast Horizon"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIG_DIR,
                "classification_auprc_vs_horizon.png"
            ),
            dpi=300
        )

        plt.close()

        # --------------------------------------------------------
        # FPR
        # --------------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        for model in MODELS:

            sub = df[
                df["model"] == model
            ]

            plt.plot(
                sub["horizon"],
                sub["false_positive_rate"],
                marker="o",
                label=model
            )

        plt.xlabel(
            "Forecast Horizon"
        )

        plt.ylabel(
            "False Positive Rate"
        )

        plt.title(
            "False Positive Rate vs Forecast Horizon"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIG_DIR,
                "classification_fpr_vs_horizon.png"
            ),
            dpi=300
        )

        plt.close()

        # --------------------------------------------------------
        # ECE
        # --------------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        for model in MODELS:

            sub = df[
                df["model"] == model
            ]

            plt.plot(
                sub["horizon"],
                sub["ECE"],
                marker="o",
                label=model
            )

        plt.xlabel(
            "Forecast Horizon"
        )

        plt.ylabel(
            "ECE"
        )

        plt.title(
            "Expected Calibration Error vs Forecast Horizon"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIG_DIR,
                "classification_ece_vs_horizon.png"
            ),
            dpi=300
        )

        plt.close()

        # --------------------------------------------------------
        # Detection
        # --------------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        for model in MODELS:

            sub = df[
                df["model"] == model
            ]

            plt.plot(
                sub["horizon"],
                sub["event_detection_rate"],
                marker="o",
                label=model
            )

        plt.xlabel(
            "Forecast Horizon"
        )

        plt.ylabel(
            "Event Detection Rate"
        )

        plt.title(
            "Event Detection Rate vs Forecast Horizon"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIG_DIR,
                "extended_event_detection_rate_vs_horizon.png"
            ),
            dpi=300
        )

        plt.close()

        print_header(
            "FIGURES SAVED"
        )

        for filename in [
            "classification_f1_vs_horizon.png",
            "classification_auroc_vs_horizon.png",
            "classification_auprc_vs_horizon.png",
            "classification_fpr_vs_horizon.png",
            "classification_ece_vs_horizon.png",
            "extended_event_detection_rate_vs_horizon.png",
        ]:

            print(
                os.path.join(
                    FIG_DIR,
                    filename
                )
            )

    except Exception as e:

        print(
            "\nWARNING: Figure generation failed:"
        )

        print(e)

    # ============================================================
    # COMPLETE
    # ============================================================

    print_header(
        "EXTENDED FINAL EVALUATION COMPLETE"
    )

    print(
        "\nMain table:"
    )

    print(
        output_file
    )

    print(
        "\nDeployment table:"
    )

    print(
        deployment_file
    )

    print(
        "\nMetric-specific comparison:"
    )

    print(
        summary_file
    )

    print(
        "\nReport:"
    )

    print(
        report_file
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()