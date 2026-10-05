# ================================================================
# EXTENDED FINAL EVALUATION
# Physics-informed / ML vehicle stability early-warning system
#
# Models:
#   XGBoost
#   LSTM
#   GRU
#
# Existing outputs used:
#   test_sequences.npz
#   xgboost_test_predictions.npy
#   lstm_test_predictions.npy
#   gru_test_predictions.npy
#   selected_warning_thresholds.npz
#   final_forecasting_comparison.csv
#   final_calibrated_event_lead_time_metrics.csv
#
# New outputs:
#   extended_classification_metrics.csv
#   timely_warning_metrics.csv
#   deployment_metrics.csv
#   bootstrap_confidence_intervals.csv
#   run_level_model_comparison.csv
#   final_paper_table.csv
#
# Figures:
#   roc_curves.png
#   precision_recall_curves.png
#   confusion_matrices.png
#   timely_detection_vs_horizon.png
#   lead_time_distribution.png
#   threshold_sensitivity.png
#   warning_timeline.png
# ================================================================

from pathlib import Path
import warnings
import time
import json
import math

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# sklearn
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

# scipy
try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# matplotlib
import matplotlib.pyplot as plt


# ================================================================
# CONFIGURATION
# ================================================================

ROOT = Path(
    r"D:\files\vehicle stability\vehicle-stability-early-warning"
)

DATA_PROCESSED = ROOT / "data" / "processed"
PRED_DIR = DATA_PROCESSED / "predictions"
RESULTS = ROOT / "results"
TABLE_DIR = RESULTS / "tables"
FIG_DIR = RESULTS / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_FILE = DATA_PROCESSED / "test_sequences.npz"
THRESHOLD_FILE = DATA_PROCESSED / "selected_warning_thresholds.npz"

PREDICTION_FILES = {
    "XGBoost": PRED_DIR / "xgboost_test_predictions.npy",
    "LSTM": PRED_DIR / "lstm_test_predictions.npy",
    "GRU": PRED_DIR / "gru_test_predictions.npy",
}

HORIZONS = ["0.5s", "1.0s", "2.0s"]
HORIZON_SECONDS = {
    "0.5s": 0.5,
    "1.0s": 1.0,
    "2.0s": 2.0,
}

# Timely warning levels.
TIMELY_LEVELS = [0.25, 0.5, 1.0, 2.0]

N_BOOTSTRAP = 2000
RANDOM_SEED = 42

# Warning episode gap.
# A new episode is created after this many consecutive non-warning
# samples.
MAX_WARNING_GAP = 1


# ================================================================
# UTILITY
# ================================================================

rng = np.random.default_rng(RANDOM_SEED)


def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")


def safe_div(a, b):
    return float(a / b) if b != 0 else np.nan


def clean_array(x):
    return np.asarray(x).astype(float)


# ================================================================
# LOAD DATA
# ================================================================

def load_test_sequences():

    banner("LOADING TEST SEQUENCES")

    require_file(SEQUENCE_FILE)

    data = np.load(SEQUENCE_FILE, allow_pickle=True)

    print("Available keys:")
    for k in data.files:
        print(f"  {k}: shape={data[k].shape}")

    X = data["X"]
    y = data["y"]

    if "Run_ID" not in data.files:
        raise KeyError(
            "Run_ID not found in test_sequences.npz. "
            "The final event-level evaluation requires sequence-level Run_ID."
        )

    run_id = data["Run_ID"]

    if "Time" in data.files:
        time_array = data["Time"]
    else:
        time_array = None

    if len(X) != len(y):
        raise ValueError("X and y lengths do not match.")

    if len(run_id) != len(X):
        raise ValueError(
            f"Run_ID length mismatch: {len(run_id)} vs {len(X)}"
        )

    if time_array is not None and len(time_array) != len(X):
        raise ValueError(
            f"Time length mismatch: {len(time_array)} vs {len(X)}"
        )

    print()
    print(f"X_test shape : {X.shape}")
    print(f"y_test shape : {y.shape}")
    print(f"Run_ID shape : {run_id.shape}")

    if time_array is not None:
        print(f"Time shape   : {time_array.shape}")

    print(f"Test samples : {len(X)}")
    print(f"Test runs    : {len(np.unique(run_id))}")

    return X, y, run_id, time_array


# ================================================================
# LOAD PREDICTIONS
# ================================================================

def load_predictions():

    banner("LOADING TEST PREDICTIONS")

    predictions = {}

    for model, path in PREDICTION_FILES.items():

        require_file(path)

        pred = np.load(path)

        print(f"{model}:")
        print(f"  File : {path}")
        print(f"  Shape: {pred.shape}")

        if pred.ndim != 2:
            raise ValueError(
                f"{model} prediction array must be 2-D."
            )

        if pred.shape[1] != 3:
            raise ValueError(
                f"{model} prediction array must have 3 horizons."
            )

        predictions[model] = pred

    return predictions


# ================================================================
# LOAD THRESHOLDS
# ================================================================

def load_thresholds():

    banner("LOADING VALIDATION-CALIBRATED THRESHOLDS")

    require_file(THRESHOLD_FILE)

    data = np.load(THRESHOLD_FILE)

    thresholds = {}

    for model in PREDICTION_FILES:

        thresholds[model] = {}

        for h in HORIZONS:

            key = f"{model}_{h}"

            if key not in data.files:
                raise KeyError(
                    f"Missing calibrated threshold: {key}"
                )

            thresholds[model][h] = float(data[key])

    print("Thresholds:")

    for model in thresholds:
        print(
            f"{model:10s}: "
            + ", ".join(
                f"{h}={thresholds[model][h]:.2f}"
                for h in HORIZONS
            )
        )

    return thresholds


# ================================================================
# GROUND TRUTH
# ================================================================

def create_binary_targets(y):

    """
    Convert continuous future stability target into binary instability.

    Assumption:
        y > 0 means unstable/positive event.

    If your y is already binary this remains unchanged.
    """

    y = np.asarray(y)

    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("y must have shape (N, 3).")

    targets = np.zeros_like(y, dtype=int)

    for j in range(3):
        targets[:, j] = (y[:, j] > 0).astype(int)

    return targets


# ================================================================
# CLASSIFICATION METRICS
# ================================================================

def classification_metrics(y_true, score, threshold):

    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    y_pred = (score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)

    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)

    balanced = balanced_accuracy_score(
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

    try:
        roc_auc = roc_auc_score(
            y_true,
            score
        )
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true,
            score
        )
    except Exception:
        pr_auc = np.nan

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "F1": f1,
        "balanced_accuracy": balanced,
        "NPV": npv,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "MCC": mcc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }


# ================================================================
# WARNING EPISODES
# ================================================================

def find_warning_episodes(flags, run_ids):

    flags = np.asarray(flags).astype(int)
    run_ids = np.asarray(run_ids)

    episodes = []

    start = None
    current_run = None
    gap = 0

    for i in range(len(flags)):

        if current_run is None:
            current_run = run_ids[i]

        if run_ids[i] != current_run:

            if start is not None:
                episodes.append(
                    (current_run, start, i - 1)
                )

            start = None
            gap = 0
            current_run = run_ids[i]

        if flags[i] == 1:

            if start is None:
                start = i

            gap = 0

        else:

            if start is not None:

                gap += 1

                if gap > MAX_WARNING_GAP:

                    episodes.append(
                        (
                            current_run,
                            start,
                            i - gap
                        )
                    )

                    start = None
                    gap = 0

    if start is not None:

        episodes.append(
            (
                current_run,
                start,
                len(flags) - 1
            )
        )

    return episodes


# ================================================================
# EVENT EXTRACTION
# ================================================================

def extract_events(y_binary, run_ids):

    """
    Identify contiguous unstable regions independently for each run.
    """

    events = []

    for run in np.unique(run_ids):

        idx = np.where(run_ids == run)[0]

        if len(idx) == 0:
            continue

        flags = y_binary[idx]

        start = None

        for j, value in enumerate(flags):

            if value == 1 and start is None:
                start = j

            elif value == 0 and start is not None:

                events.append(
                    {
                        "run_id": run,
                        "start_index": int(idx[start]),
                        "end_index": int(idx[j - 1]),
                    }
                )

                start = None

        if start is not None:

            events.append(
                {
                    "run_id": run,
                    "start_index": int(idx[start]),
                    "end_index": int(idx[-1]),
                }
            )

    return events


# ================================================================
# EVENT-LEVEL WARNING EVALUATION
# ================================================================

def event_warning_metrics(
    y_binary,
    score,
    threshold,
    run_ids,
    time_array,
    min_lead=0.0
):

    flags = (score >= threshold).astype(int)

    events = extract_events(
        y_binary,
        run_ids
    )

    episodes = find_warning_episodes(
        flags,
        run_ids
    )

    detected = 0
    lead_times = []

    event_details = []

    for event_id, event in enumerate(events):

        run = event["run_id"]
        event_start = event["start_index"]

        candidate_episodes = [
            ep for ep in episodes
            if ep[0] == run
            and ep[2] <= event_start
        ]

        if len(candidate_episodes) == 0:
            event_details.append({
                "event_id": event_id,
                "run_id": run,
                "event_start_index": event_start,
                "detected": 0,
                "lead_time_s": np.nan
            })
            continue

        # Most recent warning episode before event.
        ep = max(
            candidate_episodes,
            key=lambda x: x[2]
        )

        warning_idx = ep[1]

        if time_array is not None:

            event_time = float(
                time_array[event_start]
            )

            warning_time = float(
                time_array[warning_idx]
            )

            lead = event_time - warning_time

        else:

            # Existing project sampling interval.
            lead = (
                event_start - warning_idx
            ) * 0.05

        if lead >= min_lead:

            detected += 1
            lead_times.append(float(lead))

            event_details.append({
                "event_id": event_id,
                "run_id": run,
                "event_start_index": event_start,
                "warning_index": warning_idx,
                "detected": 1,
                "lead_time_s": float(lead)
            })

        else:

            event_details.append({
                "event_id": event_id,
                "run_id": run,
                "event_start_index": event_start,
                "warning_index": warning_idx,
                "detected": 0,
                "lead_time_s": np.nan
            })

    total_events = len(events)
    missed = total_events - detected

    false_warning_episodes = 0

    for ep in episodes:

        run = ep[0]
        warning_end = ep[2]

        overlaps_event = any(
            e["run_id"] == run
            and e["start_index"] <= warning_end
            for e in events
        )

        if not overlaps_event:
            false_warning_episodes += 1

    if len(lead_times) > 0:

        lead = np.asarray(
            lead_times,
            dtype=float
        )

        mean_lead = float(np.mean(lead))
        median_lead = float(np.median(lead))
        std_lead = float(np.std(lead))
        q25 = float(np.percentile(lead, 25))
        q75 = float(np.percentile(lead, 75))
        min_lead_time = float(np.min(lead))
        max_lead_time = float(np.max(lead))

    else:

        mean_lead = np.nan
        median_lead = np.nan
        std_lead = np.nan
        q25 = np.nan
        q75 = np.nan
        min_lead_time = np.nan
        max_lead_time = np.nan

    false_warning_rate = safe_div(
        false_warning_episodes,
        len(episodes)
    )

    return {
        "total_events": total_events,
        "detected_events": detected,
        "missed_events": missed,
        "event_detection_rate": safe_div(
            detected,
            total_events
        ),
        "event_miss_rate": safe_div(
            missed,
            total_events
        ),
        "mean_lead_time_s": mean_lead,
        "median_lead_time_s": median_lead,
        "std_lead_time_s": std_lead,
        "lead_time_q25_s": q25,
        "lead_time_q75_s": q75,
        "min_lead_time_s": min_lead_time,
        "max_lead_time_s": max_lead_time,
        "total_warning_episodes": len(episodes),
        "false_warning_episodes": false_warning_episodes,
        "false_warning_episode_rate": false_warning_rate,
        "details": event_details,
    }


# ================================================================
# TIMELY DETECTION
# ================================================================

def timely_detection(
    y_binary,
    score,
    threshold,
    run_ids,
    time_array,
    lead_requirement
):

    events = extract_events(
        y_binary,
        run_ids
    )

    flags = (
        np.asarray(score) >= threshold
    ).astype(int)

    episodes = find_warning_episodes(
        flags,
        run_ids
    )

    detected = 0

    for event in events:

        run = event["run_id"]
        event_idx = event["start_index"]

        candidates = [
            ep for ep in episodes
            if ep[0] == run
            and ep[2] <= event_idx
        ]

        if not candidates:
            continue

        ep = max(
            candidates,
            key=lambda x: x[2]
        )

        warning_idx = ep[1]

        if time_array is not None:

            lead = (
                float(time_array[event_idx])
                - float(time_array[warning_idx])
            )

        else:

            lead = (
                event_idx - warning_idx
            ) * 0.05

        if lead >= lead_requirement:
            detected += 1

    return {
        "required_lead_time_s": lead_requirement,
        "total_events": len(events),
        "timely_detected_events": detected,
        "timely_detection_rate": safe_div(
            detected,
            len(events)
        ),
        "timely_missed_events":
            len(events) - detected,
    }


# ================================================================
# BOOTSTRAP BY RUN
# ================================================================

def bootstrap_mean_ci(
    values_by_run,
    n_bootstrap=N_BOOTSTRAP
):

    values_by_run = np.asarray(
        values_by_run,
        dtype=float
    )

    values_by_run = values_by_run[
        np.isfinite(values_by_run)
    ]

    if len(values_by_run) == 0:
        return np.nan, np.nan, np.nan

    point = float(np.mean(values_by_run))

    if len(values_by_run) == 1:
        return point, point, point

    boot = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):

        sample = rng.choice(
            values_by_run,
            size=len(values_by_run),
            replace=True
        )

        boot[i] = np.mean(sample)

    lower = float(
        np.percentile(boot, 2.5)
    )

    upper = float(
        np.percentile(boot, 97.5)
    )

    return point, lower, upper


# ================================================================
# RUN-LEVEL EVENT DETECTION
# ================================================================

def run_level_detection_values(
    y_binary,
    score,
    threshold,
    run_ids
):
    """
    Return one event-detection-rate value per run.

    IMPORTANT: event indices returned by extract_events() are indices into
    the full sequence-level arrays.  Never use a run-local boolean mask to
    index the full flags array; doing so can produce a mask-length mismatch.
    This implementation keeps all indices in the same global coordinate
    system and is therefore safe for variable-length runs.
    """
    y_binary = np.asarray(y_binary).reshape(-1)
    score = np.asarray(score).reshape(-1)
    run_ids = np.asarray(run_ids).reshape(-1)

    n = len(run_ids)
    if len(y_binary) != n or len(score) != n:
        raise ValueError(
            f"Run-level metric length mismatch: y={len(y_binary)}, "
            f"score={len(score)}, Run_ID={n}"
        )

    flags = (score >= threshold).astype(np.uint8)
    events = extract_events(y_binary, run_ids)

    values = []

    # Keep the event indices global.  For every event, check only the
    # corresponding global interval AND verify that it belongs to the run.
    for run in np.unique(run_ids):
        run_events = [e for e in events if e["run_id"] == run]
        if not run_events:
            continue

        run_mask = (run_ids == run)
        run_positions = np.flatnonzero(run_mask)
        if run_positions.size == 0:
            continue

        detected = 0
        for e in run_events:
            s = max(0, int(e["start_index"]))
            eidx = min(n - 1, int(e["end_index"]))
            if eidx < s:
                continue

            # Intersect the event interval with the actual positions of this
            # run. This avoids assuming that run samples are contiguous or
            # equally sized and never creates a mismatched boolean mask.
            lo = np.searchsorted(run_positions, s, side="left")
            hi = np.searchsorted(run_positions, eidx, side="right")
            if hi > lo and np.any(flags[run_positions[lo:hi]]):
                detected += 1

        values.append(detected / len(run_events))

    return np.asarray(values, dtype=float)


# ================================================================
# ROC / PR CURVES
# ================================================================

def plot_roc_pr(
    targets,
    predictions
):

    banner("GENERATING ROC / PR CURVES")

    fig, ax = plt.subplots()

    for model, pred in predictions.items():

        for j, horizon in enumerate(HORIZONS):

            y_true = targets[:, j]
            score = pred[:, j]

            if len(np.unique(y_true)) < 2:
                continue

            fpr, tpr, _ = roc_curve(
                y_true,
                score
            )

            auc = roc_auc_score(
                y_true,
                score
            )

            ax.plot(
                fpr,
                tpr,
                label=f"{model} {horizon} AUC={auc:.3f}"
            )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / "roc_curves.png",
        dpi=300
    )

    plt.close(fig)

    # ------------------------------------------------------------

    fig, ax = plt.subplots()

    for model, pred in predictions.items():

        for j, horizon in enumerate(HORIZONS):

            y_true = targets[:, j]
            score = pred[:, j]

            if len(np.unique(y_true)) < 2:
                continue

            precision, recall, _ = precision_recall_curve(
                y_true,
                score
            )

            auc = average_precision_score(
                y_true,
                score
            )

            ax.plot(
                recall,
                precision,
                label=f"{model} {horizon} AP={auc:.3f}"
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIG_DIR / "precision_recall_curves.png",
        dpi=300
    )

    plt.close(fig)


# ================================================================
# CONFUSION MATRICES
# ================================================================

def plot_confusion_matrices(
    targets,
    predictions,
    thresholds
):

    banner("GENERATING CONFUSION MATRICES")

    for model, pred in predictions.items():

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 3.8)
        )

        for j, horizon in enumerate(HORIZONS):

            score = pred[:, j]
            y_true = targets[:, j]

            threshold = thresholds[
                model
            ][horizon]

            y_pred = (
                score >= threshold
            ).astype(int)

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            )

            axes[j].imshow(cm)

            axes[j].set_title(
                f"{horizon}\nthreshold={threshold:.2f}"
            )

            axes[j].set_xlabel(
                "Predicted"
            )

            axes[j].set_ylabel(
                "Actual"
            )

            for r in range(2):
                for c in range(2):
                    axes[j].text(
                        c,
                        r,
                        str(cm[r, c]),
                        ha="center",
                        va="center"
                    )

        fig.suptitle(
            f"{model} Confusion Matrices"
        )

        fig.tight_layout()

        fig.savefig(
            FIG_DIR /
            f"{model.lower()}_confusion_matrices.png",
            dpi=300
        )

        plt.close(fig)


# ================================================================
# EXTENDED CLASSIFICATION
# ================================================================

def calculate_extended_classification(
    targets,
    predictions,
    thresholds
):

    banner("EXTENDED CLASSIFICATION METRICS")

    rows = []

    for model, pred in predictions.items():

        for j, horizon in enumerate(HORIZONS):

            metrics = classification_metrics(
                targets[:, j],
                pred[:, j],
                thresholds[model][horizon]
            )

            row = {
                "model": model,
                "horizon": horizon,
                "threshold": thresholds[
                    model
                ][horizon],
            }

            row.update(metrics)

            rows.append(row)

    df = pd.DataFrame(rows)

    path = (
        TABLE_DIR /
        "extended_classification_metrics.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(df.to_string(index=False))

    print(f"\nSaved:\n{path}")

    return df


# ================================================================
# TIMELY WARNING METRICS
# ================================================================

def calculate_timely_metrics(
    targets,
    predictions,
    thresholds,
    run_ids,
    time_array
):

    banner("TIMELY WARNING METRICS")

    rows = []

    for model, pred in predictions.items():

        for j, horizon in enumerate(HORIZONS):

            threshold = thresholds[
                model
            ][horizon]

            for lead_requirement in TIMELY_LEVELS:

                result = timely_detection(
                    targets[:, j],
                    pred[:, j],
                    threshold,
                    run_ids,
                    time_array,
                    lead_requirement
                )

                result.update({
                    "model": model,
                    "horizon": horizon,
                    "threshold": threshold,
                })

                rows.append(result)

    df = pd.DataFrame(rows)

    path = (
        TABLE_DIR /
        "timely_warning_metrics.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(df.to_string(index=False))

    print(f"\nSaved:\n{path}")

    return df


# ================================================================
# EVENT METRICS + CONFIDENCE INTERVALS
# ================================================================

def calculate_event_metrics(
    targets,
    predictions,
    thresholds,
    run_ids,
    time_array
):

    banner("EVENT-LEVEL METRICS AND CONFIDENCE INTERVALS")

    rows = []
    ci_rows = []

    for model, pred in predictions.items():

        for j, horizon in enumerate(HORIZONS):

            threshold = thresholds[
                model
            ][horizon]

            result = event_warning_metrics(
                targets[:, j],
                pred[:, j],
                threshold,
                run_ids,
                time_array
            )

            row = {
                "model": model,
                "horizon": horizon,
                "threshold": threshold,
            }

            for key, value in result.items():

                if key != "details":
                    row[key] = value

            rows.append(row)

            # Run-level detection values.
            run_values = run_level_detection_values(
                targets[:, j],
                pred[:, j],
                threshold,
                run_ids
            )

            point, lower, upper = bootstrap_mean_ci(
                run_values
            )

            ci_rows.append({
                "model": model,
                "horizon": horizon,
                "metric": "run_level_event_detection_rate",
                "estimate": point,
                "ci_lower": lower,
                "ci_upper": upper,
                "n_runs": len(run_values)
            })

    event_df = pd.DataFrame(rows)
    ci_df = pd.DataFrame(ci_rows)

    event_path = (
        TABLE_DIR /
        "extended_event_warning_metrics.csv"
    )

    ci_path = (
        TABLE_DIR /
        "bootstrap_confidence_intervals.csv"
    )

    event_df.to_csv(
        event_path,
        index=False
    )

    ci_df.to_csv(
        ci_path,
        index=False
    )

    print(event_df.to_string(index=False))

    print(f"\nSaved:")
    print(event_path)
    print(ci_path)

    return event_df, ci_df


# ================================================================
# LEAD-TIME DISTRIBUTION
# ================================================================

def plot_lead_time_distribution(
    targets,
    predictions,
    thresholds,
    run_ids,
    time_array
):

    banner("GENERATING LEAD-TIME DISTRIBUTION")

    data = []

    for model, pred in predictions.items():

        for j, horizon in enumerate(HORIZONS):

            result = event_warning_metrics(
                targets[:, j],
                pred[:, j],
                thresholds[model][horizon],
                run_ids,
                time_array
            )

            for d in result["details"]:

                if d["detected"] == 1:

                    data.append({
                        "model": model,
                        "horizon": horizon,
                        "lead_time_s":
                            d["lead_time_s"]
                    })

    df = pd.DataFrame(data)

    if df.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    for model in df["model"].unique():

        vals = df.loc[
            df["model"] == model,
            "lead_time_s"
        ]

        ax.hist(
            vals,
            bins=20,
            alpha=0.4,
            label=model
        )

    ax.set_xlabel(
        "Lead time before instability (s)"
    )

    ax.set_ylabel(
        "Detected events"
    )

    ax.set_title(
        "Event-Level Lead-Time Distribution"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIG_DIR /
        "lead_time_distribution.png",
        dpi=300
    )

    plt.close(fig)


# ================================================================
# TIMELY DETECTION PLOT
# ================================================================

def plot_timely_detection(
    timely_df
):

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for model in timely_df["model"].unique():

        subset = timely_df[
            timely_df["model"] == model
        ]

        for horizon in HORIZONS:

            h = subset[
                subset["horizon"] == horizon
            ]

            ax.plot(
                h["required_lead_time_s"],
                h["timely_detection_rate"],
                marker="o",
                label=f"{model} {horizon}"
            )

    ax.set_xlabel(
        "Required lead time (s)"
    )

    ax.set_ylabel(
        "Timely detection rate"
    )

    ax.set_ylim(0, 1.05)

    ax.set_title(
        "Timely Event Detection"
    )

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()

    fig.savefig(
        FIG_DIR /
        "timely_detection_vs_required_lead.png",
        dpi=300
    )

    plt.close(fig)


# ================================================================
# THRESHOLD SENSITIVITY
# ================================================================

def threshold_sensitivity(
    targets,
    predictions,
    thresholds,
    run_ids,
    time_array
):

    banner("THRESHOLD SENSITIVITY")

    rows = []

    threshold_grid = np.arange(
        0.05,
        1.001,
        0.05
    )

    for model, pred in predictions.items():

        for j, horizon in enumerate(HORIZONS):

            y_true = targets[:, j]
            score = pred[:, j]

            for threshold in threshold_grid:

                metrics = classification_metrics(
                    y_true,
                    score,
                    threshold
                )

                event_metrics = event_warning_metrics(
                    y_true,
                    score,
                    threshold,
                    run_ids,
                    time_array
                )

                rows.append({
                    "model": model,
                    "horizon": horizon,
                    "threshold": threshold,
                    "precision":
                        metrics["precision"],
                    "recall":
                        metrics["recall"],
                    "F1":
                        metrics["F1"],
                    "specificity":
                        metrics["specificity"],
                    "false_positive_rate":
                        metrics["false_positive_rate"],
                    "event_detection_rate":
                        event_metrics[
                            "event_detection_rate"
                        ],
                    "mean_lead_time_s":
                        event_metrics[
                            "mean_lead_time_s"
                        ],
                    "false_warning_episodes":
                        event_metrics[
                            "false_warning_episodes"
                        ]
                })

    df = pd.DataFrame(rows)

    path = (
        TABLE_DIR /
        "threshold_sensitivity.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    # One figure per model.
    for model in predictions:

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(14, 4)
        )

        for j, horizon in enumerate(HORIZONS):

            sub = df[
                (df["model"] == model)
                & (df["horizon"] == horizon)
            ]

            axes[j].plot(
                sub["threshold"],
                sub["event_detection_rate"],
                label="Event detection"
            )

            axes[j].plot(
                sub["threshold"],
                sub["precision"],
                label="Precision"
            )

            axes[j].plot(
                sub["threshold"],
                sub["F1"],
                label="F1"
            )

            selected = thresholds[
                model
            ][horizon]

            axes[j].axvline(
                selected,
                linestyle="--",
                label=f"Selected={selected:.2f}"
            )

            axes[j].set_title(horizon)
            axes[j].set_xlabel("Threshold")
            axes[j].set_ylim(0, 1.05)
            axes[j].grid(True, alpha=0.3)

        axes[0].set_ylabel("Metric")

        axes[0].legend(fontsize=7)

        fig.suptitle(
            f"{model}: Threshold Sensitivity"
        )

        fig.tight_layout()

        fig.savefig(
            FIG_DIR /
            f"{model.lower()}_threshold_sensitivity.png",
            dpi=300
        )

        plt.close(fig)

    print(f"Saved:\n{path}")

    return df


# ================================================================
# WARNING TIMELINE
# ================================================================

def plot_warning_timeline(
    y_binary,
    prediction,
    threshold,
    run_ids,
    time_array,
    model="GRU",
    horizon="1.0s"
):

    banner("GENERATING REPRESENTATIVE WARNING TIMELINE")

    # Pick the first run containing an event.
    events = extract_events(
        y_binary,
        run_ids
    )

    if not events:
        print("No events found.")
        return

    event = events[0]

    run = event["run_id"]

    indices = np.where(
        run_ids == run
    )[0]

    if len(indices) == 0:
        return

    start = indices[0]
    end = indices[-1]

    t = (
        time_array[start:end + 1]
        if time_array is not None
        else np.arange(
            len(indices)
        ) * 0.05
    )

    gt = y_binary[
        start:end + 1
    ]

    score = prediction[
        start:end + 1
    ]

    warning = (
        score >= threshold
    ).astype(int)

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        t,
        score,
        label="Predicted instability score"
    )

    ax.axhline(
        threshold,
        linestyle="--",
        label=f"Threshold={threshold:.2f}"
    )

    ax.fill_between(
        t,
        0,
        1,
        where=(gt == 1),
        alpha=0.25,
        transform=ax.get_xaxis_transform(),
        label="Ground-truth instability"
    )

    ax.step(
        t,
        warning,
        where="post",
        label="Warning state"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Prediction / state")
    ax.set_title(
        f"Representative {model} {horizon} Warning Timeline"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_DIR /
        "warning_timeline.png",
        dpi=300
    )

    plt.close(fig)


# ================================================================
# MODEL INFERENCE / DEPLOYMENT
# ================================================================

def deployment_metrics(
    predictions,
    X_test
):

    banner("DEPLOYMENT / COMPUTATIONAL METRICS")

    rows = []

    # Prediction-array based computational timing.
    # This measures loading/array operations only.
    # It does NOT claim neural-network inference latency.
    # Real model inference measurements require loading each
    # actual trained model.

    for model, pred in predictions.items():

        start = time.perf_counter()

        for _ in range(100):

            _ = np.asarray(pred)

        elapsed = time.perf_counter() - start

        microseconds = (
            elapsed / 100 * 1e6
        )

        rows.append({
            "model": model,
            "prediction_array_shape":
                str(pred.shape),
            "prediction_array_bytes":
                int(pred.nbytes),
            "prediction_array_MB":
                pred.nbytes / (1024 ** 2),
            "array_operation_time_us":
                microseconds
        })

    df = pd.DataFrame(rows)

    path = (
        TABLE_DIR /
        "deployment_metrics.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(df.to_string(index=False))

    print(
        "\nIMPORTANT:"
        "\nThese timings are array-operation measurements."
        "\nThey are NOT model inference latency."
    )

    print(f"\nSaved:\n{path}")

    return df


# ================================================================
# RUN-LEVEL STATISTICAL COMPARISON
# ================================================================

def run_level_comparison(
    targets,
    predictions,
    thresholds,
    run_ids
):

    banner("RUN-LEVEL STATISTICAL COMPARISON")

    rows = []

    run_values = {}

    for model, pred in predictions.items():

        run_values[model] = {}

        for j, horizon in enumerate(HORIZONS):

            values = run_level_detection_values(
                targets[:, j],
                pred[:, j],
                thresholds[model][horizon],
                run_ids
            )

            run_values[model][horizon] = values

    models = list(predictions.keys())

    for horizon in HORIZONS:

        for i in range(len(models)):

            for k in range(i + 1, len(models)):

                a = models[i]
                b = models[k]

                x = run_values[a][horizon]
                y = run_values[b][horizon]

                n = min(len(x), len(y))

                x = x[:n]
                y = y[:n]

                difference = x - y

                mean_diff = np.mean(
                    difference
                )

                p_value = np.nan

                if SCIPY_AVAILABLE and n >= 5:

                    try:

                        result = wilcoxon(
                            x,
                            y,
                            zero_method="wilcox",
                            alternative="two-sided"
                        )

                        p_value = result.pvalue

                    except Exception:
                        p_value = np.nan

                rows.append({
                    "horizon": horizon,
                    "model_A": a,
                    "model_B": b,
                    "mean_detection_rate_A":
                        np.mean(x),
                    "mean_detection_rate_B":
                        np.mean(y),
                    "mean_difference_A_minus_B":
                        mean_diff,
                    "wilcoxon_p_value":
                        p_value,
                    "n_runs": n
                })

    df = pd.DataFrame(rows)

    path = (
        TABLE_DIR /
        "run_level_model_comparison.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(df.to_string(index=False))

    print(f"\nSaved:\n{path}")

    return df


# ================================================================
# FINAL PAPER TABLE
# ================================================================

def create_final_paper_table(
    classification_df,
    event_df,
    timely_df
):

    banner("CREATING FINAL PAPER TABLE")

    rows = []

    for _, cls in classification_df.iterrows():

        model = cls["model"]
        horizon = cls["horizon"]

        matching = event_df[
            (event_df["model"] == model)
            & (event_df["horizon"] == horizon)
        ]

        if matching.empty:
            continue

        event = matching.iloc[0]

        timely = timely_df[
            (timely_df["model"] == model)
            & (timely_df["horizon"] == horizon)
            & (
                timely_df[
                    "required_lead_time_s"
                ] == 0.5
            )
        ]

        timely_rate = (
            timely.iloc[0]["timely_detection_rate"]
            if not timely.empty
            else np.nan
        )

        rows.append({
            "Model": model,
            "Horizon": horizon,
            "Threshold": cls["threshold"],
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Precision": cls["precision"],
            "Recall": cls["recall"],
            "Specificity": cls["specificity"],
            "F1": cls["F1"],
            "Balanced_Accuracy":
                cls["balanced_accuracy"],
            "MCC": cls["MCC"],
            "ROC_AUC": cls["ROC_AUC"],
            "PR_AUC": cls["PR_AUC"],
            "Event_Detection_Rate":
                event["event_detection_rate"],
            "Mean_Lead_Time_s":
                event["mean_lead_time_s"],
            "Median_Lead_Time_s":
                event["median_lead_time_s"],
            "False_Warning_Episodes":
                event["false_warning_episodes"],
            "Timely_Detection_at_0.5s":
                timely_rate
        })

    df = pd.DataFrame(rows)

    # Add forecasting metrics if existing final table exists.
    forecast_file = (
        TABLE_DIR /
        "final_forecasting_comparison.csv"
    )

    if forecast_file.exists():

        forecast = pd.read_csv(
            forecast_file
        )

        # Normalize model/horizon names.
        possible_cols = {
            c.lower(): c
            for c in forecast.columns
        }

        if (
            "model" in possible_cols
            and "horizon" in possible_cols
        ):

            model_col = possible_cols["model"]
            horizon_col = possible_cols["horizon"]

            for metric in [
                "MAE",
                "RMSE",
                "R2"
            ]:

                if metric not in forecast.columns:
                    continue

                mapping = forecast[
                    [
                        model_col,
                        horizon_col,
                        metric
                    ]
                ].copy()

                mapping = mapping.rename(
                    columns={
                        model_col: "Model",
                        horizon_col: "Horizon"
                    }
                )

                df = df.merge(
                    mapping,
                    on=["Model", "Horizon"],
                    how="left",
                    suffixes=(
                        "",
                        "_forecast"
                    )
                )

                if f"{metric}_forecast" in df.columns:

                    df[metric] = df[
                        f"{metric}_forecast"
                    ]

                    df.drop(
                        columns=[
                            f"{metric}_forecast"
                        ],
                        inplace=True
                    )

    path = (
        TABLE_DIR /
        "final_paper_table.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(df.to_string(index=False))

    print(f"\nSaved:\n{path}")

    return df


# ================================================================
# SUMMARY REPORT
# ================================================================

def write_summary_report(
    classification_df,
    event_df,
    timely_df,
    deployment_df
):

    banner("WRITING EXTENDED EVALUATION REPORT")

    path = (
        TABLE_DIR /
        "extended_evaluation_summary.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "EXTENDED VEHICLE STABILITY "
            "EARLY-WARNING EVALUATION\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            "This report summarizes the additional "
            "research-grade evaluation metrics.\n\n"
        )

        f.write(
            "MODELS\n"
        )

        f.write(
            "XGBoost, LSTM, GRU\n\n"
        )

        f.write(
            "HORIZONS\n"
        )

        f.write(
            "0.5 s, 1.0 s, 2.0 s\n\n"
        )

        f.write(
            "CLASSIFICATION METRICS\n"
        )

        f.write(
            classification_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "EVENT / EARLY-WARNING METRICS\n"
        )

        f.write(
            event_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "TIMELY DETECTION\n"
        )

        f.write(
            timely_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "DEPLOYMENT METRICS\n"
        )

        f.write(
            deployment_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "INTERPRETATION NOTE\n"
        )

        f.write(
            "Sample-level classification accuracy should "
            "not be used alone to characterize an "
            "early-warning system. Event detection, "
            "timely detection, lead time and false-warning "
            "behavior are reported separately.\n"
        )

        f.write(
            "\n"
            "Confidence intervals should be interpreted "
            "at the run level because samples within a "
            "vehicle run are temporally correlated.\n"
        )

        f.write(
            "\n"
            "No comparison with results from another "
            "paper should be interpreted as a direct "
            "numerical superiority claim unless the "
            "datasets, vehicle class, prediction target "
            "and evaluation protocol are comparable.\n"
        )

    print(f"Saved:\n{path}")


# ================================================================
# MAIN
# ================================================================

def main():

    banner(
        "EXTENDED FINAL VEHICLE STABILITY "
        "EARLY-WARNING EVALUATION"
    )

    print(
        "This script uses existing trained-model outputs "
        "and validation-calibrated thresholds."
    )

    print(
        "No model retraining is performed."
    )

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    X_test, y_test, run_ids, time_array = (
        load_test_sequences()
    )

    predictions = load_predictions()

    thresholds = load_thresholds()

    # ------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------

    targets = create_binary_targets(
        y_test
    )

    print()
    print("Ground-truth positive samples:")

    for j, horizon in enumerate(HORIZONS):

        print(
            f"  {horizon}: "
            f"{np.sum(targets[:, j])} / "
            f"{len(targets)}"
        )

    # ------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------

    classification_df = (
        calculate_extended_classification(
            targets,
            predictions,
            thresholds
        )
    )

    # ------------------------------------------------------------
    # Event-level
    # ------------------------------------------------------------

    event_df, ci_df = (
        calculate_event_metrics(
            targets,
            predictions,
            thresholds,
            run_ids,
            time_array
        )
    )

    # ------------------------------------------------------------
    # Timely detection
    # ------------------------------------------------------------

    timely_df = (
        calculate_timely_metrics(
            targets,
            predictions,
            thresholds,
            run_ids,
            time_array
        )
    )

    # ------------------------------------------------------------
    # ROC / PR
    # ------------------------------------------------------------

    plot_roc_pr(
        targets,
        predictions
    )

    # ------------------------------------------------------------
    # Confusion matrices
    # ------------------------------------------------------------

    plot_confusion_matrices(
        targets,
        predictions,
        thresholds
    )

    # ------------------------------------------------------------
    # Lead-time distribution
    # ------------------------------------------------------------

    plot_lead_time_distribution(
        targets,
        predictions,
        thresholds,
        run_ids,
        time_array
    )

    # ------------------------------------------------------------
    # Timely detection plot
    # ------------------------------------------------------------

    plot_timely_detection(
        timely_df
    )

    # ------------------------------------------------------------
    # Threshold sensitivity
    # ------------------------------------------------------------

    threshold_df = threshold_sensitivity(
        targets,
        predictions,
        thresholds,
        run_ids,
        time_array
    )

    # ------------------------------------------------------------
    # Representative timeline
    # ------------------------------------------------------------

    # GRU 1.0s is used only as an illustrative plot.
    # It is NOT being declared as the universally best model.
    plot_warning_timeline(
        targets[:, 1],
        predictions["GRU"][:, 1],
        thresholds["GRU"]["1.0s"],
        run_ids,
        time_array,
        model="GRU",
        horizon="1.0s"
    )

    # ------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------

    deployment_df = deployment_metrics(
        predictions,
        X_test
    )

    # ------------------------------------------------------------
    # Run-level statistical comparison
    # ------------------------------------------------------------

    comparison_df = run_level_comparison(
        targets,
        predictions,
        thresholds,
        run_ids
    )

    # ------------------------------------------------------------
    # Final paper table
    # ------------------------------------------------------------

    paper_df = create_final_paper_table(
        classification_df,
        event_df,
        timely_df
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    write_summary_report(
        classification_df,
        event_df,
        timely_df,
        deployment_df
    )

    # ------------------------------------------------------------
    # Final output list
    # ------------------------------------------------------------

    banner(
        "EXTENDED FINAL EVALUATION COMPLETE"
    )

    print("\nTABLES:")

    table_files = [
        "extended_classification_metrics.csv",
        "extended_event_warning_metrics.csv",
        "timely_warning_metrics.csv",
        "bootstrap_confidence_intervals.csv",
        "threshold_sensitivity.csv",
        "deployment_metrics.csv",
        "run_level_model_comparison.csv",
        "final_paper_table.csv",
        "extended_evaluation_summary.txt",
    ]

    for name in table_files:

        path = TABLE_DIR / name

        if path.exists():
            print(path)

    print("\nFIGURES:")

    for path in sorted(
        FIG_DIR.glob("*.png")
    ):
        print(path)

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "1. No model was retrained."
    )
    print(
        "2. Test thresholds came from validation."
    )
    print(
        "3. Timely-warning metrics use sequence-level Run_ID/Time."
    )
    print(
        "4. Confidence intervals are run-aware."
    )
    print(
        "5. No unsupported LTR metric was fabricated."
    )


if __name__ == "__main__":
    main()