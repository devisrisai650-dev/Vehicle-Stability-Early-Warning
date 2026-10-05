"""
FINAL TEST EVALUATION USING VALIDATION-CALIBRATED THRESHOLDS

Important:
- Uses the EXACT sequence-level Run_ID stored in test_sequences.npz.
- Uses sequence-level Time if it exists.
- Does NOT reconstruct sequence metadata from raw samples.
- Uses thresholds calibrated ONLY on validation data.
- Evaluates XGBoost, LSTM and GRU on the untouched test set.
- Computes event detection and lead-time statistics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = ROOT / "data" / "processed"
PRED_DIR = DATA_PROCESSED / "predictions"

RESULT_TABLES = ROOT / "results" / "tables"
RESULT_TABLES.mkdir(parents=True, exist_ok=True)

TEST_FILE = DATA_PROCESSED / "test_sequences.npz"

THRESHOLD_FILE = (
    DATA_PROCESSED / "selected_warning_thresholds.npz"
)

THRESHOLD_CSV = (
    RESULT_TABLES / "selected_warning_thresholds.csv"
)

GROUND_TRUTH_FILE = (
    ROOT / "data" / "raw" / "ground_truth_dynamics.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = ["XGBoost", "LSTM", "GRU"]

HORIZONS = ["0.5s", "1.0s", "2.0s"]

HORIZON_SECONDS = {
    "0.5s": 0.5,
    "1.0s": 1.0,
    "2.0s": 2.0,
}

PREDICTION_FILES = {
    "XGBoost": PRED_DIR / "xgboost_test_predictions.npy",
    "LSTM": PRED_DIR / "lstm_test_predictions.npy",
    "GRU": PRED_DIR / "gru_test_predictions.npy",
}

# Ground truth instability threshold.
# margin < 0 means unstable.
GROUND_TRUTH_THRESHOLD = 0.0


# ============================================================
# PRINT HEADER
# ============================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD TEST SEQUENCES
# ============================================================

def load_test_sequences():

    header("LOADING TEST SEQUENCES")

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test sequence file not found:\n{TEST_FILE}"
        )

    data = np.load(TEST_FILE, allow_pickle=True)

    print("Available keys:")
    for key in data.files:
        print(f"  {key}: shape={data[key].shape}")

    if "X" not in data:
        raise KeyError("test_sequences.npz does not contain X")

    if "y" not in data:
        raise KeyError("test_sequences.npz does not contain y")

    X = data["X"]
    y = data["y"]

    print()
    print(f"X_test shape: {X.shape}")
    print(f"y_test shape: {y.shape}")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X/y sample mismatch: {X.shape[0]} vs {y.shape[0]}"
        )

    # --------------------------------------------------------
    # CRITICAL:
    # Use sequence-level metadata directly.
    # --------------------------------------------------------

    run_key = None

    for key in ["Run_ID", "run_id", "runID", "RunID"]:
        if key in data.files:
            run_key = key
            break

    if run_key is None:
        raise KeyError(
            "\nNo sequence-level Run_ID found in test_sequences.npz.\n"
            "Do NOT reconstruct it from raw data.\n"
            "Regenerate test_sequences.npz with Run_ID saved."
        )

    Run_ID = np.asarray(data[run_key])

    if len(Run_ID) != len(X):
        raise ValueError(
            "\nSEQUENCE METADATA ERROR\n"
            f"X samples      : {len(X)}\n"
            f"Run_ID samples : {len(Run_ID)}\n\n"
            "The Run_ID stored in test_sequences.npz must have "
            "exactly one value per sequence."
        )

    print()
    print(f"Using sequence-level Run_ID key: '{run_key}'")
    print(f"Run_ID shape: {Run_ID.shape}")

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    time_key = None

    for key in ["Time", "time", "Timestamp", "timestamp"]:
        if key in data.files:
            time_key = key
            break

    Time = None

    if time_key is not None:

        Time = np.asarray(data[time_key])

        if len(Time) != len(X):
            raise ValueError(
                f"Sequence Time length mismatch:\n"
                f"X samples : {len(X)}\n"
                f"Time      : {len(Time)}"
            )

        print(f"Using sequence-level Time key: '{time_key}'")
        print(f"Time shape: {Time.shape}")

    else:
        print()
        print(
            "WARNING: No sequence-level Time found in test_sequences.npz."
        )
        print(
            "Lead time will be calculated using sequence ordering "
            "and the inferred sampling interval."
        )

    unique_runs = np.unique(Run_ID)

    print()
    print(f"Test samples : {len(X)}")
    print(f"Test runs    : {len(unique_runs)}")

    return X, y, Run_ID, Time


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    header("LOADING TEST PREDICTIONS")

    predictions = {}

    expected_n = None

    for model in MODELS:

        path = PREDICTION_FILES[model]

        if not path.exists():
            raise FileNotFoundError(
                f"{model} prediction file not found:\n{path}"
            )

        pred = np.load(path)

        print(f"{model}:")
        print(f"  File : {path}")
        print(f"  Shape: {pred.shape}")

        if pred.ndim != 2 or pred.shape[1] != 3:
            raise ValueError(
                f"{model} predictions must have shape "
                f"(N, 3), got {pred.shape}"
            )

        if expected_n is None:
            expected_n = pred.shape[0]

        if pred.shape[0] != expected_n:
            raise ValueError(
                f"Prediction sample mismatch for {model}:\n"
                f"Expected: {expected_n}\n"
                f"Got     : {pred.shape[0]}"
            )

        predictions[model] = pred

    return predictions


# ============================================================
# LOAD CALIBRATED THRESHOLDS
# ============================================================

def load_thresholds():

    header("LOADING VALIDATION-CALIBRATED THRESHOLDS")

    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            f"Threshold file not found:\n{THRESHOLD_FILE}"
        )

    data = np.load(THRESHOLD_FILE)

    print(f"Threshold file: {THRESHOLD_FILE}")
    print(f"Keys: {data.files}")

    thresholds = {}

    for model in MODELS:

        thresholds[model] = {}

        for horizon in HORIZONS:

            key = f"{model}_{horizon}"

            if key not in data.files:
                raise KeyError(
                    f"Missing threshold: {key}"
                )

            thresholds[model][horizon] = float(data[key])

    print()
    print("CALIBRATED THRESHOLDS")
    print("-" * 70)

    for model in MODELS:

        vals = thresholds[model]

        print(
            f"{model:<10}: "
            f"0.5s={vals['0.5s']:.2f}, "
            f"1.0s={vals['1.0s']:.2f}, "
            f"2.0s={vals['2.0s']:.2f}"
        )

    return thresholds


# ============================================================
# CREATE INSTABILITY FLAGS
# ============================================================

def create_ground_truth_flags(y):

    """
    y[:,0] = 0.5s stability margin
    y[:,1] = 1.0s stability margin
    y[:,2] = 2.0s stability margin

    Stability margin < 0 => instability.
    """

    return y < GROUND_TRUTH_THRESHOLD


def create_prediction_flags(predictions, threshold):

    """
    Predictions are stability-margin forecasts.

    Predicted margin < calibrated threshold => warning.
    """

    return predictions < threshold


# ============================================================
# EVENT EXTRACTION
# ============================================================

def extract_events(flags, run_ids, times=None):
    """
    Extract contiguous instability events separately for every run.

    Returns list of dictionaries.
    """

    events = []

    unique_runs = np.unique(run_ids)

    for run_id in unique_runs:

        indices = np.where(run_ids == run_id)[0]

        if len(indices) == 0:
            continue

        # Preserve original sequence order.
        indices = np.sort(indices)

        run_flags = flags[indices]

        start = None

        for i, flag in enumerate(run_flags):

            if flag and start is None:
                start = i

            elif not flag and start is not None:

                end = i - 1

                event_indices = indices[start:end + 1]

                events.append(
                    {
                        "run_id": run_id,
                        "start_index": int(event_indices[0]),
                        "end_index": int(event_indices[-1]),
                        "indices": event_indices,
                    }
                )

                start = None

        # Event continues until end of run.
        if start is not None:

            end = len(run_flags) - 1

            event_indices = indices[start:end + 1]

            events.append(
                {
                    "run_id": run_id,
                    "start_index": int(event_indices[0]),
                    "end_index": int(event_indices[-1]),
                    "indices": event_indices,
                }
            )

    return events


# ============================================================
# TIME DIFFERENCE
# ============================================================

def get_time_difference(Time, index1, index2):
    """
    Returns seconds between two sequence timestamps.
    """

    if Time is None:
        return None

    t1 = Time[index1]
    t2 = Time[index2]

    try:
        return float(t2 - t1)
    except Exception:
        return None


# ============================================================
# INFER SAMPLING INTERVAL
# ============================================================

def infer_dt(Time, Run_ID):

    if Time is None:
        return 0.05

    deltas = []

    for run_id in np.unique(Run_ID):

        idx = np.where(Run_ID == run_id)[0]

        if len(idx) < 2:
            continue

        idx = np.sort(idx)

        t = np.asarray(Time[idx], dtype=float)

        d = np.diff(t)

        d = d[np.isfinite(d)]

        d = d[d > 0]

        if len(d):
            deltas.extend(d.tolist())

    if not deltas:
        print(
            "Could not infer sampling interval. "
            "Using dt = 0.05 s."
        )
        return 0.05

    dt = float(np.median(deltas))

    print(f"Inferred sequence interval: {dt:.4f} s")

    return dt


# ============================================================
# EVENT MATCHING
# ============================================================

def evaluate_event_detection(
    truth_events,
    warning_events,
    horizon_seconds,
    Time,
    dt,
):
    """
    Match a warning episode to an instability event.

    A warning is considered useful if it begins before or at
    the beginning of the ground-truth instability event.

    Lead time = event start - first warning time.
    """

    matched_warning_indices = set()

    detected = 0
    missed = 0

    lead_times = []

    event_details = []

    for event_number, truth_event in enumerate(truth_events, start=1):

        run_id = truth_event["run_id"]

        event_start = truth_event["start_index"]
        event_end = truth_event["end_index"]

        candidate_indices = []

        for j, warning_event in enumerate(warning_events):

            if j in matched_warning_indices:
                continue

            if warning_event["run_id"] != run_id:
                continue

            warning_start = warning_event["start_index"]
            warning_end = warning_event["end_index"]

            # Warning must occur before event end.
            if warning_start > event_end:
                continue

            # Warning episode that overlaps the instability event
            # is also considered detection.
            if warning_end < event_start:
                # warning completely before event
                pass
            elif warning_start > event_end:
                continue

            candidate_indices.append(j)

        if not candidate_indices:

            missed += 1

            event_details.append(
                {
                    "event_id": event_number,
                    "run_id": run_id,
                    "event_start_index": event_start,
                    "event_end_index": event_end,
                    "detected": False,
                    "warning_index": None,
                    "lead_time_s": np.nan,
                }
            )

            continue

        # Select warning closest to event onset from before it.
        best_j = None
        best_lead = None

        for j in candidate_indices:

            warning_event = warning_events[j]

            warning_start = warning_event["start_index"]

            if Time is not None:

                lead = get_time_difference(
                    Time,
                    warning_start,
                    event_start
                )

                if lead is None:
                    lead = (event_start - warning_start) * dt

            else:
                lead = (event_start - warning_start) * dt

            # If warning starts after event, lead time becomes negative.
            # It still detects the event, but does not provide advance warning.
            if best_lead is None or abs(lead) < abs(best_lead):
                best_lead = lead
                best_j = j

        matched_warning_indices.add(best_j)

        detected += 1

        # For reporting, retain only positive/zero useful lead time.
        reported_lead = max(0.0, float(best_lead))

        lead_times.append(reported_lead)

        event_details.append(
            {
                "event_id": event_number,
                "run_id": run_id,
                "event_start_index": event_start,
                "event_end_index": event_end,
                "detected": True,
                "warning_index": best_j,
                "lead_time_s": reported_lead,
            }
        )

    missed = len(truth_events) - detected

    detection_rate = (
        detected / len(truth_events)
        if truth_events
        else 0.0
    )

    return {
        "detected": detected,
        "missed": missed,
        "detection_rate": detection_rate,
        "lead_times": lead_times,
        "matched_warning_indices": matched_warning_indices,
        "event_details": event_details,
    }


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def summarize_lead_times(lead_times):

    if not lead_times:

        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }

    arr = np.asarray(lead_times, dtype=float)

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


# ============================================================
# RUN MODEL / HORIZON
# ============================================================

def evaluate_model_horizon(
    model,
    horizon,
    predictions,
    y_test,
    Run_ID,
    Time,
    threshold,
    dt,
):

    horizon_idx = HORIZONS.index(horizon)

    pred_margin = predictions[:, horizon_idx]

    true_margin = y_test[:, horizon_idx]

    # --------------------------------------------------------
    # Ground truth instability
    # --------------------------------------------------------

    true_flags = true_margin < GROUND_TRUTH_THRESHOLD

    # --------------------------------------------------------
    # Predicted warning
    # --------------------------------------------------------

    warning_flags = pred_margin < threshold

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    truth_events = extract_events(
        true_flags,
        Run_ID,
        Time
    )

    warning_events = extract_events(
        warning_flags,
        Run_ID,
        Time
    )

    result = evaluate_event_detection(
        truth_events=truth_events,
        warning_events=warning_events,
        horizon_seconds=HORIZON_SECONDS[horizon],
        Time=Time,
        dt=dt,
    )

    lead_stats = summarize_lead_times(
        result["lead_times"]
    )

    matched_warning_indices = result[
        "matched_warning_indices"
    ]

    false_warning_episodes = 0

    for j in range(len(warning_events)):

        if j not in matched_warning_indices:
            false_warning_episodes += 1

    return {
        "model": model,
        "horizon": horizon,
        "threshold": threshold,
        "total_events": len(truth_events),
        "detected_events": result["detected"],
        "missed_events": result["missed"],
        "event_detection_rate": result["detection_rate"],
        "mean_lead_time_s": lead_stats["mean"],
        "median_lead_time_s": lead_stats["median"],
        "std_lead_time_s": lead_stats["std"],
        "lead_time_q25_s": lead_stats["q25"],
        "lead_time_q75_s": lead_stats["q75"],
        "total_warning_episodes": len(warning_events),
        "false_warning_episodes": false_warning_episodes,
    }, result["event_details"]


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "FINAL TEST EVALUATION USING "
        "VALIDATION-CALIBRATED THRESHOLDS"
    )

    # ========================================================
    # TEST DATA
    # ========================================================

    X_test, y_test, Run_ID, Time = load_test_sequences()

    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions = load_predictions()

    expected_n = len(X_test)

    for model in MODELS:

        if predictions[model].shape[0] != expected_n:
            raise ValueError(
                f"\n{model} prediction count does not match "
                f"test sequences.\n"
                f"Test samples : {expected_n}\n"
                f"Predictions  : "
                f"{predictions[model].shape[0]}"
            )

    # ========================================================
    # THRESHOLDS
    # ========================================================

    thresholds = load_thresholds()

    # ========================================================
    # TIME STEP
    # ========================================================

    header("SEQUENCE TIMING")

    dt = infer_dt(Time, Run_ID)

    if Time is not None:

        print(
            "Using sequence-aligned timestamps directly "
            "from test_sequences.npz."
        )

    else:

        print(
            "Time is unavailable in test_sequences.npz."
        )

        print(
            f"Using inferred/default dt = {dt:.4f} s."
        )

    # ========================================================
    # TEST EVENT COUNT
    # ========================================================

    header("TEST GROUND-TRUTH EVENTS")

    all_truth_events = {}

    for horizon in HORIZONS:

        idx = HORIZONS.index(horizon)

        flags = y_test[:, idx] < GROUND_TRUTH_THRESHOLD

        events = extract_events(
            flags,
            Run_ID,
            Time
        )

        all_truth_events[horizon] = events

        print(
            f"{horizon}: "
            f"{len(events)} events"
        )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    header("FINAL TEST EVALUATION")

    results = []
    details = []

    for model in MODELS:

        print()
        print("=" * 70)
        print(model)
        print("=" * 70)

        for horizon in HORIZONS:

            threshold = thresholds[model][horizon]

            print()
            print(
                f"Evaluating {horizon} "
                f"(threshold={threshold:.2f})..."
            )

            summary, event_details = evaluate_model_horizon(
                model=model,
                horizon=horizon,
                predictions=predictions[model],
                y_test=y_test,
                Run_ID=Run_ID,
                Time=Time,
                threshold=threshold,
                dt=dt,
            )

            results.append(summary)

            for detail in event_details:

                detail["model"] = model
                detail["horizon"] = horizon
                detail["threshold"] = threshold

                details.append(detail)

            print(
                f"Total events          : "
                f"{summary['total_events']}"
            )

            print(
                f"Detected events       : "
                f"{summary['detected_events']}"
            )

            print(
                f"Missed events         : "
                f"{summary['missed_events']}"
            )

            print(
                f"Detection rate        : "
                f"{summary['event_detection_rate']:.4f}"
            )

            if np.isfinite(summary["mean_lead_time_s"]):

                print(
                    f"Mean lead time        : "
                    f"{summary['mean_lead_time_s']:.4f} s"
                )

                print(
                    f"Median lead time      : "
                    f"{summary['median_lead_time_s']:.4f} s"
                )

                print(
                    f"Lead-time std         : "
                    f"{summary['std_lead_time_s']:.4f} s"
                )

                print(
                    f"Lead-time Q25         : "
                    f"{summary['lead_time_q25_s']:.4f} s"
                )

                print(
                    f"Lead-time Q75         : "
                    f"{summary['lead_time_q75_s']:.4f} s"
                )

            print(
                f"Warning episodes      : "
                f"{summary['total_warning_episodes']}"
            )

            print(
                f"False-warning episodes: "
                f"{summary['false_warning_episodes']}"
            )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    results_df = pd.DataFrame(results)

    details_df = pd.DataFrame(details)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    metrics_path = (
        RESULT_TABLES /
        "final_calibrated_event_lead_time_metrics.csv"
    )

    details_path = (
        RESULT_TABLES /
        "final_calibrated_event_lead_time_details.csv"
    )

    results_df.to_csv(
        metrics_path,
        index=False
    )

    details_df.to_csv(
        details_path,
        index=False
    )

    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================

    header("FINAL CALIBRATED TEST RESULTS")

    display_columns = [
        "model",
        "horizon",
        "threshold",
        "total_events",
        "detected_events",
        "missed_events",
        "event_detection_rate",
        "mean_lead_time_s",
        "median_lead_time_s",
        "std_lead_time_s",
        "lead_time_q25_s",
        "lead_time_q75_s",
        "total_warning_episodes",
        "false_warning_episodes",
    ]

    print(
        results_df[
            display_columns
        ].to_string(index=False)
    )

    # ========================================================
    # SAVE PRETTY TXT
    # ========================================================

    txt_path = (
        RESULT_TABLES /
        "final_calibrated_event_lead_time_results.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FINAL CALIBRATED TEST EVENT-LEVEL RESULTS\n"
        )

        f.write("=" * 100 + "\n\n")

        f.write(
            results_df[
                display_columns
            ].to_string(index=False)
        )

        f.write("\n")

    # ========================================================
    # DONE
    # ========================================================

    header("FINAL TEST EVALUATION COMPLETE")

    print()
    print("Saved:")
    print(metrics_path)
    print(details_path)
    print(txt_path)
    print()

    print(
        "IMPORTANT: Test Run_ID and Time were taken directly "
        "from test_sequences.npz."
    )

    print(
        "No raw-data sequence reconstruction was used."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()