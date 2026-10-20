"""Event-level early-warning evaluation on the held-out test runs.

Warnings are issued at the prediction timestamp (t), using the model's
forecast for the selected future horizon. A warning counts as advance
detection only when its onset is at or before the ground-truth event onset
and within MAX_WARNING_WINDOW_S. Late warnings are reported separately and
never counted as successful early warnings.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PRED_DIR = PROCESSED / "predictions"
GT_PATH = ROOT / "data" / "raw" / "ground_truth_dynamics.csv"
TABLE_DIR = ROOT / "results" / "tables"
TEST_PATH = PROCESSED / "test_sequences.npz"

MODELS = {
    "XGBoost": PRED_DIR / "xgboost_test_predictions.npy",
    "LSTM": PRED_DIR / "lstm_test_predictions.npy",
    "GRU": PRED_DIR / "gru_test_predictions.npy",
}
HORIZONS = {"0.5s": 0, "1.0s": 1, "2.0s": 2}
THRESHOLD = 0.0  # physics-defined margin boundary; calibrate on validation for final reporting
CONSECUTIVE_WARNINGS = 2
WARNING_GAP_S = 0.20
MAX_WARNING_WINDOW_S = 5.0


def load_test_data():
    data = np.load(TEST_PATH, allow_pickle=True)
    required = {"X", "y", "Run_ID", "Time"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"{TEST_PATH} missing keys: {sorted(missing)}")
    X = data["X"]
    y = data["y"].astype(float)
    run_ids = np.asarray(data["Run_ID"]).astype(int)
    times = np.asarray(data["Time"]).astype(float)
    if not (len(X) == len(y) == len(run_ids) == len(times)):
        raise ValueError("Test sequence arrays have inconsistent lengths.")
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError(f"Expected y shape (N, 3), got {y.shape}")
    return X, y, run_ids, times


def load_ground_truth():
    gt = pd.read_csv(GT_PATH)
    required = {"Run_ID", "Time", "instability_flag", "stability_margin", "beta", "yaw_rate"}
    missing = required.difference(gt.columns)
    if missing:
        raise ValueError(f"Ground-truth file missing columns: {sorted(missing)}")
    gt["Run_ID"] = gt["Run_ID"].astype(int)
    gt["Time"] = gt["Time"].astype(float)
    # CSV bool values can be either bools or strings depending on pandas/version.
    if gt["instability_flag"].dtype != bool:
        gt["instability_flag"] = gt["instability_flag"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    if gt["instability_flag"].isna().any():
        raise ValueError("Could not parse some instability_flag values.")
    return gt


def extract_events(gt):
    events = []
    event_id = 0
    for run_id, frame in gt.groupby("Run_ID", sort=True):
        frame = frame.sort_values("Time").reset_index(drop=True)
        flags = frame["instability_flag"].to_numpy(dtype=bool)
        i = 0
        while i < len(frame):
            if not flags[i]:
                i += 1
                continue
            start = i
            while i + 1 < len(frame) and flags[i + 1]:
                i += 1
            event = frame.iloc[start:i + 1]
            event_id += 1
            events.append({
                "Run_ID": int(run_id),
                "event_id": event_id,
                "onset_time": float(event["Time"].iloc[0]),
                "end_time": float(event["Time"].iloc[-1]),
                "duration_s": float(event["Time"].iloc[-1] - event["Time"].iloc[0]),
                "minimum_margin": float(event["stability_margin"].min()),
                "max_abs_beta": float(event["beta"].abs().max()),
                "max_abs_yaw_rate": float(event["yaw_rate"].abs().max()),
            })
            i += 1
    return pd.DataFrame(events, columns=[
        "Run_ID", "event_id", "onset_time", "end_time", "duration_s",
        "minimum_margin", "max_abs_beta", "max_abs_yaw_rate"
    ])


def build_timeline(run_ids, times):
    return pd.DataFrame({
        "sample_index": np.arange(len(run_ids), dtype=int),
        "Run_ID": np.asarray(run_ids, dtype=int),
        "Time": np.asarray(times, dtype=float),
    })


def find_warning_episodes(timeline, predictions, threshold):
    """Group sustained threshold crossings into episodes, independently per run."""
    predictions = np.asarray(predictions, dtype=float)
    if len(predictions) != len(timeline):
        raise ValueError("Prediction count does not match timeline length.")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    episodes = []
    for run_id, frame in timeline.groupby("Run_ID", sort=True):
        frame = frame.sort_values("Time").reset_index(drop=True)
        indices = frame["sample_index"].to_numpy(dtype=int)
        times = frame["Time"].to_numpy(dtype=float)
        flags = predictions[indices] < threshold

        sustained = np.zeros(len(flags), dtype=bool)
        run_length = 0
        for i, flag in enumerate(flags):
            run_length = run_length + 1 if flag else 0
            if run_length >= CONSECUTIVE_WARNINGS:
                sustained[i - CONSECUTIVE_WARNINGS + 1:i + 1] = True

        idx = np.flatnonzero(sustained)
        if len(idx) == 0:
            continue
        groups = []
        start = previous = int(idx[0])
        for current in idx[1:]:
            current = int(current)
            if times[current] - times[previous] > WARNING_GAP_S:
                groups.append((start, previous))
                start = current
            previous = current
        groups.append((start, previous))

        for start, end in groups:
            episode_idx = np.arange(start, end + 1)
            episodes.append({
                "Run_ID": int(run_id),
                "warning_start": float(times[start]),
                "warning_end": float(times[end]),
                "duration_s": float(times[end] - times[start]),
                "minimum_prediction": float(predictions[indices[episode_idx]].min()),
            })
    return pd.DataFrame(episodes, columns=[
        "Run_ID", "warning_start", "warning_end", "duration_s", "minimum_prediction"
    ])


def match_warnings_to_events(events, warnings):
    """One-to-one match each event to its closest preceding relevant warning.

    A warning is a true early warning only if its start is no later than event
    onset and no more than MAX_WARNING_WINDOW_S earlier. Warnings after onset
    are classified as late, not successful detections. Each warning can match
    at most one event.
    """
    event_rows = []
    used_warning_ids = set()
    warning_frame = warnings.reset_index(drop=True).copy()
    warning_frame["warning_id"] = np.arange(len(warning_frame), dtype=int)

    for _, event in events.sort_values(["Run_ID", "onset_time"]).iterrows():
        run_id = int(event["Run_ID"])
        onset = float(event["onset_time"])
        same_run = warning_frame[warning_frame["Run_ID"] == run_id]
        available = same_run[~same_run["warning_id"].isin(used_warning_ids)]

        eligible = available[
            (available["warning_start"] <= onset)
            & (available["warning_start"] >= onset - MAX_WARNING_WINDOW_S)
        ]
        late = available[
            (available["warning_start"] > onset)
            & (available["warning_start"] <= float(event["end_time"]))
        ]

        matched = False
        warning_id = np.nan
        warning_time = np.nan
        lead_time = np.nan
        late_warning = False
        if not eligible.empty:
            # Most recent warning before onset gives the conservative usable lead.
            chosen = eligible.sort_values("warning_start").iloc[-1]
            warning_id = int(chosen["warning_id"])
            used_warning_ids.add(warning_id)
            warning_time = float(chosen["warning_start"])
            lead_time = onset - warning_time
            matched = True
        elif not late.empty:
            # Record late detection separately; do not mark it as early detection.
            chosen = late.sort_values("warning_start").iloc[0]
            warning_id = int(chosen["warning_id"])
            used_warning_ids.add(warning_id)
            warning_time = float(chosen["warning_start"])
            late_warning = True

        row = event.to_dict()
        row.update({
            "early_warning_detected": matched,
            "late_warning_only": late_warning,
            "matched_warning_id": warning_id,
            "first_warning_time": warning_time,
            "lead_time_s": lead_time,
        })
        event_rows.append(row)

    event_results = pd.DataFrame(event_rows)
    if warning_frame.empty:
        false_warnings = warning_frame.copy()
    else:
        # Any warning not matched to an event is a false/unassociated episode.
        false_warnings = warning_frame[
            ~warning_frame["warning_id"].isin(used_warning_ids)
        ].copy()
        false_warnings["classification"] = "unassociated_warning"
    return event_results, false_warnings


def evaluate_model(model_name, prediction_path, y_true, timeline, events):
    predictions = np.load(prediction_path)
    if predictions.shape != y_true.shape:
        raise ValueError(
            f"{model_name} prediction shape {predictions.shape} != target shape {y_true.shape}"
        )
    rows = []
    event_tables = []
    false_tables = []
    for horizon, column in HORIZONS.items():
        warnings = find_warning_episodes(timeline, predictions[:, column], THRESHOLD)
        event_results, false_warnings = match_warnings_to_events(events, warnings)
        leads = event_results.loc[event_results["early_warning_detected"], "lead_time_s"]
        total_events = len(event_results)
        detected = int(event_results["early_warning_detected"].sum()) if total_events else 0
        late_count = int(event_results["late_warning_only"].sum()) if total_events else 0
        rows.append({
            "model": model_name,
            "horizon": horizon,
            "threshold": THRESHOLD,
            "total_events": total_events,
            "early_detected_events": detected,
            "missed_events": total_events - detected,
            "late_warning_events": late_count,
            "event_detection_rate": detected / total_events if total_events else np.nan,
            "mean_lead_time_s": float(leads.mean()) if len(leads) else np.nan,
            "median_lead_time_s": float(leads.median()) if len(leads) else np.nan,
            "std_lead_time_s": float(leads.std(ddof=1)) if len(leads) > 1 else np.nan,
            "lead_time_q25_s": float(leads.quantile(0.25)) if len(leads) else np.nan,
            "lead_time_q75_s": float(leads.quantile(0.75)) if len(leads) else np.nan,
            "total_warning_episodes": len(warnings),
            "false_warning_episodes": len(false_warnings),
            "false_warning_episodes_per_test_run": (
                len(false_warnings) / timeline["Run_ID"].nunique()
                if timeline["Run_ID"].nunique() else np.nan
            ),
        })
        if not event_results.empty:
            event_results["model"] = model_name
            event_results["horizon"] = horizon
            event_tables.append(event_results)
        if not false_warnings.empty:
            false_warnings["model"] = model_name
            false_warnings["horizon"] = horizon
            false_tables.append(false_warnings)
        print(
            f"{model_name} {horizon}: early detection={detected}/{total_events}, "
            f"late-only={late_count}, false/unassociated warnings={len(false_warnings)}, "
            f"median lead={leads.median() if len(leads) else float('nan'):.3f}s"
        )
    return (
        pd.DataFrame(rows),
        pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame(),
        pd.concat(false_tables, ignore_index=True) if false_tables else pd.DataFrame(),
    )


def main():
    _, y_test, run_ids, times = load_test_data()
    timeline = build_timeline(run_ids, times)
    gt = load_ground_truth()
    test_run_ids = set(np.unique(run_ids).astype(int))
    gt_test = gt[gt["Run_ID"].isin(test_run_ids)].copy()
    events = extract_events(gt_test)
    if events.empty:
        raise RuntimeError("No ground-truth instability events found in test runs.")

    summaries, event_tables, false_tables = [], [], []
    for model_name, prediction_path in MODELS.items():
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing test predictions for {model_name}: {prediction_path}")
        summary, event_details, false_details = evaluate_model(
            model_name, prediction_path, y_test, timeline, events
        )
        summaries.append(summary)
        if not event_details.empty:
            event_tables.append(event_details)
        if not false_details.empty:
            false_tables.append(false_details)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(TABLE_DIR / "event_lead_time_metrics.csv", index=False)
    if event_tables:
        pd.concat(event_tables, ignore_index=True).to_csv(
            TABLE_DIR / "event_lead_time_details.csv", index=False
        )
    if false_tables:
        pd.concat(false_tables, ignore_index=True).to_csv(
            TABLE_DIR / "event_false_warning_details.csv", index=False
        )
    print("\nEVENT-LEVEL SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved results to {TABLE_DIR}")


if __name__ == "__main__":
    main()
