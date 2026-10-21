"""Reviewer revision analyses: surface-stratified regression and run-cluster bootstrap.

Run from repository root:
    python .\\evaluation\\reviewer_revision_experiments.py

Uses existing test sequences/predictions and run metadata. Does not retrain models
or regenerate the dataset. Persistence uses the simulator's current ground-truth
margin and is a reference forecast, not a sensor-available baseline. Bootstrap
intervals quantify test-run resampling only;
they are not a substitute for repeated-seed training. Outputs are written under
results/tables/reviewer_revision/.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import event_lead_time as ev  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PRED_DIR = PROCESSED / "predictions"
OUT = ROOT / "results" / "tables" / "reviewer_revision"
HORIZONS = {"0.5s": 0, "1.0s": 1, "2.0s": 2}
FORECAST_MODELS = {
    "Persistence": None,
    "XGBoost": PRED_DIR / "xgboost_test_predictions.npy",
    "LSTM": PRED_DIR / "lstm_test_predictions.npy",
    "GRU": PRED_DIR / "gru_test_predictions.npy",
    "Hybrid": PRED_DIR / "hybrid_test_predictions.npy",
}
N_BOOTSTRAP = 2000
SEED = 20261020


def load_persistence_predictions(run_ids, times, y):
    """Build a genuine persistence forecast from current ground-truth margin.

    This is a simulated-data baseline, not an operational sensor-only baseline:
    current margin is taken from the simulator ground-truth file. It is included
    to compare forecast persistence consistently on the same held-out targets.
    """
    gt_path = RAW / "ground_truth_dynamics.csv"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Cannot build persistence baseline without {gt_path}"
        )
    gt = pd.read_csv(gt_path)
    required = {"Run_ID", "Time", "stability_margin"}
    missing = required.difference(gt.columns)
    if missing:
        raise ValueError(
            f"Ground-truth file missing columns: {sorted(missing)}"
        )
    keys = pd.DataFrame({
        "Run_ID": np.asarray(run_ids, dtype=int),
        "Time": np.asarray(times, dtype=float),
        "row": np.arange(len(run_ids), dtype=int),
    })
    current = keys.merge(
        gt[["Run_ID", "Time", "stability_margin"]],
        on=["Run_ID", "Time"], how="left", validate="one_to_one",
    ).sort_values("row")
    values = current["stability_margin"].to_numpy(dtype=float)
    if len(values) != len(y) or not np.isfinite(values).all():
        missing_count = int(np.isnan(values).sum())
        raise ValueError(
            "Persistence current-margin alignment failed: "
            f"{missing_count} missing values for {len(y)} test sequences."
        )
    return np.repeat(values[:, None], y.shape[1], axis=1)


def load_physics_observer_predictions(run_ids, times, y):
    """Compute sensor-only current-margin persistence for each held-out sequence."""
    sensor_path = RAW / "retrofit_sensor_stream.csv"
    if not sensor_path.exists():
        print("NOTE: sensor stream missing; physics-observer baseline omitted.")
        return None
    simulator_dir = ROOT / "simulator"
    sys.path.insert(0, str(simulator_dir))
    from vehicle_model import ALPHA_R_PEAK  # noqa: E402

    sensor = pd.read_csv(sensor_path)
    required = {
        "Run_ID", "Time", "yaw_rate_meas", "lateral_accel_meas", "speed_meas"
    }
    missing = required.difference(sensor.columns)
    if missing:
        raise ValueError(
            f"Sensor stream missing columns: {sorted(missing)}"
        )
    rows = []
    for run_id, frame in sensor.groupby("Run_ID", sort=True):
        frame = frame.sort_values("Time").reset_index(drop=True)
        t = frame["Time"].to_numpy(dtype=float)
        r = frame["yaw_rate_meas"].to_numpy(dtype=float)
        ay = frame["lateral_accel_meas"].to_numpy(dtype=float)
        vx = np.maximum(frame["speed_meas"].to_numpy(dtype=float), 1.0)
        vy_dot = ay - vx * r
        vy = np.zeros(len(frame), dtype=float)
        for i in range(1, len(frame)):
            dt = t[i] - t[i - 1]
            if dt <= 0 or dt > 0.5:
                vy[i] = 0.0
            else:
                vy[i] = vy[i - 1] + 0.5 * (vy_dot[i - 1] + vy_dot[i]) * dt
        alpha_r = np.arctan2(vy - 1.6 * r, vx)
        margin = (ALPHA_R_PEAK - np.abs(alpha_r)) / ALPHA_R_PEAK
        rows.append(pd.DataFrame({
            "Run_ID": int(run_id), "Time": t, "physics_margin": margin
        }))
    estimates = pd.concat(rows, ignore_index=True)
    keys = pd.DataFrame({
        "Run_ID": np.asarray(run_ids, dtype=int),
        "Time": np.asarray(times, dtype=float),
        "row": np.arange(len(run_ids), dtype=int),
    })
    aligned = keys.merge(
        estimates, on=["Run_ID", "Time"], how="left", validate="one_to_one"
    ).sort_values("row")
    margin = aligned["physics_margin"].to_numpy(dtype=float)
    if len(margin) != len(y) or not np.isfinite(margin).all():
        raise ValueError("Physics-observer margin did not align to all test sequences.")
    # Current sensor-estimated margin held constant for each forecast horizon.
    return np.repeat(margin[:, None], y.shape[1], axis=1)


def metric_rows_by_surface():
    seq = np.load(PROCESSED / "test_sequences.npz", allow_pickle=True)
    X, y = seq["X"], seq["y"].astype(float)
    run_ids = np.asarray(seq["Run_ID"]).astype(int)
    times = np.asarray(seq["Time"], dtype=float)
    metadata_path = RAW / "run_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = pd.read_csv(metadata_path).drop_duplicates("Run_ID")
    if "mu_surface" not in metadata.columns:
        raise ValueError("run_metadata.csv has no mu_surface field")
    frame = pd.DataFrame({"Run_ID": run_ids}).merge(
        metadata[["Run_ID", "mu_surface"]], on="Run_ID", how="left", validate="many_to_one"
    )
    if frame["mu_surface"].isna().any():
        raise ValueError("Some test sequences could not be assigned a surface category")
    predictions = {}
    for name, path in FORECAST_MODELS.items():
        if name == "Persistence":
            pred = load_persistence_predictions(run_ids, times, y)
        else:
            if not path.exists():
                print(f"NOTE: {name} predictions missing ({path}); skipping this model.")
                continue
            pred = np.load(path)
        if pred.shape != y.shape:
            raise ValueError(f"{name} predictions shape {pred.shape} != targets {y.shape}")
        predictions[name] = pred

    physics_pred = load_physics_observer_predictions(run_ids, times, y)
    if physics_pred is not None:
        predictions["Physics observer"] = physics_pred

    rows = []
    for surface, surface_frame in frame.groupby("mu_surface", sort=True):
        indices = surface_frame.index.to_numpy()
        n_runs = int(surface_frame["Run_ID"].nunique())
        for model, pred in predictions.items():
            for horizon, col in HORIZONS.items():
                truth = y[indices, col]
                estimate = pred[indices, col]
                rows.append({
                    "surface": str(surface), "model": model, "horizon": horizon,
                    "test_runs": n_runs, "test_sequences": len(indices),
                    "MAE": mean_absolute_error(truth, estimate),
                    "RMSE": float(np.sqrt(mean_squared_error(truth, estimate))),
                    "R2": r2_score(truth, estimate) if len(truth) > 1 else np.nan,
                })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "surface_stratified_forecasting_metrics.csv", index=False)
    return result


def bootstrap_event_metrics():
    """Fast run-cluster bootstrap using independent per-run event assignments.

    Warning-event matching is performed once per original run because matching
    is strictly run-local. Bootstrap draws then resample those run-level results.
    This preserves the cluster as the unit of resampling and avoids repeatedly
    rebuilding pandas timelines thousands of times.
    """
    rng = np.random.default_rng(SEED)
    thresholds = ev.load_calibrated_thresholds()
    _, y, run_ids, times = ev.load_test_data()
    base_timeline = ev.build_timeline(run_ids, times)
    gt = ev.load_ground_truth()
    test_runs = np.sort(np.unique(run_ids).astype(int))
    gt = gt[gt["Run_ID"].isin(test_runs)].copy()
    events = ev.extract_events(gt)
    if events.empty:
        raise RuntimeError("No test events available for bootstrap.")

    prediction_arrays = {}
    for model, path in ev.MODELS.items():
        if not path.exists():
            print(f"NOTE: {model} predictions missing ({path}); skipping bootstrap.")
            continue
        arr = np.load(path)
        if arr.shape != y.shape:
            raise ValueError(
                f"{model} prediction shape {arr.shape} != target shape {y.shape}"
            )
        prediction_arrays[model] = arr

    run_indices = {
        int(run): np.flatnonzero(np.asarray(run_ids, dtype=int) == int(run))
        for run in test_runs
    }
    event_by_run = {
        int(run): events[events["Run_ID"] == int(run)].copy()
        for run in test_runs
    }
    out_rows = []
    for model, predictions in prediction_arrays.items():
        for horizon, col in ev.HORIZONS.items():
            threshold = thresholds.get(model, {}).get(horizon, ev.THRESHOLD)
            # Cache independent run-level summaries. No event or warning can
            # match across run boundaries, so this is equivalent to resampling
            # complete runs and avoids repeating expensive matching work.
            run_summaries = {}
            for run in test_runs:
                run = int(run)
                idx = run_indices[run]
                local_timeline = pd.DataFrame({
                    "sample_index": np.arange(len(idx), dtype=int),
                    "Run_ID": np.full(len(idx), run, dtype=int),
                    "Time": np.asarray(times, dtype=float)[idx],
                })
                warning = ev.find_warning_episodes(
                    local_timeline, predictions[idx, col], threshold
                )
                run_events = event_by_run[run]
                matched, false = ev.match_warnings_to_events(run_events, warning)
                leads = matched.loc[
                    matched["early_warning_detected"], "lead_time_s"
                ].to_numpy(dtype=float)
                run_summaries[run] = {
                    "n_events": len(matched),
                    "n_detected": int(matched["early_warning_detected"].sum()),
                    "n_false": len(false),
                    "leads": leads,
                }

            samples = np.full((N_BOOTSTRAP, 3), np.nan, dtype=float)
            for b in range(N_BOOTSTRAP):
                drawn = rng.choice(test_runs, size=len(test_runs), replace=True)
                total_events = sum(run_summaries[int(r)]["n_events"] for r in drawn)
                total_detected = sum(run_summaries[int(r)]["n_detected"] for r in drawn)
                total_false = sum(run_summaries[int(r)]["n_false"] for r in drawn)
                if total_events:
                    samples[b, 0] = total_detected / total_events
                samples[b, 1] = total_false / len(drawn)
                lead_parts = [
                    run_summaries[int(r)]["leads"]
                    for r in drawn if len(run_summaries[int(r)]["leads"])
                ]
                if lead_parts:
                    samples[b, 2] = float(np.median(np.concatenate(lead_parts)))

            for metric_idx, metric in enumerate([
                "event_detection_rate",
                "unassociated_episodes_per_test_run",
                "median_lead_time_s",
            ]):
                values = samples[:, metric_idx]
                values = values[np.isfinite(values)]
                out_rows.append({
                    "model": model,
                    "horizon": horizon,
                    "metric": metric,
                    "bootstrap_replicates": N_BOOTSTRAP,
                    "seed": SEED,
                    "estimate": float(np.median(values)) if len(values) else np.nan,
                    "ci95_lower": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                    "ci95_upper": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                    "valid_replicates": len(values),
                })
            print(f"Bootstrap complete: {model} / {horizon} ({N_BOOTSTRAP} resamples)")
    result = pd.DataFrame(out_rows)
    result.to_csv(OUT / "run_cluster_bootstrap_intervals.csv", index=False)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    surface = metric_rows_by_surface()
    print("\nSURFACE-STRATIFIED FORECASTING METRICS")
    print(surface.to_string(index=False))
    bootstrap = bootstrap_event_metrics()
    print("\nRUN-CLUSTER BOOTSTRAP INTERVALS")
    print(bootstrap.to_string(index=False))
    print(f"\nSaved reviewer experiment outputs to: {OUT}")
    print(
        "\nInterpretation: bootstrap intervals quantify uncertainty across sampled test runs "
        "for this fixed trained model and threshold. They do not measure retraining-seed "
        "variability, external-vehicle generalization, or real-world safety."
    )


if __name__ == "__main__":
    main()
