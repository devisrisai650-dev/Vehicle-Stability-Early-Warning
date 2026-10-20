"""Reviewer revision analyses: surface-stratified regression and run-cluster bootstrap.

Run from repository root:
    python .\\evaluation\\reviewer_revision_experiments.py

Uses existing test sequences/predictions and run metadata. Does not retrain models
or regenerate the dataset. Bootstrap intervals quantify test-run resampling only;
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


def load_persistence(y):
    """Persistence forecast: use the latest observed target proxy.

    This baseline is valid only if a matching observed current-margin series
    exists. We do not silently substitute the first future target. If no
    observed current-margin array is available, the script reports it missing.
    """
    candidates = [
        PROCESSED / "persistence_test_predictions.npy",
        PRED_DIR / "persistence_test_predictions.npy",
    ]
    for path in candidates:
        if path.exists():
            pred = np.load(path)
            if pred.shape == y.shape:
                return pred
            raise ValueError(f"Persistence predictions shape {pred.shape} != y shape {y.shape}")
    return None


def metric_rows_by_surface():
    seq = np.load(PROCESSED / "test_sequences.npz", allow_pickle=True)
    X, y = seq["X"], seq["y"].astype(float)
    run_ids = np.asarray(seq["Run_ID"]).astype(int)
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
            pred = load_persistence(y)
            if pred is None:
                print("NOTE: persistence_test_predictions.npy not found; persistence surface metrics omitted.")
                continue
        else:
            if not path.exists():
                print(f"NOTE: {name} predictions missing ({path}); skipping this model.")
                continue
            pred = np.load(path)
        if pred.shape != y.shape:
            raise ValueError(f"{name} predictions shape {pred.shape} != targets {y.shape}")
        predictions[name] = pred

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
    """Cluster bootstrap independent test runs, preserving each run's sequence.

    Each sampled run copy gets a synthetic ID so duplicate bootstrap draws are
    treated as distinct clusters. Thresholds remain the frozen validation
    thresholds; no threshold is selected from the test data.
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

    # Cache per-model prediction arrays and per-run event/warning ingredients.
    prediction_arrays = {}
    for model, path in ev.MODELS.items():
        if not path.exists():
            print(f"NOTE: {model} predictions missing ({path}); skipping bootstrap.")
            continue
        arr = np.load(path)
        if arr.shape != y.shape:
            raise ValueError(f"{model} prediction shape {arr.shape} != target shape {y.shape}")
        prediction_arrays[model] = arr

    run_indices = {
        int(run): base_timeline.index[base_timeline["Run_ID"] == int(run)].to_numpy()
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
            samples = []
            for b in range(N_BOOTSTRAP):
                drawn = rng.choice(test_runs, size=len(test_runs), replace=True)
                timeline_parts, event_parts = [], []
                next_boot_id = 0
                for source_run in drawn:
                    new_id = next_boot_id
                    next_boot_id += 1
                    idx = run_indices[int(source_run)]
                    part = base_timeline.loc[idx, ["sample_index", "Time"]].copy()
                    part["Run_ID"] = new_id
                    timeline_parts.append(part)
                    epart = event_by_run[int(source_run)].copy()
                    if not epart.empty:
                        epart["Run_ID"] = new_id
                        event_parts.append(epart)
                timeline = pd.concat(timeline_parts, ignore_index=True)
                boot_events = (
                    pd.concat(event_parts, ignore_index=True)
                    if event_parts else pd.DataFrame(columns=events.columns)
                )
                if boot_events.empty:
                    samples.append((np.nan, np.nan, np.nan))
                    continue
                warning = ev.find_warning_episodes(
                    timeline, predictions[:, col], threshold
                )
                matched, false = ev.match_warnings_to_events(boot_events, warning)
                detection = float(matched["early_warning_detected"].mean())
                false_per_run = float(len(false) / len(drawn))
                leads = matched.loc[matched["early_warning_detected"], "lead_time_s"]
                median_lead = float(leads.median()) if len(leads) else np.nan
                samples.append((detection, false_per_run, median_lead))
            arr = np.asarray(samples, dtype=float)
            for metric_idx, metric in enumerate([
                "event_detection_rate", "unassociated_episodes_per_test_run", "median_lead_time_s"
            ]):
                values = arr[:, metric_idx]
                values = values[np.isfinite(values)]
                out_rows.append({
                    "model": model, "horizon": horizon, "metric": metric,
                    "bootstrap_replicates": N_BOOTSTRAP,
                    "seed": SEED,
                    "estimate": float(np.nanmedian(values)) if len(values) else np.nan,
                    "ci95_lower": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                    "ci95_upper": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                    "valid_replicates": len(values),
                })
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
