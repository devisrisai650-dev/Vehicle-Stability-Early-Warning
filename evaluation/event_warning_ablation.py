r"""Matched validation-calibrated event-warning ablation.

Run from repository root:
    python .\evaluation\event_warning_ablation.py

Compares sensor-only XGBoost, physics-only sensor observer, and the
physics-plus-residual Hybrid. Each method/horizon receives a threshold selected
only on validation runs, using a fixed policy: maximize early-event detection
subject to no more than 0.5 unassociated warning episodes per validation run.
Ties are resolved by fewer unassociated warnings, then the higher threshold.
The frozen threshold is applied to test runs once.

This is simulation-only analysis. It does not prove real-vehicle safety.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import event_lead_time as ev  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PRED_DIR = PROCESSED / "predictions"
OUT = ROOT / "results" / "tables" / "reviewer_revision"
METHODS = {
    "XGBoost sensor-only": {
        "validation": PRED_DIR / "xgboost_validation_predictions.npy",
        "test": PRED_DIR / "xgboost_test_predictions.npy",
    },
    "Hybrid physics + residual": {
        "validation": PRED_DIR / "hybrid_validation_predictions.npy",
        "test": PRED_DIR / "hybrid_test_predictions.npy",
    },
}
THRESHOLDS = np.round(np.arange(-0.50, 0.5001, 0.05), 8)
MAX_UNASSOCIATED_PER_RUN = 0.5
LR = 1.6


def physics_margin_for_split(split):
    """Estimate current margin from measured/simulated sensor channels only."""
    sensor_path = RAW / "retrofit_sensor_stream.csv"
    sensor = pd.read_csv(sensor_path)
    required = {
        "Run_ID", "Time", "yaw_rate_meas", "lateral_accel_meas", "speed_meas"
    }
    missing = required.difference(sensor.columns)
    if missing:
        raise ValueError(f"Sensor stream missing columns: {sorted(missing)}")

    simulator_dir = ROOT / "simulator"
    sys.path.insert(0, str(simulator_dir))
    from vehicle_model import ALPHA_R_PEAK  # noqa: E402

    estimates = []
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
        alpha_r = np.arctan2(vy - LR * r, vx)
        margin = (ALPHA_R_PEAK - np.abs(alpha_r)) / ALPHA_R_PEAK
        estimates.append(pd.DataFrame({
            "Run_ID": int(run_id), "Time": t, "physics_margin": margin
        }))
    estimates = pd.concat(estimates, ignore_index=True)
    keys = pd.DataFrame({
        "Run_ID": np.asarray(split["Run_ID"], dtype=int),
        "Time": np.asarray(split["Time"], dtype=float),
        "row": np.arange(len(split["Run_ID"]), dtype=int),
    })
    aligned = keys.merge(
        estimates, on=["Run_ID", "Time"], how="left", validate="one_to_one"
    ).sort_values("row")
    margin = aligned["physics_margin"].to_numpy(dtype=float)
    if len(margin) != len(split["y"]) or not np.isfinite(margin).all():
        raise ValueError("Sensor-only physics margin did not align to all sequences.")
    return np.repeat(margin[:, None], split["y"].shape[1], axis=1)


def load_split(name):
    path = PROCESSED / f"{name}_sequences.npz"
    archive = np.load(path, allow_pickle=True)
    required = {"y", "Run_ID", "Time"}
    missing = required.difference(archive.files)
    if missing:
        raise KeyError(f"{path} missing keys: {sorted(missing)}")
    return {
        "y": archive["y"].astype(float),
        "Run_ID": archive["Run_ID"].astype(int),
        "Time": archive["Time"].astype(float),
    }


def load_prediction(path, target_shape):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file {path}. Regenerate validation/test "
            "predictions for this method before running the ablation."
        )
    pred = np.load(path).astype(float)
    if pred.shape != target_shape:
        raise ValueError(f"{path} shape {pred.shape} != target {target_shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    return pred


def split_events(run_ids):
    gt = ev.load_ground_truth()
    keep = set(np.unique(run_ids).astype(int))
    return ev.extract_events(gt[gt["Run_ID"].isin(keep)].copy())


def evaluate(predictions, split, events, threshold):
    timeline = ev.build_timeline(split["Run_ID"], split["Time"])
    rows = []
    for horizon, col in ev.HORIZONS.items():
        warnings = ev.find_warning_episodes(timeline, predictions[:, col], threshold)
        matched, unassociated = ev.match_warnings_to_events(events, warnings)
        detected = int(matched["early_warning_detected"].sum()) if len(matched) else 0
        total = len(matched)
        leads = matched.loc[matched["early_warning_detected"], "lead_time_s"]
        rows.append({
            "horizon": horizon,
            "threshold": float(threshold),
            "total_events": total,
            "early_detected_events": detected,
            "event_detection_rate": detected / total if total else np.nan,
            "late_warning_events": int(matched["late_warning_only"].sum()) if total else 0,
            "unassociated_warning_episodes": int(len(unassociated)),
            "unassociated_episodes_per_run": (
                len(unassociated) / timeline["Run_ID"].nunique()
                if timeline["Run_ID"].nunique() else np.nan
            ),
            "median_lead_time_s": float(leads.median()) if len(leads) else np.nan,
            "total_warning_episodes": int(len(warnings)),
        })
    return pd.DataFrame(rows)


def calibrate_threshold(pred, split, events, horizon_col):
    """Select a validation-only threshold under the predeclared warning-burden cap."""
    timeline = ev.build_timeline(split["Run_ID"], split["Time"])
    candidates = []
    for threshold in THRESHOLDS:
        warnings = ev.find_warning_episodes(timeline, pred[:, horizon_col], float(threshold))
        matched, unassociated = ev.match_warnings_to_events(events, warnings)
        total = len(matched)
        detected = int(matched["early_warning_detected"].sum()) if total else 0
        false_per_run = len(unassociated) / timeline["Run_ID"].nunique()
        leads = matched.loc[matched["early_warning_detected"], "lead_time_s"]
        candidates.append({
            "threshold": float(threshold),
            "detected": detected,
            "total_events": total,
            "detection_rate": detected / total if total else np.nan,
            "unassociated": int(len(unassociated)),
            "unassociated_per_run": float(false_per_run),
            "median_lead_time_s": float(leads.median()) if len(leads) else np.nan,
            "warning_episodes": int(len(warnings)),
        })
    feasible = [
        x for x in candidates
        if x["unassociated_per_run"] <= MAX_UNASSOCIATED_PER_RUN
    ]
    if feasible:
        # Predeclared: detection rate, fewer unassociated episodes, then higher threshold.
        chosen = sorted(
            feasible,
            key=lambda x: (
                -x["detection_rate"], x["unassociated"], -x["threshold"]
            ),
        )[0]
        chosen["calibration_feasible"] = True
    else:
        # No threshold meets burden cap: choose the lowest burden, then highest detection.
        chosen = sorted(
            candidates,
            key=lambda x: (
                x["unassociated_per_run"], -x["detection_rate"], -x["threshold"]
            ),
        )[0]
        chosen["calibration_feasible"] = False
    return chosen, candidates


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    validation = load_split("validation")
    test = load_split("test")
    val_events = split_events(validation["Run_ID"])
    test_events = split_events(test["Run_ID"])
    if val_events.empty or test_events.empty:
        raise RuntimeError("Validation and test partitions must both contain events.")

    val_predictions = {}
    test_predictions = {}
    for name, paths in METHODS.items():
        val_predictions[name] = load_prediction(paths["validation"], validation["y"].shape)
        test_predictions[name] = load_prediction(paths["test"], test["y"].shape)

    val_physics = physics_margin_for_split(validation)
    test_physics = physics_margin_for_split(test)
    val_predictions["Physics-only sensor observer"] = val_physics
    test_predictions["Physics-only sensor observer"] = test_physics

    calibration_rows = []
    calibration_grid_rows = []
    test_rows = []
    for method in METHODS.keys() | {"Physics-only sensor observer"}:
        for horizon, col in ev.HORIZONS.items():
            chosen, grid = calibrate_threshold(
                val_predictions[method], validation, val_events, col
            )
            calibration_rows.append({
                "method": method,
                "horizon": horizon,
                **chosen,
                "validation_runs": int(np.unique(validation["Run_ID"]).size),
            })
            for item in grid:
                calibration_grid_rows.append({
                    "method": method, "horizon": horizon, **item
                })
            test_result = evaluate(
                test_predictions[method], test, test_events, chosen["threshold"]
            )
            row = test_result.loc[test_result["horizon"] == horizon].iloc[0].to_dict()
            row.update({
                "method": method,
                "validation_threshold": chosen["threshold"],
                "validation_burden_constraint_feasible": chosen["calibration_feasible"],
                "test_runs": int(np.unique(test["Run_ID"]).size),
            })
            test_rows.append(row)
            print(
                f"{method} {horizon}: threshold={chosen['threshold']:+.2f}, "
                f"early={int(row['early_detected_events'])}/{int(row['total_events'])}, "
                f"unassociated/run={row['unassociated_episodes_per_run']:.3f}, "
                f"median lead={row['median_lead_time_s']:.3f}s"
            )

    pd.DataFrame(calibration_rows).to_csv(
        OUT / "event_warning_ablation_validation_thresholds.csv", index=False
    )
    pd.DataFrame(calibration_grid_rows).to_csv(
        OUT / "event_warning_ablation_validation_grid.csv", index=False
    )
    results = pd.DataFrame(test_rows)
    results.to_csv(OUT / "event_warning_ablation_test_results.csv", index=False)
    print("\nTEST EVENT-WARNING ABLATION RESULTS")
    print(results.to_string(index=False))
    print(f"\nSaved ablation outputs to {OUT}")
    print(
        "\nInterpretation: thresholds were selected only on validation runs under "
        f"a fixed cap of {MAX_UNASSOCIATED_PER_RUN:.2f} unassociated episodes/run. "
        "If the cap was infeasible, the output flags that and selects the lowest-burden "
        "candidate. All outcomes remain simulation-only."
    )


if __name__ == "__main__":
    main()
