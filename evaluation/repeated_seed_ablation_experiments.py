r"""Repeated-seed regression and physics/residual ablation experiments.

Run from the repository root:
    python .\evaluation\repeated_seed_ablation_experiments.py

This script uses the existing run-level train/validation/test split. For each
predeclared seed it trains sensor-only XGBoost and a physics-plus-XGBoost
residual model, then evaluates them on the same untouched test sequences.
The physics-only comparator holds a sensor-derived current-margin estimate
constant at each horizon. No thresholds are selected on the test set.

This is a reproducible research experiment, not real-vehicle validation.
It does not retrain LSTM/GRU and does not measure target-hardware latency.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "tables" / "reviewer_revision"
SIMULATOR = ROOT / "simulator"
sys.path.insert(0, str(SIMULATOR))
from vehicle_model import ALPHA_R_PEAK  # noqa: E402

# Predeclared seeds: results must be reported across all of them.
SEEDS = (11, 23, 37, 42, 58)
HORIZONS = ("0.5s", "1.0s", "2.0s")
LR = 1.6


def load_archive(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run dataset preprocessing before this script."
        )
    archive = np.load(path, allow_pickle=True)
    required = {"X", "y", "Run_ID", "Time"}
    missing = required.difference(archive.files)
    if missing:
        raise ValueError(f"{path} missing keys: {sorted(missing)}")
    return {
        "X": archive["X"].astype(np.float32),
        "y": archive["y"].astype(np.float32),
        "Run_ID": archive["Run_ID"].astype(int),
        "Time": archive["Time"].astype(float),
    }


def sensor_physics_margin() -> pd.DataFrame:
    path = RAW / "retrofit_sensor_stream.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    sensor = pd.read_csv(path)
    required = {
        "Run_ID", "Time", "yaw_rate_meas",
        "lateral_accel_meas", "speed_meas",
    }
    missing = required.difference(sensor.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    chunks = []
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
        chunks.append(pd.DataFrame({
            "Run_ID": int(run_id), "Time": t, "physics_margin": margin
        }))
    return pd.concat(chunks, ignore_index=True)


def align_margin(split, lookup: pd.DataFrame, name: str) -> np.ndarray:
    keys = pd.DataFrame({
        "Run_ID": split["Run_ID"],
        "Time": split["Time"],
        "row": np.arange(len(split["Run_ID"]), dtype=int),
    })
    aligned = keys.merge(
        lookup, on=["Run_ID", "Time"], how="left", validate="one_to_one"
    ).sort_values("row")
    margin = aligned["physics_margin"].to_numpy(dtype=np.float32)
    if len(margin) != len(split["y"]) or not np.isfinite(margin).all():
        raise ValueError(
            f"{name}: physics-margin alignment failed; "
            f"{int((~np.isfinite(margin)).sum())} missing rows."
        )
    return margin


def flatten(split) -> np.ndarray:
    return split["X"].reshape(len(split["X"]), -1).astype(np.float32)


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def model_params(seed: int):
    return dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=int(seed),
        n_jobs=-1,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_archive(PROCESSED / "train_sequences.npz")
    val = load_archive(PROCESSED / "validation_sequences.npz")
    test = load_archive(PROCESSED / "test_sequences.npz")
    lookup = sensor_physics_margin()

    # Match the established final-model convention: train on train + validation,
    # while leaving the test runs untouched for final reporting.
    X_train = np.concatenate([flatten(train), flatten(val)], axis=0)
    y_train = np.concatenate([train["y"], val["y"]], axis=0)
    m_train = np.concatenate([
        align_margin(train, lookup, "train"),
        align_margin(val, lookup, "validation"),
    ])
    X_test = flatten(test)
    y_test = test["y"]
    m_test = align_margin(test, lookup, "test")

    rows = []
    for seed in SEEDS:
        print(f"\nSeed {seed}: fitting XGBoost sensor-only and hybrid residual models")
        sensor_predictions = np.zeros_like(y_test, dtype=np.float32)
        hybrid_predictions = np.zeros_like(y_test, dtype=np.float32)

        for j, horizon in enumerate(HORIZONS):
            sensor_model = XGBRegressor(**model_params(seed))
            sensor_model.fit(X_train, y_train[:, j])
            sensor_predictions[:, j] = sensor_model.predict(X_test)

            residual_target = y_train[:, j] - m_train
            residual_model = XGBRegressor(**model_params(seed))
            residual_model.fit(X_train, residual_target)
            hybrid_predictions[:, j] = (
                m_test + residual_model.predict(X_test)
            )

        # Physics-only = current sensor-derived margin held constant.
        predictions = {
            "Physics-only": np.repeat(m_test[:, None], len(HORIZONS), axis=1),
            "XGBoost sensor-only": sensor_predictions,
            "Hybrid physics + residual": hybrid_predictions,
        }
        for method, pred in predictions.items():
            if pred.shape != y_test.shape:
                raise ValueError(
                    f"{method} prediction shape {pred.shape} != target {y_test.shape}"
                )
            for j, horizon in enumerate(HORIZONS):
                row = {
                    "seed": seed,
                    "method": method,
                    "horizon": horizon,
                    "test_runs": int(np.unique(test["Run_ID"]).size),
                    "test_sequences": int(len(y_test)),
                }
                row.update(metrics(y_test[:, j], pred[:, j]))
                rows.append(row)
        print(f"  Seed {seed} complete.")

    raw = pd.DataFrame(rows)
    raw_path = OUT / "repeated_seed_ablation_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = (
        raw.groupby(["method", "horizon"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            test_runs=("test_runs", "first"),
            test_sequences=("test_sequences", "first"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
    )
    summary_path = OUT / "repeated_seed_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nREPEATED-SEED / ABLATION SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nRaw per-seed metrics: {raw_path}")
    print(f"Mean/std summary: {summary_path}")
    print(
        "\nInterpretation limits: seed variation is measured only for the "
        "XGBoost sensor-only and physics-plus-residual implementations. "
        "Physics-only is a sensor-derived current-margin persistence comparator. "
        "No LSTM/GRU retraining, event-threshold recalibration, hardware benchmark, "
        "or real-vehicle validation is performed by this script."
    )


if __name__ == "__main__":
    main()
