"""Physics-baseline + learned residual forecaster.

The physics branch estimates the current rear-slip margin from retrofit
measurements only. XGBoost predicts the future-margin residual relative to
that current estimate. This is an experimental hybrid baseline, not a
validated production stability controller.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PRED_DIR = PROCESSED / "predictions"
TABLE_DIR = ROOT / "results" / "tables"
MODEL_DIR = ROOT / "results" / "models"
SIMULATOR = ROOT / "simulator"
sys.path.insert(0, str(SIMULATOR))
from vehicle_model import ALPHA_R_PEAK  # noqa: E402

SENSOR_PATH = RAW / "retrofit_sensor_stream.csv"
TRAIN_PATH = PROCESSED / "train_sequences.npz"
VAL_PATH = PROCESSED / "validation_sequences.npz"
TEST_PATH = PROCESSED / "test_sequences.npz"
HORIZONS = ["0.5s", "1.0s", "2.0s"]
LR = 1.6


def physics_margin_by_timestamp(sensor):
    """Estimate margin per run from sensors; no ground-truth states are used."""
    outputs = []
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
                # Restart integration after a timestamp gap to avoid integrating across it.
                vy[i] = 0.0
            else:
                vy[i] = vy[i - 1] + 0.5 * (vy_dot[i - 1] + vy_dot[i]) * dt
        alpha_r = np.arctan2(vy - LR * r, vx)
        margin = (ALPHA_R_PEAK - np.abs(alpha_r)) / ALPHA_R_PEAK
        outputs.append(pd.DataFrame({
            "Run_ID": int(run_id),
            "Time": t,
            "physics_margin": margin,
        }))
    return pd.concat(outputs, ignore_index=True)


def load_split(path, physics_lookup):
    archive = np.load(path, allow_pickle=True)
    required = {"X", "y", "Run_ID", "Time"}
    missing = required.difference(archive.files)
    if missing:
        raise KeyError(f"{path} missing keys: {sorted(missing)}")
    X = archive["X"].astype(np.float32)
    y = archive["y"].astype(np.float32)
    runs = archive["Run_ID"].astype(int)
    times = archive["Time"].astype(float)
    key = pd.DataFrame({"Run_ID": runs, "Time": times, "row": np.arange(len(runs))})
    key = key.merge(physics_lookup, on=["Run_ID", "Time"], how="left", validate="one_to_one")
    key = key.sort_values("row")
    margin = key["physics_margin"].to_numpy(dtype=np.float32)
    if not np.isfinite(margin).all():
        raise ValueError(f"Physics margin alignment failed for {path}; missing values found.")
    flat = X.reshape(len(X), -1)
    features = np.column_stack([flat, margin]).astype(np.float32)
    return features, y, margin


def metrics(y, pred):
    rows = []
    for i, horizon in enumerate(HORIZONS):
        rows.append({
            "model": "Physics+XGBoost residual",
            "horizon": horizon,
            "MAE": mean_absolute_error(y[:, i], pred[:, i]),
            "RMSE": float(np.sqrt(mean_squared_error(y[:, i], pred[:, i]))),
            "R2": r2_score(y[:, i], pred[:, i]),
        })
    return pd.DataFrame(rows)


def main():
    for path in [SENSOR_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)
    sensor = pd.read_csv(SENSOR_PATH)
    physics_lookup = physics_margin_by_timestamp(sensor)
    X_train, y_train, m_train = load_split(TRAIN_PATH, physics_lookup)
    X_val, y_val, m_val = load_split(VAL_PATH, physics_lookup)
    X_test, y_test, m_test = load_split(TEST_PATH, physics_lookup)

    # The physics branch is a current-margin persistence baseline.
    # The learner predicts the residual at each future horizon.
    residual_train = y_train - m_train[:, None]
    residual_val = y_val - m_val[:, None]
    residual_test = y_test - m_test[:, None]

    params = dict(
        n_estimators=350, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
    )
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    val_pred = np.zeros_like(y_val, dtype=np.float32)
    test_pred = np.zeros_like(y_test, dtype=np.float32)
    for j, horizon in enumerate(HORIZONS):
        model = XGBRegressor(**params)
        model.fit(X_train, residual_train[:, j])
        val_pred[:, j] = m_val + model.predict(X_val)
        test_pred[:, j] = m_test + model.predict(X_test)
        model.save_model(MODEL_DIR / f"hybrid_residual_{horizon}.json")
        print(f"{horizon}: validation residual MAE={mean_absolute_error(y_val[:, j], val_pred[:, j]):.4f}")

    np.save(PRED_DIR / "hybrid_test_predictions.npy", test_pred)
    np.save(PRED_DIR / "hybrid_validation_predictions.npy", val_pred)
    np.save(PRED_DIR / "hybrid_test_physics_margin.npy", m_test)
    np.save(PRED_DIR / "hybrid_validation_physics_margin.npy", m_val)
    table = metrics(y_test, test_pred)
    table.to_csv(TABLE_DIR / "hybrid_test_metrics.csv", index=False)
    print("\nHeld-out test metrics:")
    print(table.to_string(index=False))
    print(f"Saved predictions to {PRED_DIR}")


if __name__ == "__main__":
    main()
