
"""
ROBUSTNESS + GENERALIZATION EVALUATION
Vehicle Stability Early-Warning Project

What this script adds:
1. Stratified forecasting performance by:
   - road friction / surface (mu_surface)
   - target speed
   - maneuver
   - steering amplitude
2. Sensor-noise robustness:
   - 0%, 5%, 10%, 20% feature-wise Gaussian perturbation
3. Per-condition degradation relative to clean test performance
4. Worst-case condition identification
5. Publication-ready CSV tables and PNG figures

IMPORTANT:
- This script DOES NOT change your trained models.
- It evaluates the already-trained final models.
- It never uses ground-truth beta/stability margin as model input.
- It uses test_sequences.npz Run_ID/Time directly.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import xgboost as xgb


# ==============================================================
# PATHS
# ==============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
PRED = PROCESSED / "predictions"
MODELS = ROOT / "results" / "models"
OUT_T = ROOT / "results" / "tables"
OUT_F = ROOT / "results" / "figures"

OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

TEST_SEQ = PROCESSED / "test_sequences.npz"
RUN_META = RAW / "run_metadata.csv"
NORM_FILE = PROCESSED / "lstm_normalization.npz"

HORIZONS = ["0.5s", "1.0s", "2.0s"]
FEATURE_NAMES = [
    "yaw_rate_meas",
    "lateral_accel_meas",
    "speed_meas",
    "steering_angle_meas",
]

MODELS_TO_USE = ["XGBoost", "LSTM", "GRU"]


# ==============================================================
# HELPERS
# ==============================================================

def metric_row(model, horizon, y_true, y_pred, subset_name,
               subset_value, noise_pct=0):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "model": model,
        "horizon": horizon,
        "subset": subset_name,
        "condition": str(subset_value),
        "noise_percent": noise_pct,
        "n_samples": len(y_true),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def load_test():
    d = np.load(TEST_SEQ)

    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)
    run_id = d["Run_ID"]

    if len(X) != len(y) or len(X) != len(run_id):
        raise ValueError(
            f"Sequence alignment error: X={len(X)}, y={len(y)}, "
            f"Run_ID={len(run_id)}"
        )

    return X, y, run_id


def load_metadata():
    if not RUN_META.exists():
        raise FileNotFoundError(
            f"Missing run metadata:\n{RUN_META}\n\n"
            "This file is required for condition-wise generalization."
        )

    meta = pd.read_csv(RUN_META)

    required = {"Run_ID"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            f"run_metadata.csv is missing columns: {sorted(missing)}"
        )

    return meta


def merge_metadata(run_ids, meta):
    # Only keep one row per Run_ID.
    m = meta.drop_duplicates("Run_ID").copy()

    cols = [
        c for c in [
            "Run_ID",
            "maneuver",
            "mu_surface",
            "mu_true",
            "target_speed_kmh",
            "amplitude_deg",
        ] if c in m.columns
    ]

    merged = pd.DataFrame({"Run_ID": run_ids}).merge(
        m[cols],
        on="Run_ID",
        how="left",
        validate="many_to_one",
    )

    if merged["Run_ID"].isna().any():
        raise ValueError("Run_ID metadata merge failed.")

    if merged.isna().all(axis=1).any():
        bad = merged.loc[merged.isna().all(axis=1), "Run_ID"].unique()
        raise ValueError(f"No metadata found for Run_ID values: {bad}")

    return merged


def load_clean_predictions():
    paths = {
        "XGBoost": PRED / "xgboost_test_predictions.npy",
        "LSTM": PRED / "lstm_test_predictions.npy",
        "GRU": PRED / "gru_test_predictions.npy",
    }

    out = {}

    for model, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        arr = np.load(path)

        if arr.shape != (4584, 3):
            raise ValueError(
                f"{model} prediction shape is {arr.shape}; expected (4584,3)."
            )

        out[model] = arr.astype(np.float32)

    return out


def normalized_input(X):
    if not NORM_FILE.exists():
        raise FileNotFoundError(NORM_FILE)

    d = np.load(NORM_FILE)

    mean = d["mean"]
    std = d["std"]

    # Expected shape is [1,1,4], but make this robust.
    mean = np.asarray(mean).reshape(1, 1, -1)
    std = np.asarray(std).reshape(1, 1, -1)

    if mean.shape[-1] != 4 or std.shape[-1] != 4:
        raise ValueError(
            f"Unexpected normalization shape: mean={mean.shape}, std={std.shape}"
        )

    return (X - mean) / np.maximum(std, 1e-8)


def load_models():
    models = {}

    # XGBoost: one model per horizon
    xgb_models = {}
    for h in HORIZONS:
        p = MODELS / f"xgboost_{h}.json"
        if not p.exists():
            raise FileNotFoundError(p)

        model = xgb.XGBRegressor()
        model.load_model(p)
        xgb_models[h] = model

    models["XGBoost"] = xgb_models

    # Neural models
    lstm_path = MODELS / "lstm_forecaster.keras"
    gru_path = MODELS / "gru_forecaster.keras"

    if not lstm_path.exists():
        raise FileNotFoundError(lstm_path)

    if not gru_path.exists():
        raise FileNotFoundError(gru_path)

    models["LSTM"] = load_model(lstm_path)
    models["GRU"] = load_model(gru_path)

    return models


def predict_all(models, X):
    """
    X is raw sensor sequence [N,20,4].
    """
    flat = X.reshape(len(X), -1)

    pred = {}

    # XGBoost
    xgb_pred = np.zeros((len(X), 3), dtype=np.float32)
    for i, h in enumerate(HORIZONS):
        xgb_pred[:, i] = models["XGBoost"][h].predict(flat)

    pred["XGBoost"] = xgb_pred

    # Neural networks
    Xn = normalized_input(X)

    pred["LSTM"] = models["LSTM"].predict(
        Xn, verbose=0
    ).astype(np.float32)

    pred["GRU"] = models["GRU"].predict(
        Xn, verbose=0
    ).astype(np.float32)

    return pred


# ==============================================================
# 1. CONDITION-WISE GENERALIZATION
# ==============================================================

def condition_analysis(X, y, run_ids, predictions, metadata):
    aligned = merge_metadata(run_ids, metadata)

    condition_specs = []

    if "mu_surface" in aligned:
        condition_specs.append(("surface", "mu_surface"))

    if "target_speed_kmh" in aligned:
        condition_specs.append(("speed_kmh", "target_speed_kmh"))

    if "maneuver" in aligned:
        condition_specs.append(("maneuver", "maneuver"))

    if "amplitude_deg" in aligned:
        condition_specs.append(("amplitude_deg", "amplitude_deg"))

    rows = []

    print("\n" + "=" * 70)
    print("CONDITION-WISE GENERALIZATION")
    print("=" * 70)

    for label, column in condition_specs:
        print(f"\n--- {label} ---")

        for value in sorted(aligned[column].dropna().unique(),
                             key=lambda z: str(z)):

            mask = aligned[column].values == value

            if mask.sum() < 10:
                continue

            for model in MODELS_TO_USE:
                for j, h in enumerate(HORIZONS):
                    rows.append(
                        metric_row(
                            model=model,
                            horizon=h,
                            y_true=y[mask, j],
                            y_pred=predictions[model][mask, j],
                            subset_name=label,
                            subset_value=value,
                        )
                    )

            print(
                f"{label}={value}: "
                f"{mask.sum()} sequences"
            )

    df = pd.DataFrame(rows)

    path = OUT_T / "condition_wise_generalization.csv"
    df.to_csv(path, index=False)

    return df


# ==============================================================
# 2. NOISE ROBUSTNESS
# ==============================================================

def add_noise(X, percentage, rng):
    """
    Feature-wise perturbation relative to each feature's empirical
    standard deviation in the test input.

    percentage=0.10 means sigma_noise = 0.10 * feature_std.
    """
    if percentage == 0:
        return X.copy()

    scale = X.std(axis=(0, 1), keepdims=True)
    noise = rng.normal(
        0.0,
        percentage * scale,
        size=X.shape
    ).astype(np.float32)

    return X + noise


def noise_robustness(X, y, models):
    levels = [0, 5, 10, 20, 30]
    rng = np.random.default_rng(42)

    rows = []

    print("\n" + "=" * 70)
    print("SENSOR-NOISE ROBUSTNESS")
    print("=" * 70)

    for level in levels:
        print(f"\nNoise level: {level}%")

        X_noisy = add_noise(
            X,
            level / 100.0,
            rng
        )

        pred = predict_all(models, X_noisy)

        for model in MODELS_TO_USE:
            for j, h in enumerate(HORIZONS):

                row = metric_row(
                    model=model,
                    horizon=h,
                    y_true=y[:, j],
                    y_pred=pred[model][:, j],
                    subset_name="sensor_noise",
                    subset_value=f"{level}%",
                    noise_pct=level,
                )

                rows.append(row)

                print(
                    f"{model:8s} {h}: "
                    f"MAE={row['MAE']:.4f}, "
                    f"RMSE={row['RMSE']:.4f}, "
                    f"R2={row['R2']:.4f}"
                )

    df = pd.DataFrame(rows)

    # Calculate degradation relative to 0%.
    df["MAE_degradation_percent"] = np.nan
    df["RMSE_degradation_percent"] = np.nan
    df["R2_drop"] = np.nan

    for model in MODELS_TO_USE:
        for h in HORIZONS:
            base = df[
                (df.model == model) &
                (df.horizon == h) &
                (df.noise_percent == 0)
            ].iloc[0]

            mask = (
                (df.model == model) &
                (df.horizon == h)
            )

            df.loc[mask, "MAE_degradation_percent"] = (
                (df.loc[mask, "MAE"] - base["MAE"])
                / max(abs(base["MAE"]), 1e-12)
                * 100
            )

            df.loc[mask, "RMSE_degradation_percent"] = (
                (df.loc[mask, "RMSE"] - base["RMSE"])
                / max(abs(base["RMSE"]), 1e-12)
                * 100
            )

            df.loc[mask, "R2_drop"] = (
                base["R2"] - df.loc[mask, "R2"]
            )

    path = OUT_T / "sensor_noise_robustness.csv"
    df.to_csv(path, index=False)

    return df


# ==============================================================
# 3. PLOTS
# ==============================================================

def plot_condition_heatmaps(condition_df):
    for subset in condition_df["subset"].unique():

        d = condition_df[condition_df["subset"] == subset]

        for horizon in HORIZONS:
            p = d[d["horizon"] == horizon]

            if p.empty:
                continue

            pivot = p.pivot_table(
                index="condition",
                columns="model",
                values="MAE",
                aggfunc="mean"
            )

            if pivot.empty:
                continue

            ax = pivot.plot(
                kind="bar",
                figsize=(9, 5)
            )

            ax.set_title(
                f"Forecast MAE by {subset} — {horizon}"
            )
            ax.set_ylabel("MAE")
            ax.set_xlabel(subset)
            plt.xticks(rotation=0)
            plt.tight_layout()

            filename = (
                f"generalization_mae_"
                f"{subset}_{horizon.replace('.', '_')}.png"
            )

            plt.savefig(OUT_F / filename, dpi=300)
            plt.close()


def plot_noise(noise_df):
    for metric in ["MAE", "RMSE", "R2"]:

        for h in HORIZONS:

            plt.figure(figsize=(8, 5))

            for model in MODELS_TO_USE:
                d = noise_df[
                    (noise_df.model == model) &
                    (noise_df.horizon == h)
                ].sort_values("noise_percent")

                plt.plot(
                    d["noise_percent"],
                    d[metric],
                    marker="o",
                    label=model
                )

            plt.xlabel("Sensor noise level (%)")
            plt.ylabel(metric)
            plt.title(
                f"{metric} vs Sensor Noise — {h}"
            )
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()

            filename = (
                f"noise_robustness_{metric}_{h.replace('.', '_')}.png"
            )

            plt.savefig(OUT_F / filename, dpi=300)
            plt.close()


# ==============================================================
# 4. SUMMARY TABLE
# ==============================================================

def build_summary(condition_df, noise_df):
    rows = []

    for model in MODELS_TO_USE:
        for h in HORIZONS:

            clean = noise_df[
                (noise_df.model == model) &
                (noise_df.horizon == h) &
                (noise_df.noise_percent == 0)
            ].iloc[0]

            noisy20 = noise_df[
                (noise_df.model == model) &
                (noise_df.horizon == h) &
                (noise_df.noise_percent == 20)
            ].iloc[0]

            rows.append({
                "model": model,
                "horizon": h,
                "clean_MAE": clean["MAE"],
                "noise20_MAE": noisy20["MAE"],
                "MAE_degradation_20pct": noisy20[
                    "MAE_degradation_percent"
                ],
                "clean_RMSE": clean["RMSE"],
                "noise20_RMSE": noisy20["RMSE"],
                "RMSE_degradation_20pct": noisy20[
                    "RMSE_degradation_percent"
                ],
                "clean_R2": clean["R2"],
                "noise20_R2": noisy20["R2"],
                "R2_drop_20pct": noisy20["R2_drop"],
            })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUT_T / "robustness_summary.csv",
        index=False
    )

    return summary


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 70)
    print("ROBUSTNESS AND GENERALIZATION EVALUATION")
    print("=" * 70)

    X, y, run_ids = load_test()
    metadata = load_metadata()
    clean_predictions = load_clean_predictions()
    models = load_models()

    print("\nTest:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("Run_ID:", run_ids.shape)
    print("Unique runs:", len(np.unique(run_ids)))

    # Condition-wise analysis uses the already generated clean predictions.
    condition_df = condition_analysis(
        X,
        y,
        run_ids,
        clean_predictions,
        metadata
    )

    # Noise robustness requires model inference.
    noise_df = noise_robustness(
        X,
        y,
        models
    )

    plot_condition_heatmaps(condition_df)
    plot_noise(noise_df)

    summary = build_summary(
        condition_df,
        noise_df
    )

    # Human-readable report.
    report = OUT_T / "robustness_generalization_report.txt"

    with open(report, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("ROBUSTNESS AND GENERALIZATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            "This evaluation measures whether the final models retain "
            "forecasting performance across operating conditions and "
            "under sensor perturbations.\n\n"
        )

        f.write(
            "The analysis is based on the held-out test sequences and "
            "does not retrain or alter the final models.\n\n"
        )

        f.write("20% SENSOR-NOISE SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")

        f.write("CONDITION-WISE RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(condition_df.to_string(index=False))
        f.write("\n")

    print("\n" + "=" * 70)
    print("ROBUSTNESS EVALUATION COMPLETE")
    print("=" * 70)

    print("\nTABLES:")
    print(OUT_T / "condition_wise_generalization.csv")
    print(OUT_T / "sensor_noise_robustness.csv")
    print(OUT_T / "robustness_summary.csv")
    print(OUT_T / "robustness_generalization_report.txt")

    print("\nFIGURES:")
    print(OUT_F)

    print(
        "\nIMPORTANT: Do not claim superiority from these tests until "
        "the generated values have been inspected. Report the actual "
        "condition-wise and noise-degradation numbers."
    )


if __name__ == "__main__":
    main()
