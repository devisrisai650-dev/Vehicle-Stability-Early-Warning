
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FINAL RESULTS ANALYSIS
# Run from:
# vehicle-stability-early-warning
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

HORIZONS = ["0.5s", "1.0s", "2.0s"]
FORECAST_MODELS = ["Persistence", "XGBoost", "LSTM", "GRU"]
WARNING_MODELS = ["XGBoost", "LSTM", "GRU"]


def locate_csv(*names):
    for name in names:
        p = TABLES / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find any of:\n" +
        "\n".join(str(TABLES / n) for n in names)
    )


def load_metric_file(model, *names):
    p = locate_csv(*names)
    df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    if "horizon" not in df.columns:
        raise ValueError(f"{p} does not contain a 'horizon' column.")
    df["horizon"] = df["horizon"].astype(str)
    df["model"] = model
    return df


# ------------------------------------------------------------
# 1. Forecasting metrics
# ------------------------------------------------------------
persistence = load_metric_file(
    "Persistence",
    "persistence_metrics.csv",
)

xgb = load_metric_file(
    "XGBoost",
    "xgboost_test_metrics.csv",
)

lstm = load_metric_file(
    "LSTM",
    "lstm_test_metrics.csv",
)

gru = load_metric_file(
    "GRU",
    "gru_test_metrics.csv",
    "gru_metrics.csv",
)

required = ["MAE", "RMSE", "R2"]

for name, df in [
    ("Persistence", persistence),
    ("XGBoost", xgb),
    ("LSTM", lstm),
    ("GRU", gru),
]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} metric file is missing columns: {missing}"
        )

forecast = pd.concat(
    [persistence, xgb, lstm, gru],
    ignore_index=True
)

forecast = forecast[
    forecast["horizon"].isin(HORIZONS)
][["model", "horizon", "MAE", "RMSE", "R2"]]

forecast["horizon_order"] = forecast["horizon"].map(
    {h: i for i, h in enumerate(HORIZONS)}
)
forecast = forecast.sort_values(
    ["horizon_order", "model"]
).drop(columns="horizon_order")

forecast.to_csv(
    TABLES / "final_forecasting_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 2. Final calibrated early-warning metrics
# ------------------------------------------------------------
event_path = locate_csv(
    "final_calibrated_event_lead_time_metrics.csv"
)

event = pd.read_csv(event_path)
event.columns = [str(c).strip() for c in event.columns]

required_event = [
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

missing = [c for c in required_event if c not in event.columns]
if missing:
    raise ValueError(
        f"Final event file is missing columns: {missing}"
    )

event["horizon"] = event["horizon"].astype(str)
event = event[
    event["model"].isin(WARNING_MODELS)
    & event["horizon"].isin(HORIZONS)
].copy()

event["horizon_order"] = event["horizon"].map(
    {h: i for i, h in enumerate(HORIZONS)}
)
event = event.sort_values(
    ["horizon_order", "model"]
).drop(columns="horizon_order")

event.to_csv(
    TABLES / "final_early_warning_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 3. Combined final table
# ------------------------------------------------------------
combined = event.merge(
    forecast,
    on=["model", "horizon"],
    how="left"
)

combined_columns = [
    "model",
    "horizon",
    "MAE",
    "RMSE",
    "R2",
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

combined = combined[combined_columns]

combined.to_csv(
    TABLES / "final_model_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 4. Metric-specific comparisons
#
# IMPORTANT:
# This table identifies the model with the best value for
# each individual metric. It is NOT an overall ranking.
# ------------------------------------------------------------
rows = []

for horizon in HORIZONS:
    d_forecast = forecast[
        forecast["horizon"] == horizon
    ].copy()

    d_event = event[
        event["horizon"] == horizon
    ].copy()

    if d_forecast.empty:
        continue

    row = {
        "horizon": horizon,

        "lowest_MAE_model":
            d_forecast.loc[d_forecast["MAE"].idxmin(), "model"],
        "lowest_MAE":
            d_forecast["MAE"].min(),

        "lowest_RMSE_model":
            d_forecast.loc[d_forecast["RMSE"].idxmin(), "model"],
        "lowest_RMSE":
            d_forecast["RMSE"].min(),

        "highest_R2_model":
            d_forecast.loc[d_forecast["R2"].idxmax(), "model"],
        "highest_R2":
            d_forecast["R2"].max(),
    }

    if not d_event.empty:
        row.update({
            "highest_detection_model":
                d_event.loc[
                    d_event["event_detection_rate"].idxmax(),
                    "model"
                ],
            "highest_detection_rate":
                d_event["event_detection_rate"].max(),

            "highest_mean_lead_model":
                d_event.loc[
                    d_event["mean_lead_time_s"].idxmax(),
                    "model"
                ],
            "highest_mean_lead_time_s":
                d_event["mean_lead_time_s"].max(),

            "lowest_false_warning_model":
                d_event.loc[
                    d_event["false_warning_episodes"].idxmin(),
                    "model"
                ],
            "lowest_false_warning_episodes":
                d_event["false_warning_episodes"].min(),
        })

    rows.append(row)

metric_comparison = pd.DataFrame(rows)

metric_comparison.to_csv(
    TABLES / "metric_specific_comparisons.csv",
    index=False
)


# ------------------------------------------------------------
# 5. Plot helper
# ------------------------------------------------------------
def plot_forecast_metric(metric, ylabel, filename, title):
    plt.figure(figsize=(8, 5))

    for model in FORECAST_MODELS:
        d = forecast[
            forecast["model"] == model
        ].copy()

        d["order"] = d["horizon"].map(
            {h: i for i, h in enumerate(HORIZONS)}
        )
        d = d.sort_values("order")

        plt.plot(
            d["horizon"],
            d[metric],
            marker="o",
            linewidth=2,
            label=model
        )

    plt.xlabel("Forecast horizon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_warning_metric(metric, ylabel, filename, title):
    plt.figure(figsize=(8, 5))

    for model in WARNING_MODELS:
        d = event[
            event["model"] == model
        ].copy()

        d["order"] = d["horizon"].map(
            {h: i for i, h in enumerate(HORIZONS)}
        )
        d = d.sort_values("order")

        plt.plot(
            d["horizon"],
            d[metric],
            marker="o",
            linewidth=2,
            label=model
        )

    plt.xlabel("Forecast horizon")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 6. Forecasting figures
# ------------------------------------------------------------
plot_forecast_metric(
    "MAE",
    "MAE",
    "forecast_mae_vs_horizon.png",
    "Forecasting MAE vs Horizon"
)

plot_forecast_metric(
    "RMSE",
    "RMSE",
    "forecast_rmse_vs_horizon.png",
    "Forecasting RMSE vs Horizon"
)

plot_forecast_metric(
    "R2",
    "R²",
    "forecast_r2_vs_horizon.png",
    "Forecasting R² vs Horizon"
)


# ------------------------------------------------------------
# 7. Early-warning figures
# ------------------------------------------------------------
plot_warning_metric(
    "event_detection_rate",
    "Event detection rate",
    "event_detection_rate_vs_horizon.png",
    "Event Detection Rate vs Horizon"
)

plot_warning_metric(
    "mean_lead_time_s",
    "Mean lead time (s)",
    "mean_lead_time_vs_horizon.png",
    "Mean Lead Time vs Horizon"
)

plot_warning_metric(
    "false_warning_episodes",
    "False-warning episodes",
    "false_warning_episodes_vs_horizon.png",
    "False-Warning Episodes vs Horizon"
)


# ------------------------------------------------------------
# 8. Print final results
# ------------------------------------------------------------
print("=" * 70)
print("FINAL RESULTS ANALYSIS COMPLETE")
print("=" * 70)

print("\nFORECASTING RESULTS")
print("-" * 70)
print(forecast.to_string(index=False))

print("\nFINAL CALIBRATED EARLY-WARNING RESULTS")
print("-" * 70)
print(event.to_string(index=False))

print("\nCOMBINED MODEL COMPARISON")
print("-" * 70)
print(combined.to_string(index=False))

print("\nMETRIC-SPECIFIC COMPARISONS")
print("-" * 70)
print(metric_comparison.to_string(index=False))

print("\nOUTPUT TABLES")
print("-" * 70)
for name in [
    "final_forecasting_comparison.csv",
    "final_early_warning_comparison.csv",
    "final_model_comparison.csv",
    "metric_specific_comparisons.csv",
]:
    print(TABLES / name)

print("\nOUTPUT FIGURES")
print("-" * 70)
for name in [
    "forecast_mae_vs_horizon.png",
    "forecast_rmse_vs_horizon.png",
    "forecast_r2_vs_horizon.png",
    "event_detection_rate_vs_horizon.png",
    "mean_lead_time_vs_horizon.png",
    "false_warning_episodes_vs_horizon.png",
]:
    print(FIGURES / name)

print("\nNEXT STEP:")
print("Use final_model_comparison.csv and the six PNG figures")
print("for the Results section of the paper/report.")
