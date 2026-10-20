"""Illustrative low-cost retrofit sensor-noise model.

Noise magnitudes are assumptions for simulation and are not calibrated to a
specific hardware datasheet. Each Run_ID receives independent constant bias.
"""
import numpy as np
import pandas as pd


def add_sensor_noise(ground_truth: pd.DataFrame, rng=None) -> pd.DataFrame:
    """Return the measured channels, applying independent noise per run."""
    required = {
        "Run_ID", "Time", "yaw_rate", "lateral_accel",
        "target_speed_kmh", "steer_true", "maneuver", "mu_surface",
    }
    missing = required.difference(ground_truth.columns)
    if missing:
        raise ValueError(f"Ground truth missing columns: {sorted(missing)}")
    if rng is None:
        rng = np.random.default_rng(7)

    measured = ground_truth[
        ["Run_ID", "Time", "maneuver", "mu_surface"]
    ].copy()
    for col in ["yaw_rate_meas", "lateral_accel_meas", "speed_meas", "steering_angle_meas"]:
        measured[col] = np.nan

    for _, run in ground_truth.groupby("Run_ID", sort=False):
        idx = run.index
        n = len(run)
        yaw_bias = rng.normal(0, np.radians(0.1))
        accel_bias = rng.normal(0, 0.05)
        measured.loc[idx, "yaw_rate_meas"] = (
            run["yaw_rate"].to_numpy()
            + rng.normal(0, np.radians(0.5), n)
            + yaw_bias
        )
        measured.loc[idx, "lateral_accel_meas"] = (
            run["lateral_accel"].to_numpy()
            + rng.normal(0, 0.15, n)
            + accel_bias
        )
        speed = run["target_speed_kmh"].to_numpy() / 3.6
        speed = speed + rng.normal(0, 0.1, n)
        measured.loc[idx, "speed_meas"] = np.round(speed / 0.28) * 0.28
        measured.loc[idx, "steering_angle_meas"] = (
            run["steer_true"].to_numpy()
            + rng.normal(0, np.radians(0.2), n)
        )

    return measured.reset_index(drop=True)


def main() -> None:
    """Module is intended to be imported by the dataset generator."""
    print("Use add_sensor_noise(ground_truth, rng) from generate_dataset.py.")
