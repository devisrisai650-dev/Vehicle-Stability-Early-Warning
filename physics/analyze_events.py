from pathlib import Path

import numpy as np
import pandas as pd


# ==============================================================
# Paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ground_truth_dynamics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)


# Dataset sampling rate
DT = 0.05

# Minimum number of consecutive negative-margin samples
MIN_EVENT_SAMPLES = 3


# ==============================================================
# Detect instability events
# ==============================================================

def detect_events(run_df):
    """
    Detect contiguous negative stability-margin segments.

    A segment becomes a valid instability event only if it contains
    at least MIN_EVENT_SAMPLES consecutive samples.
    """

    run_df = run_df.sort_values("Time").reset_index(drop=True)

    negative = (
        run_df["stability_margin"].to_numpy() < 0
    )

    events = []

    start = None

    for i, is_negative in enumerate(negative):

        if is_negative and start is None:
            start = i

        elif not is_negative and start is not None:

            end = i - 1

            length = end - start + 1

            if length >= MIN_EVENT_SAMPLES:
                events.append(
                    (start, end)
                )

            start = None

    # Handle event continuing until final sample
    if start is not None:

        end = len(run_df) - 1

        length = end - start + 1

        if length >= MIN_EVENT_SAMPLES:
            events.append(
                (start, end)
            )

    return events


# ==============================================================
# Main analysis
# ==============================================================

def main():

    print("=" * 70)
    print("EVENT-LEVEL VEHICLE STABILITY ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    all_events = []

    for run_id, run_df in df.groupby("Run_ID"):

        run_df = run_df.sort_values("Time").reset_index(drop=True)

        event_segments = detect_events(run_df)

        for event_number, (start, end) in enumerate(
            event_segments,
            start=1
        ):

            event = run_df.iloc[start:end + 1]

            onset_row = event.iloc[0]
            final_row = event.iloc[-1]

            min_margin_idx = event[
                "stability_margin"
            ].idxmin()

            min_margin_row = run_df.loc[min_margin_idx]

            n_samples = len(event)

            all_events.append(
                {
                    "Run_ID": run_id,
                    "Event_ID": event_number,

                    "maneuver":
                        onset_row["maneuver"],

                    "mu_surface":
                        onset_row["mu_surface"],

                    "target_speed_kmh":
                        onset_row["target_speed_kmh"],

                    "amplitude_deg":
                        onset_row["amplitude_deg"],

                    # Event timing
                    "onset_time_s":
                        onset_row["Time"],

                    "end_time_s":
                        final_row["Time"],

                    "n_samples":
                        n_samples,

                    "event_duration_s":
                        (n_samples - 1) * DT,

                    # Margin severity
                    "min_stability_margin":
                        event["stability_margin"].min(),

                    "mean_stability_margin":
                        event["stability_margin"].mean(),

                    # Vehicle-state severity
                    "max_abs_beta_rad":
                        event["beta"].abs().max(),

                    "max_abs_yaw_rate_rad_s":
                        event["yaw_rate"].abs().max(),

                    "max_abs_rear_slip_rad":
                        event["alpha_r"].abs().max(),

                    # Time at which maximum severity occurred
                    "min_margin_time_s":
                        min_margin_row["Time"],
                }
            )

    events = pd.DataFrame(all_events)

    # ==========================================================
    # Create output directory
    # ==========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "stability_events.csv"
    )

    events.to_csv(
        output_path,
        index=False
    )

    # ==========================================================
    # Summary
    # ==========================================================

    print("\n1. EVENT SUMMARY")
    print("-" * 70)

    print(
        "Total instability events:",
        len(events)
    )

    print(
        "Runs containing events:",
        events["Run_ID"].nunique()
    )

    print(
        "Total runs:",
        df["Run_ID"].nunique()
    )

    print(
        "Average events per unstable run:",
        events.groupby("Run_ID").size().mean()
        if len(events)
        else 0
    )

    # ==========================================================
    # Event duration
    # ==========================================================

    print("\n2. EVENT DURATION")
    print("-" * 70)

    if len(events):

        print(
            events["event_duration_s"].describe()
        )

    # ==========================================================
    # Severity
    # ==========================================================

    print("\n3. EVENT SEVERITY")
    print("-" * 70)

    if len(events):

        print(
            "Minimum margin:",
            events["min_stability_margin"].min()
        )

        print(
            "Mean event minimum margin:",
            events["min_stability_margin"].mean()
        )

        print(
            "Maximum |beta|:",
            events["max_abs_beta_rad"].max()
        )

        print(
            "Maximum |yaw rate|:",
            events["max_abs_yaw_rate_rad_s"].max()
        )

    # ==========================================================
    # By road surface
    # ==========================================================

    print("\n4. EVENTS BY ROAD SURFACE")
    print("-" * 70)

    if len(events):

        surface_summary = (
            events.groupby("mu_surface")
            .agg(
                events=("Event_ID", "count"),
                runs=("Run_ID", "nunique"),
                mean_min_margin=(
                    "min_stability_margin",
                    "mean"
                ),
                mean_duration=(
                    "event_duration_s",
                    "mean"
                ),
            )
            .reset_index()
        )

        print(
            surface_summary.to_string(
                index=False
            )
        )

    # ==========================================================
    # By maneuver
    # ==========================================================

    print("\n5. EVENTS BY MANEUVER")
    print("-" * 70)

    if len(events):

        maneuver_summary = (
            events.groupby("maneuver")
            .agg(
                events=("Event_ID", "count"),
                runs=("Run_ID", "nunique"),
                mean_min_margin=(
                    "min_stability_margin",
                    "mean"
                ),
                mean_duration=(
                    "event_duration_s",
                    "mean"
                ),
            )
            .reset_index()
        )

        print(
            maneuver_summary.to_string(
                index=False
            )
        )

    # ==========================================================
    # By speed
    # ==========================================================

    print("\n6. EVENTS BY SPEED")
    print("-" * 70)

    if len(events):

        speed_summary = (
            events.groupby("target_speed_kmh")
            .agg(
                events=("Event_ID", "count"),
                runs=("Run_ID", "nunique"),
                mean_min_margin=(
                    "min_stability_margin",
                    "mean"
                ),
            )
            .reset_index()
        )

        print(
            speed_summary.to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)

    print(
        f"Saved event table to:\n{output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()