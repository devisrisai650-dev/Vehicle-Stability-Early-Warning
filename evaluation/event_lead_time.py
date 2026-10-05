from pathlib import Path

import numpy as np
import pandas as pd


# ==============================================================
# PROJECT PATHS
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PREDICTION_DIR = (
    PROCESSED_DIR / "predictions"
)

RESULTS_DIR = PROJECT_ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"

TEST_SEQUENCE_PATH = (
    PROCESSED_DIR
    / "test_sequences.npz"
)

GT_PATH = (
    DATA_DIR
    / "raw"
    / "ground_truth_dynamics.csv"
)


# ==============================================================
# MODELS
# ==============================================================

MODELS = {

    "XGBoost":
        PREDICTION_DIR
        / "xgboost_test_predictions.npy",

    "LSTM":
        PREDICTION_DIR
        / "lstm_test_predictions.npy",

    "GRU":
        PREDICTION_DIR
        / "gru_test_predictions.npy",
}


# ==============================================================
# HORIZONS
# ==============================================================

HORIZONS = {

    "0.5s": 0,

    "1.0s": 1,

    "2.0s": 2,
}


# ==============================================================
# WARNING CONFIGURATION
# ==============================================================

# Physics-defined instability threshold.
#
# Stability margin < 0
#     → predicted unstable
#
# We will later calibrate this threshold using validation data.
THRESHOLD = 0.0


# Dataset sampling frequency:
#
# 20 Hz
# → 0.05 s/sample
#
# Requiring two consecutive warning samples means
# approximately 0.10 s of sustained warning.
CONSECUTIVE_WARNINGS = 2


# If two warning portions are separated by less than
# this amount of time, consider them part of the same
# warning episode.
WARNING_GAP_S = 0.20


# A warning is considered relevant to an instability
# event only if it begins within this amount of time
# before the event onset.
MAX_WARNING_WINDOW_S = 5.0


# ==============================================================
# LOAD TEST SEQUENCES
# ==============================================================

def load_test_data():

    """
    Load the test sequence file.

    Expected arrays:

        X
        y
        Run_ID
        Time

    """

    data = np.load(
        TEST_SEQUENCE_PATH,
        allow_pickle=True
    )

    print(
        "\nTEST FILE CONTENTS"
    )

    print("-" * 70)

    for key in data.files:

        value = data[key]

        if hasattr(value, "shape"):

            print(
                f"{key}: {value.shape}"
            )

        else:

            print(
                f"{key}: {type(value)}"
            )

    required_keys = [
        "X",
        "y",
        "Run_ID",
        "Time",
    ]

    missing = [
        key
        for key in required_keys
        if key not in data.files
    ]

    if missing:

        raise KeyError(
            "Missing required keys in "
            f"test_sequences.npz: {missing}"
        )

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.float32
    )

    run_ids = np.asarray(
        data["Run_ID"]
    ).astype(int)

    times = np.asarray(
        data["Time"]
    ).astype(float)

    return (
        X,
        y,
        run_ids,
        times,
    )


# ==============================================================
# LOAD GROUND TRUTH
# ==============================================================

def load_ground_truth():

    """
    Load complete ground-truth dynamics.
    """

    gt = pd.read_csv(
        GT_PATH
    )

    required_columns = [
        "Run_ID",
        "Time",
        "instability_flag",
        "stability_margin",
        "beta",
        "yaw_rate",
    ]

    missing = [
        column
        for column in required_columns
        if column not in gt.columns
    ]

    if missing:

        raise ValueError(
            "Ground-truth file is missing "
            f"columns: {missing}"
        )

    gt["Run_ID"] = (
        gt["Run_ID"]
        .astype(int)
    )

    gt["Time"] = (
        gt["Time"]
        .astype(float)
    )

    gt["instability_flag"] = (
        gt["instability_flag"]
        .astype(bool)
    )

    return gt


# ==============================================================
# BUILD TEST TIMELINE
# ==============================================================

def build_test_timeline(
    run_ids,
    times,
):

    """
    Create a timeline connecting every prediction
    to its Run_ID and physical timestamp.

    The row/sample index is preserved so that prediction
    arrays can be mapped back to the correct run.
    """

    timeline = pd.DataFrame({

        "Run_ID":
            np.asarray(run_ids)
            .astype(int),

        "Time":
            np.asarray(times)
            .astype(float),

    })

    timeline = timeline.reset_index()

    timeline = timeline.rename(
        columns={
            "index": "sample_index"
        }
    )

    return timeline


# ==============================================================
# EXTRACT GROUND-TRUTH INSTABILITY EVENTS
# ==============================================================

def extract_events(gt):

    """
    Extract continuous instability regions.

    Each continuous instability region is treated as
    one ground-truth instability event.
    """

    events = []

    global_event_id = 0

    for run_id, run_df in gt.groupby(
        "Run_ID"
    ):

        run_df = (
            run_df
            .sort_values("Time")
            .reset_index(drop=True)
        )

        flag = (
            run_df["instability_flag"]
            .astype(bool)
            .to_numpy()
        )

        i = 0

        while i < len(run_df):

            # --------------------------------------------------
            # Find event start
            # --------------------------------------------------

            if not flag[i]:

                i += 1

                continue

            start = i

            # --------------------------------------------------
            # Find event end
            # --------------------------------------------------

            while (
                i + 1 < len(run_df)
                and flag[i + 1]
            ):

                i += 1

            end = i

            event = (
                run_df
                .iloc[start:end + 1]
            )

            global_event_id += 1

            events.append({

                "Run_ID":
                    int(run_id),

                "event_id":
                    global_event_id,

                "onset_time":
                    float(
                        event["Time"]
                        .iloc[0]
                    ),

                "end_time":
                    float(
                        event["Time"]
                        .iloc[-1]
                    ),

                "duration_s":
                    float(
                        event["Time"]
                        .iloc[-1]
                        -
                        event["Time"]
                        .iloc[0]
                    ),

                "minimum_margin":
                    float(
                        event[
                            "stability_margin"
                        ].min()
                    ),

                "max_abs_beta":
                    float(
                        event[
                            "beta"
                        ].abs().max()
                    ),

                "max_abs_yaw_rate":
                    float(
                        event[
                            "yaw_rate"
                        ].abs().max()
                    ),
            })

            i += 1

    return pd.DataFrame(
        events
    )


# ==============================================================
# FIND WARNING EPISODES
# ==============================================================

def find_warning_episodes(
    timeline,
    predictions,
    threshold,
):

    """
    Convert sample-level predictions into
    independent warning episodes.

    Step 1:
        predicted margin < threshold

    Step 2:
        require CONSECUTIVE_WARNINGS

    Step 3:
        group nearby warning samples into
        one warning episode.
    """

    episodes = []

    for run_id, run_df in timeline.groupby(
        "Run_ID"
    ):

        run_df = (
            run_df
            .sort_values("Time")
            .reset_index(drop=True)
        )

        indices = (
            run_df[
                "sample_index"
            ]
            .to_numpy()
        )

        times = (
            run_df["Time"]
            .to_numpy()
        )

        pred = predictions[
            indices
        ]

        # ------------------------------------------------------
        # Step 1: threshold
        # ------------------------------------------------------

        is_warning = (
            pred < threshold
        )

        # ------------------------------------------------------
        # Step 2: require sustained warning
        # ------------------------------------------------------

        sustained = np.zeros(
            len(is_warning),
            dtype=bool
        )

        run_length = 0

        for i, warning in enumerate(
            is_warning
        ):

            if warning:

                run_length += 1

            else:

                run_length = 0

            if (
                run_length
                >= CONSECUTIVE_WARNINGS
            ):

                start_idx = (
                    i
                    - CONSECUTIVE_WARNINGS
                    + 1
                )

                sustained[
                    start_idx:
                    i + 1
                ] = True

        # ------------------------------------------------------
        # Step 3: collect sustained-warning samples
        # ------------------------------------------------------

        warning_indices = np.where(
            sustained
        )[0]

        if len(warning_indices) == 0:

            continue

        # ------------------------------------------------------
        # Group sustained warnings
        # into warning episodes
        # ------------------------------------------------------

        episode_start = (
            warning_indices[0]
        )

        previous = (
            warning_indices[0]
        )

        for current in (
            warning_indices[1:]
        ):

            time_gap = (
                times[current]
                -
                times[previous]
            )

            # A sufficiently large gap means
            # a new warning episode.
            if (
                time_gap
                > WARNING_GAP_S
            ):

                episode_indices = (
                    np.arange(
                        episode_start,
                        previous + 1
                    )
                )

                episode_predictions = (
                    pred[
                        episode_indices
                    ]
                )

                episodes.append({

                    "Run_ID":
                        int(run_id),

                    "warning_start":
                        float(
                            times[
                                episode_start
                            ]
                        ),

                    "warning_end":
                        float(
                            times[
                                previous
                            ]
                        ),

                    "duration_s":
                        float(
                            times[
                                previous
                            ]
                            -
                            times[
                                episode_start
                            ]
                        ),

                    "minimum_prediction":
                        float(
                            episode_predictions
                            .min()
                        ),
                })

                episode_start = current

            previous = current

        # ------------------------------------------------------
        # Save final warning episode
        # ------------------------------------------------------

        episode_indices = (
            np.arange(
                episode_start,
                previous + 1
            )
        )

        episode_predictions = (
            pred[
                episode_indices
            ]
        )

        episodes.append({

            "Run_ID":
                int(run_id),

            "warning_start":
                float(
                    times[
                        episode_start
                    ]
                ),

            "warning_end":
                float(
                    times[
                        previous
                    ]
                ),

            "duration_s":
                float(
                    times[
                        previous
                    ]
                    -
                    times[
                        episode_start
                    ]
                ),

            "minimum_prediction":
                float(
                    episode_predictions
                    .min()
                ),
        })

    if len(episodes) == 0:

        return pd.DataFrame(
            columns=[
                "Run_ID",
                "warning_start",
                "warning_end",
                "duration_s",
                "minimum_prediction",
            ]
        )

    return pd.DataFrame(
        episodes
    )


# ==============================================================
# MATCH WARNING EPISODES TO EVENTS
# ==============================================================

def calculate_event_lead_times(
    events,
    warning_episodes,
):

    """
    For every actual instability event, identify
    the earliest relevant warning episode.

    Lead time:

        event onset - warning start
    """

    results = []

    for _, event in events.iterrows():

        run_id = int(
            event["Run_ID"]
        )

        onset = float(
            event["onset_time"]
        )

        candidate_warnings = (
            warning_episodes[
                warning_episodes[
                    "Run_ID"
                ] == run_id
            ]
            .copy()
        )

        # Warning must start before event onset.
        candidate_warnings = (
            candidate_warnings[
                candidate_warnings[
                    "warning_start"
                ] <= onset
            ]
        )

        # Warning must be close enough to the
        # actual event to be considered relevant.
        candidate_warnings = (
            candidate_warnings[
                candidate_warnings[
                    "warning_start"
                ]
                >=
                onset
                -
                MAX_WARNING_WINDOW_S
            ]
        )

        if len(
            candidate_warnings
        ) == 0:

            results.append({

                **event.to_dict(),

                "warning_issued":
                    False,

                "first_warning_time":
                    np.nan,

                "lead_time_s":
                    np.nan,

            })

            continue

        # ------------------------------------------------------
        # Earliest relevant warning
        # ------------------------------------------------------

        warning = (
            candidate_warnings
            .sort_values(
                "warning_start"
            )
            .iloc[0]
        )

        warning_time = float(
            warning[
                "warning_start"
            ]
        )

        lead_time = (
            onset
            -
            warning_time
        )

        results.append({

            **event.to_dict(),

            "warning_issued":
                True,

            "first_warning_time":
                warning_time,

            "lead_time_s":
                lead_time,

        })

    return pd.DataFrame(
        results
    )


# ==============================================================
# FALSE WARNING EPISODES
# ==============================================================

def calculate_false_alarm_episodes(
    events,
    warning_episodes,
):

    """
    A warning episode is a false warning if it cannot
    be associated with any actual instability event
    within MAX_WARNING_WINDOW_S before event onset.
    """

    false_alarms = []

    for _, warning in (
        warning_episodes.iterrows()
    ):

        run_id = int(
            warning["Run_ID"]
        )

        warning_time = float(
            warning["warning_start"]
        )

        run_events = (
            events[
                events["Run_ID"] == run_id
            ]
        )

        associated = False

        for _, event in (
            run_events.iterrows()
        ):

            onset = float(
                event["onset_time"]
            )

            if (
                onset
                -
                MAX_WARNING_WINDOW_S
                <= warning_time
                <= onset
            ):

                associated = True

                break

        if not associated:

            false_alarms.append(
                warning.to_dict()
            )

    if len(false_alarms) == 0:

        return pd.DataFrame(
            columns=warning_episodes.columns
        )

    return pd.DataFrame(
        false_alarms
    )


# ==============================================================
# MODEL EVALUATION
# ==============================================================

def evaluate_model(
    model_name,
    prediction_path,
    y_test,
    timeline,
    events,
):

    """
    Evaluate one model across all forecast horizons.
    """

    predictions = np.load(
        prediction_path
    )

    # ----------------------------------------------------------
    # Shape validation
    # ----------------------------------------------------------

    if predictions.shape != y_test.shape:

        raise ValueError(

            f"{model_name}: prediction shape "
            f"{predictions.shape} does not match "
            f"y_test shape {y_test.shape}"

        )

    results = []

    event_tables = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        model_name
    )

    print(
        "=" * 70
    )

    # ----------------------------------------------------------
    # Each horizon
    # ----------------------------------------------------------

    for (
        horizon_name,
        horizon_index
    ) in HORIZONS.items():

        print(
            f"\nEvaluating {horizon_name}..."
        )

        horizon_predictions = (
            predictions[
                :,
                horizon_index
            ]
        )

        # ------------------------------------------------------
        # Warning episodes
        # ------------------------------------------------------

        warning_episodes = (
            find_warning_episodes(

                timeline=timeline,

                predictions=
                    horizon_predictions,

                threshold=THRESHOLD,
            )
        )

        # ------------------------------------------------------
        # Match warnings to events
        # ------------------------------------------------------

        event_results = (
            calculate_event_lead_times(

                events=events,

                warning_episodes=
                    warning_episodes,
            )
        )

        # ------------------------------------------------------
        # False-warning episodes
        # ------------------------------------------------------

        false_alarm_df = (
            calculate_false_alarm_episodes(

                events=events,

                warning_episodes=
                    warning_episodes,
            )
        )

        # ------------------------------------------------------
        # Event statistics
        # ------------------------------------------------------

        total_events = len(
            events
        )

        detected = int(
            event_results[
                "warning_issued"
            ].sum()
        )

        missed = (
            total_events
            -
            detected
        )

        detection_rate = (
            detected / total_events
            if total_events > 0
            else np.nan
        )

        # ------------------------------------------------------
        # Lead-time statistics
        # ------------------------------------------------------

        detected_leads = (
            event_results.loc[
                event_results[
                    "warning_issued"
                ],
                "lead_time_s",
            ]
        )

        if len(
            detected_leads
        ) > 0:

            mean_lead = float(
                detected_leads.mean()
            )

            median_lead = float(
                detected_leads.median()
            )

            min_lead = float(
                detected_leads.min()
            )

            max_lead = float(
                detected_leads.max()
            )

            std_lead = float(
                detected_leads.std(
                    ddof=1
                )
            )

            q25_lead = float(
                detected_leads.quantile(
                    0.25
                )
            )

            q75_lead = float(
                detected_leads.quantile(
                    0.75
                )
            )

        else:

            mean_lead = np.nan
            median_lead = np.nan
            min_lead = np.nan
            max_lead = np.nan
            std_lead = np.nan
            q25_lead = np.nan
            q75_lead = np.nan

        # ------------------------------------------------------
        # Warning episode statistics
        # ------------------------------------------------------

        total_warning_episodes = len(
            warning_episodes
        )

        false_warning_episodes = len(
            false_alarm_df
        )

        # ------------------------------------------------------
        # Save summary
        # ------------------------------------------------------

        results.append({

            "model":
                model_name,

            "horizon":
                horizon_name,

            "threshold":
                THRESHOLD,

            "total_events":
                total_events,

            "detected_events":
                detected,

            "missed_events":
                missed,

            "event_detection_rate":
                detection_rate,

            "mean_lead_time_s":
                mean_lead,

            "median_lead_time_s":
                median_lead,

            "std_lead_time_s":
                std_lead,

            "minimum_lead_time_s":
                min_lead,

            "maximum_lead_time_s":
                max_lead,

            "lead_time_q25_s":
                q25_lead,

            "lead_time_q75_s":
                q75_lead,

            "total_warning_episodes":
                total_warning_episodes,

            "false_warning_episodes":
                false_warning_episodes,
        })

        # ------------------------------------------------------
        # Save event-level table
        # ------------------------------------------------------

        event_results = (
            event_results.copy()
        )

        event_results[
            "model"
        ] = model_name

        event_results[
            "horizon"
        ] = horizon_name

        event_tables.append(
            event_results
        )

        # ------------------------------------------------------
        # Print results
        # ------------------------------------------------------

        print(
            f"  Total events          : "
            f"{total_events}"
        )

        print(
            f"  Detected events       : "
            f"{detected}"
        )

        print(
            f"  Missed events         : "
            f"{missed}"
        )

        print(
            f"  Detection rate        : "
            f"{detection_rate:.4f}"
        )

        print(
            f"  Mean lead time        : "
            f"{mean_lead:.4f} s"
        )

        print(
            f"  Median lead time      : "
            f"{median_lead:.4f} s"
        )

        print(
            f"  Lead-time std         : "
            f"{std_lead:.4f} s"
        )

        print(
            f"  Lead-time Q25         : "
            f"{q25_lead:.4f} s"
        )

        print(
            f"  Lead-time Q75         : "
            f"{q75_lead:.4f} s"
        )

        print(
            f"  Warning episodes      : "
            f"{total_warning_episodes}"
        )

        print(
            f"  False-warning episodes: "
            f"{false_warning_episodes}"
        )

    return (
        pd.DataFrame(results),
        pd.concat(
            event_tables,
            ignore_index=True
        ),
    )


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("=" * 70)

    print(
        "EVENT-LEVEL EARLY-WARNING "
        "LEAD-TIME EVALUATION"
    )

    print("=" * 70)

    # ==========================================================
    # LOAD TEST SEQUENCES
    # ==========================================================

    (
        X_test,
        y_test,
        run_ids,
        times,
    ) = load_test_data()

    print(
        "\nTEST DATA"
    )

    print(
        "-" * 70
    )

    print(
        "X_test shape:",
        X_test.shape
    )

    print(
        "y_test shape:",
        y_test.shape
    )

    print(
        "Unique test runs:",
        len(
            np.unique(
                run_ids
            )
        )
    )

    # ==========================================================
    # BUILD TIMELINE
    # ==========================================================

    timeline = (
        build_test_timeline(
            run_ids,
            times,
        )
    )

    # ==========================================================
    # LOAD GROUND TRUTH
    # ==========================================================

    gt = load_ground_truth()

    # ==========================================================
    # RESTRICT TO TEST RUNS
    # ==========================================================

    test_run_ids = set(
        np.unique(
            run_ids
        ).astype(int)
    )

    gt_test = gt[
        gt["Run_ID"].isin(
            test_run_ids
        )
    ].copy()

    # ==========================================================
    # EXTRACT EVENTS
    # ==========================================================

    events = extract_events(
        gt_test
    )

    print(
        "\nGROUND-TRUTH EVENTS"
    )

    print(
        "-" * 70
    )

    print(
        "Test instability events:",
        len(events)
    )

    print(
        "Test unstable runs:",
        events["Run_ID"].nunique()
        if len(events) > 0
        else 0
    )

    # ==========================================================
    # MODEL EVALUATION
    # ==========================================================

    all_results = []

    all_event_results = []

    for (
        model_name,
        prediction_path
    ) in MODELS.items():

        # ------------------------------------------------------
        # Check prediction file
        # ------------------------------------------------------

        if not prediction_path.exists():

            print(
                "\nWARNING:"
            )

            print(
                "Prediction file not found:"
            )

            print(
                prediction_path
            )

            continue

        # ------------------------------------------------------
        # Evaluate
        # ------------------------------------------------------

        (
            results,
            event_results,
        ) = evaluate_model(

            model_name=
                model_name,

            prediction_path=
                prediction_path,

            y_test=
                y_test,

            timeline=
                timeline,

            events=
                events,
        )

        all_results.append(
            results
        )

        all_event_results.append(
            event_results
        )

    # ==========================================================
    # CHECK THAT SOMETHING WAS EVALUATED
    # ==========================================================

    if len(all_results) == 0:

        raise FileNotFoundError(
            "No prediction files were found."
        )

    # ==========================================================
    # COMBINE RESULTS
    # ==========================================================

    summary = pd.concat(
        all_results,
        ignore_index=True
    )

    event_level = pd.concat(
        all_event_results,
        ignore_index=True
    )

    # ==========================================================
    # CREATE OUTPUT DIRECTORY
    # ==========================================================

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================================
    # SAVE SUMMARY
    # ==========================================================

    summary_path = (
        TABLE_DIR
        / "event_lead_time_metrics.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ==========================================================
    # SAVE EVENT DETAILS
    # ==========================================================

    event_path = (
        TABLE_DIR
        / "event_lead_time_details.csv"
    )

    event_level.to_csv(
        event_path,
        index=False
    )

    # ==========================================================
    # FINAL RESULTS
    # ==========================================================

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "FINAL EVENT-LEVEL RESULTS"
    )

    print(
        "=" * 70
    )

    display_columns = [

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

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ==========================================================
    # SAVE PATHS
    # ==========================================================

    print(
        "\nSaved:"
    )

    print(
        summary_path
    )

    print(
        event_path
    )

    # ==========================================================
    # COMPLETE
    # ==========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVENT LEAD-TIME EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    main()