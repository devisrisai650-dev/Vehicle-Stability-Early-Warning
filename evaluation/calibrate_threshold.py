# ==============================================================
# VALIDATION THRESHOLD CALIBRATION
# Vehicle Stability Early-Warning System
# ==============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ==============================================================
# PROJECT PATHS
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"


# ==============================================================
# FORECAST HORIZONS
# ==============================================================

HORIZONS = {
    "0.5s": 0,
    "1.0s": 1,
    "2.0s": 2,
}


# ==============================================================
# WARNING PARAMETERS
# ==============================================================

# Sampling interval = 0.05 s
SAMPLE_TIME_S = 0.05

# Require two consecutive warning samples.
CONSECUTIVE_WARNINGS = 2

# If two warning samples are separated by more than this,
# treat them as different warning episodes.
WARNING_GAP_S = 0.20

# A warning is associated with an event only if it occurs
# within this time before the instability onset.
MAX_WARNING_WINDOW_S = 5.0


# ==============================================================
# THRESHOLD GRID
# ==============================================================

# Physics instability boundary:
#
#     stability_margin = 0
#
# A negative value means the predicted state is beyond the
# physical stability boundary.
#
# Positive thresholds allow an EARLY warning before the
# physical boundary is crossed.
#
# These thresholds are calibrated ONLY on validation data.

THRESHOLDS = np.round(
    np.arange(
        -0.50,
        0.51,
        0.05
    ),
    2
)


# ==============================================================
# FIND VALIDATION PREDICTION FILE
# ==============================================================

def find_prediction_file(
    model_name,
    expected_shape,
):
    """
    Automatically locate the correct validation prediction
    .npy file for a model.

    We do NOT assume a particular filename.

    The selected file must:

        1. Contain the model name.
        2. Look like a validation/prediction file.
        3. Have shape (N, 3).
        4. Match the validation target shape.
    """

    all_npy_files = list(
        PROCESSED_DIR.rglob("*.npy")
    )

    if not all_npy_files:

        raise FileNotFoundError(
            "\nNo .npy files were found inside:\n"
            f"{PROCESSED_DIR}"
        )

    model_keywords = {

        "XGBoost": [
            "xgb",
            "xgboost",
        ],

        "LSTM": [
            "lstm",
        ],

        "GRU": [
            "gru",
        ],
    }

    keywords = model_keywords[
        model_name
    ]

    candidates = []

    for path in all_npy_files:

        name = path.name.lower()

        # ------------------------------------------------------
        # Model name must be present
        # ------------------------------------------------------

        if not any(
            keyword in name
            for keyword in keywords
        ):
            continue

        # ------------------------------------------------------
        # It should look like a prediction file
        # ------------------------------------------------------

        prediction_words = [
            "prediction",
            "predictions",
            "pred",
        ]

        validation_words = [
            "validation",
            "val",
        ]

        has_prediction_word = any(
            word in name
            for word in prediction_words
        )

        has_validation_word = any(
            word in name
            for word in validation_words
        )

        if not (
            has_prediction_word
            or has_validation_word
        ):
            continue

        # ------------------------------------------------------
        # Check actual numpy shape
        # ------------------------------------------------------

        try:

            arr = np.load(
                path,
                allow_pickle=False
            )

        except Exception:

            continue

        if arr.shape != expected_shape:

            continue

        candidates.append(
            path
        )

    # ----------------------------------------------------------
    # No match
    # ----------------------------------------------------------

    if len(candidates) == 0:

        print(
            "\nCould not automatically locate "
            f"{model_name} validation predictions."
        )

        print(
            "\nAll available .npy files:"
        )

        for path in all_npy_files:

            try:

                arr = np.load(
                    path,
                    allow_pickle=False
                )

                print(
                    f"  {path}"
                    f"    shape={arr.shape}"
                )

            except Exception:

                print(
                    f"  {path}"
                )

        raise FileNotFoundError(
            "\nNo suitable validation prediction "
            f"file found for {model_name}."
        )

    # ----------------------------------------------------------
    # Prefer files explicitly mentioning validation
    # ----------------------------------------------------------

    validation_candidates = [

        path

        for path in candidates

        if (
            "validation"
            in path.name.lower()
            or
            "val"
            in path.name.lower()
        )
    ]

    if len(
        validation_candidates
    ) > 0:

        candidates = (
            validation_candidates
        )

    # ----------------------------------------------------------
    # If multiple files remain, display them.
    # ----------------------------------------------------------

    if len(candidates) > 1:

        print(
            f"\nMultiple possible {model_name} "
            "prediction files found:"
        )

        for i, path in enumerate(
            candidates,
            start=1
        ):

            print(
                f"  {i}. {path}"
            )

        # Prefer the first candidate.
        selected = candidates[0]

        print(
            "\nAutomatically selecting:"
        )

        print(
            selected
        )

    else:

        selected = candidates[0]

    return selected


# ==============================================================
# LOAD VALIDATION DATA
# ==============================================================

def load_validation_data():

    sequence_path = (
        PROCESSED_DIR
        / "validation_sequences.npz"
    )

    if not sequence_path.exists():

        raise FileNotFoundError(
            "\nValidation sequence file not found:\n"
            f"{sequence_path}"
        )

    data = np.load(
        sequence_path,
        allow_pickle=True
    )

    required = [
        "X",
        "y",
        "Run_ID",
        "Time",
    ]

    missing = [

        key

        for key in required

        if key not in data.files
    ]

    if missing:

        raise KeyError(
            f"Missing validation keys: {missing}"
        )

    X = data["X"]

    y = data["y"]

    run_ids = (
        data["Run_ID"]
        .astype(int)
    )

    times = (
        data["Time"]
        .astype(float)
    )

    print(
        "=" * 70
    )

    print(
        "VALIDATION DATA"
    )

    print(
        "=" * 70
    )

    print(
        "X:",
        X.shape
    )

    print(
        "y:",
        y.shape
    )

    print(
        "Run_ID:",
        run_ids.shape
    )

    print(
        "Time:",
        times.shape
    )

    print(
        "Unique runs:",
        len(
            np.unique(
                run_ids
            )
        )
    )

    return (
        X,
        y,
        run_ids,
        times,
    )


# ==============================================================
# FIND GROUND-TRUTH FILE
# ==============================================================

def find_ground_truth_file():

    candidates = [

        DATA_DIR
        / "raw"
        / "ground_truth_dynamics.csv",

        DATA_DIR
        / "raw"
        / "ground_truth.csv",

        DATA_DIR
        / "ground_truth_dynamics.csv",

        DATA_DIR
        / "ground_truth.csv",

    ]

    for path in candidates:

        if path.exists():

            return path

    # ----------------------------------------------------------
    # Search recursively
    # ----------------------------------------------------------

    csv_files = list(
        DATA_DIR.rglob("*.csv")
    )

    for path in csv_files:

        name = path.name.lower()

        if (
            "ground"
            in name
            and
            "truth"
            in name
        ):

            return path

    raise FileNotFoundError(
        "\nCould not find ground-truth CSV."
        "\nExpected a file containing:"
        "\n  Run_ID"
        "\n  Time"
        "\n  instability_flag"
        "\n  stability_margin"
    )


# ==============================================================
# LOAD GROUND TRUTH
# ==============================================================

def load_ground_truth():

    gt_path = (
        find_ground_truth_file()
    )

    print(
        "\nGround-truth file:"
    )

    print(
        gt_path
    )

    gt = pd.read_csv(
        gt_path
    )

    required = [

        "Run_ID",

        "Time",

        "instability_flag",

        "stability_margin",
    ]

    missing = [

        column

        for column in required

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
# BUILD VALIDATION TIMELINE
# ==============================================================

def build_timeline(
    run_ids,
    times,
):

    timeline = pd.DataFrame({

        "Run_ID":
            run_ids,

        "Time":
            times,
    })

    timeline = (
        timeline
        .reset_index()
        .rename(
            columns={
                "index":
                    "sample_index"
            }
        )
    )

    return timeline


# ==============================================================
# EXTRACT INSTABILITY EVENTS
# ==============================================================

def extract_events(
    gt
):

    events = []

    event_id = 0

    for run_id, run_df in (
        gt.groupby("Run_ID")
    ):

        run_df = (
            run_df
            .sort_values(
                "Time"
            )
            .reset_index(
                drop=True
            )
        )

        flags = (
            run_df[
                "instability_flag"
            ]
            .to_numpy()
            .astype(bool)
        )

        i = 0

        while i < len(flags):

            if not flags[i]:

                i += 1

                continue

            start = i

            while (
                i + 1 < len(flags)
                and
                flags[i + 1]
            ):

                i += 1

            end = i

            event_id += 1

            event_df = (
                run_df
                .iloc[
                    start:
                    end + 1
                ]
            )

            events.append({

                "Run_ID":
                    int(run_id),

                "event_id":
                    event_id,

                "onset_time":
                    float(
                        event_df[
                            "Time"
                        ].iloc[0]
                    ),

                "end_time":
                    float(
                        event_df[
                            "Time"
                        ].iloc[-1]
                    ),
            })

            i += 1

    if len(events) == 0:

        return pd.DataFrame(
            columns=[
                "Run_ID",
                "event_id",
                "onset_time",
                "end_time",
            ]
        )

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

    episodes = []

    for run_id, run_df in (
        timeline.groupby("Run_ID")
    ):

        run_df = (
            run_df
            .sort_values(
                "Time"
            )
            .reset_index(
                drop=True
            )
        )

        sample_indices = (
            run_df[
                "sample_index"
            ]
            .to_numpy()
        )

        times = (
            run_df[
                "Time"
            ]
            .to_numpy()
        )

        run_predictions = (
            predictions[
                sample_indices
            ]
        )

        # ------------------------------------------------------
        # Prediction below warning threshold
        # ------------------------------------------------------

        raw_warning = (
            run_predictions
            < threshold
        )

        # ------------------------------------------------------
        # Require consecutive warning samples
        # ------------------------------------------------------

        sustained_warning = np.zeros(
            len(raw_warning),
            dtype=bool
        )

        consecutive_count = 0

        for i, warning in enumerate(
            raw_warning
        ):

            if warning:

                consecutive_count += 1

            else:

                consecutive_count = 0

            if (
                consecutive_count
                >=
                CONSECUTIVE_WARNINGS
            ):

                start = (
                    i
                    -
                    CONSECUTIVE_WARNINGS
                    +
                    1
                )

                sustained_warning[
                    start:
                    i + 1
                ] = True

        warning_indices = np.where(
            sustained_warning
        )[0]

        if len(
            warning_indices
        ) == 0:

            continue

        # ------------------------------------------------------
        # Group sustained warnings
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

            gap = (
                times[current]
                -
                times[previous]
            )

            if gap > WARNING_GAP_S:

                indices = np.arange(
                    episode_start,
                    previous + 1
                )

                episode_predictions = (
                    run_predictions[
                        indices
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

                    "minimum_prediction":
                        float(
                            np.min(
                                episode_predictions
                            )
                        ),
                })

                episode_start = (
                    current
                )

            previous = (
                current
            )

        # ------------------------------------------------------
        # Final episode
        # ------------------------------------------------------

        indices = np.arange(
            episode_start,
            previous + 1
        )

        episode_predictions = (
            run_predictions[
                indices
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

            "minimum_prediction":
                float(
                    np.min(
                        episode_predictions
                    )
                ),
        })

    if len(episodes) == 0:

        return pd.DataFrame(
            columns=[
                "Run_ID",
                "warning_start",
                "warning_end",
                "minimum_prediction",
            ]
        )

    return pd.DataFrame(
        episodes
    )


# ==============================================================
# EVALUATE WARNING THRESHOLD
# ==============================================================

def evaluate_threshold(
    events,
    warning_episodes,
):

    total_events = len(
        events
    )

    detected_events = 0

    lead_times = []

    matched_warning_indices = set()

    # ----------------------------------------------------------
    # Match warnings to instability events
    # ----------------------------------------------------------

    for _, event in (
        events.iterrows()
    ):

        run_id = int(
            event["Run_ID"]
        )

        onset = float(
            event["onset_time"]
        )

        candidates = (
            warning_episodes[
                warning_episodes[
                    "Run_ID"
                ]
                ==
                run_id
            ]
            .copy()
        )

        if len(
            candidates
        ) == 0:

            continue

        # Warning must happen BEFORE event onset
        candidates = candidates[
            (
                candidates[
                    "warning_start"
                ]
                <=
                onset
            )
            &
            (
                candidates[
                    "warning_start"
                ]
                >=
                (
                    onset
                    -
                    MAX_WARNING_WINDOW_S
                )
            )
        ]

        if len(
            candidates
        ) == 0:

            continue

        candidates = (
            candidates
            .sort_values(
                "warning_start"
            )
        )

        warning_index = (
            candidates.index[0]
        )

        warning = (
            candidates.iloc[0]
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

        # Only positive lead time counts.
        if lead_time <= 0:

            continue

        detected_events += 1

        lead_times.append(
            lead_time
        )

        matched_warning_indices.add(
            warning_index
        )

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    missed_events = (
        total_events
        -
        detected_events
    )

    if total_events > 0:

        detection_rate = (
            detected_events
            /
            total_events
        )

    else:

        detection_rate = 0.0

    if len(
        lead_times
    ) > 0:

        lead_array = np.asarray(
            lead_times,
            dtype=float
        )

        mean_lead = float(
            np.mean(
                lead_array
            )
        )

        median_lead = float(
            np.median(
                lead_array
            )
        )

        std_lead = float(
            np.std(
                lead_array
            )
        )

        q25 = float(
            np.percentile(
                lead_array,
                25
            )
        )

        q75 = float(
            np.percentile(
                lead_array,
                75
            )
        )

    else:

        mean_lead = np.nan

        median_lead = np.nan

        std_lead = np.nan

        q25 = np.nan

        q75 = np.nan

    warning_episode_count = (
        len(
            warning_episodes
        )
    )

    false_warning_episodes = (
        warning_episode_count
        -
        len(
            matched_warning_indices
        )
    )

    return {

        "total_events":
            total_events,

        "detected_events":
            detected_events,

        "missed_events":
            missed_events,

        "detection_rate":
            detection_rate,

        "mean_lead_time_s":
            mean_lead,

        "median_lead_time_s":
            median_lead,

        "std_lead_time_s":
            std_lead,

        "lead_time_q25_s":
            q25,

        "lead_time_q75_s":
            q75,

        "warning_episodes":
            warning_episode_count,

        "false_warning_episodes":
            false_warning_episodes,
    }


# ==============================================================
# SELECT BEST VALIDATION THRESHOLD
# ==============================================================

def select_threshold(
    results
):

    """
    Threshold selection is performed ONLY on validation data.

    Ranking:

    1. Higher event detection rate
    2. Fewer false-warning episodes
    3. Higher median lead time

    This threshold will subsequently be FROZEN and applied
    to the untouched test set.
    """

    ranked = (
        results
        .sort_values(

            by=[
                "detection_rate",

                "false_warning_episodes",

                "median_lead_time_s",
            ],

            ascending=[
                False,

                True,

                False,
            ]
        )
    )

    return ranked.iloc[0]


# ==============================================================
# MAIN
# ==============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "VALIDATION THRESHOLD CALIBRATION"
    )

    print(
        "=" * 70
    )

    # ==========================================================
    # LOAD VALIDATION SEQUENCES
    # ==========================================================

    (
        X_val,
        y_val,
        run_ids,
        times,
    ) = load_validation_data()

    # ==========================================================
    # TIMELINE
    # ==========================================================

    timeline = build_timeline(
        run_ids,
        times
    )

    # ==========================================================
    # LOAD GROUND TRUTH
    # ==========================================================

    gt = load_ground_truth()

    validation_run_ids = set(
        np.unique(
            run_ids
        )
    )

    gt_val = gt[
        gt["Run_ID"].isin(
            validation_run_ids
        )
    ].copy()

    # ==========================================================
    # EXTRACT EVENTS
    # ==========================================================

    events = extract_events(
        gt_val
    )

    print(
        "\nValidation events:",
        len(events)
    )

    if len(events) == 0:

        raise RuntimeError(
            "No validation instability events were found."
        )

    # ==========================================================
    # EXPECTED PREDICTION SHAPE
    # ==========================================================

    expected_shape = (
        y_val.shape
    )

    # ==========================================================
    # FIND PREDICTION FILES
    # ==========================================================

    model_paths = {}

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SEARCHING FOR VALIDATION PREDICTIONS"
    )

    print(
        "=" * 70
    )

    for model_name in [
        "XGBoost",
        "LSTM",
        "GRU",
    ]:

        path = find_prediction_file(
            model_name,
            expected_shape
        )

        model_paths[
            model_name
        ] = path

        print(
            f"\n{model_name}:"
        )

        print(
            path
        )

    # ==========================================================
    # STORAGE
    # ==========================================================

    all_results = []

    selected_thresholds = []

    # ==========================================================
    # MODEL LOOP
    # ==========================================================

    for model_name in [
        "XGBoost",
        "LSTM",
        "GRU",
    ]:

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

        prediction_path = (
            model_paths[
                model_name
            ]
        )

        predictions = np.load(
            prediction_path,
            allow_pickle=False
        )

        print(
            "Prediction shape:",
            predictions.shape
        )

        # ------------------------------------------------------
        # Shape verification
        # ------------------------------------------------------

        if predictions.shape != y_val.shape:

            raise ValueError(

                f"{model_name}: prediction shape "
                f"{predictions.shape} does not match "
                f"validation target shape "
                f"{y_val.shape}"
            )

        # ======================================================
        # HORIZON LOOP
        # ======================================================

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

            threshold_results = []

            # ==================================================
            # THRESHOLD LOOP
            # ==================================================

            for threshold in (
                THRESHOLDS
            ):

                warning_episodes = (
                    find_warning_episodes(

                        timeline=
                            timeline,

                        predictions=
                            horizon_predictions,

                        threshold=
                            threshold,
                    )
                )

                metrics = (
                    evaluate_threshold(

                        events=
                            events,

                        warning_episodes=
                            warning_episodes,
                    )
                )

                row = {

                    "model":
                        model_name,

                    "horizon":
                        horizon_name,

                    "threshold":
                        float(
                            threshold
                        ),

                    **metrics,
                }

                threshold_results.append(
                    row
                )

                all_results.append(
                    row
                )

            threshold_df = (
                pd.DataFrame(
                    threshold_results
                )
            )

            # ==================================================
            # SELECT THRESHOLD
            # ==================================================

            best = select_threshold(
                threshold_df
            )

            selected_thresholds.append(
                best
            )

            # ==================================================
            # DISPLAY
            # ==================================================

            print(
                "\nSelected validation threshold:"
            )

            print(
                f"  Threshold             : "
                f"{best['threshold']:.2f}"
            )

            print(
                f"  Detection rate        : "
                f"{best['detection_rate']:.4f}"
            )

            print(
                f"  Mean lead time        : "
                f"{best['mean_lead_time_s']:.4f} s"
            )

            print(
                f"  Median lead time      : "
                f"{best['median_lead_time_s']:.4f} s"
            )

            print(
                f"  Warning episodes      : "
                f"{int(best['warning_episodes'])}"
            )

            print(
                f"  False-warning episodes: "
                f"{int(best['false_warning_episodes'])}"
            )

    # ==========================================================
    # CREATE RESULTS DIRECTORY
    # ==========================================================

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================================
    # SAVE FULL THRESHOLD SWEEP
    # ==========================================================

    sweep_df = pd.DataFrame(
        all_results
    )

    sweep_path = (
        TABLE_DIR
        /
        "validation_threshold_sweep.csv"
    )

    sweep_df.to_csv(
        sweep_path,
        index=False
    )

    # ==========================================================
    # SELECTED THRESHOLDS
    # ==========================================================

    selected_df = pd.DataFrame(
        selected_thresholds
    )

    selected_path = (
        TABLE_DIR
        /
        "selected_warning_thresholds.csv"
    )

    selected_df.to_csv(
        selected_path,
        index=False
    )

    # ==========================================================
    # PRINT FINAL TABLE
    # ==========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SELECTED VALIDATION THRESHOLDS"
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

        "detection_rate",

        "mean_lead_time_s",

        "median_lead_time_s",

        "std_lead_time_s",

        "lead_time_q25_s",

        "lead_time_q75_s",

        "warning_episodes",

        "false_warning_episodes",
    ]

    print(
        selected_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ==========================================================
    # SAVE THRESHOLDS TO NPZ
    # ==========================================================

    threshold_dict = {}

    for _, row in (
        selected_df.iterrows()
    ):

        key = (

            f"{row['model']}_"
            f"{row['horizon']}"
        )

        threshold_dict[
            key
        ] = float(
            row["threshold"]
        )

    threshold_npz_path = (
        PROCESSED_DIR
        /
        "selected_warning_thresholds.npz"
    )

    np.savez(
        threshold_npz_path,
        **threshold_dict
    )

    # ==========================================================
    # SAVE HUMAN-READABLE TXT
    # ==========================================================

    txt_path = (
        TABLE_DIR
        /
        "selected_warning_thresholds.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SELECTED VALIDATION WARNING THRESHOLDS\n"
        )

        f.write(
            "=" * 60
            +
            "\n"
        )

        for _, row in (
            selected_df.iterrows()
        ):

            f.write(

                f"{row['model']} | "
                f"{row['horizon']} | "
                f"threshold="
                f"{row['threshold']:.2f} | "
                f"detection="
                f"{row['detection_rate']:.4f} | "
                f"mean_lead="
                f"{row['mean_lead_time_s']:.4f}s | "
                f"median_lead="
                f"{row['median_lead_time_s']:.4f}s\n"
            )

    # ==========================================================
    # FINAL PATHS
    # ==========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SAVED FILES"
    )

    print(
        "=" * 70
    )

    print(
        sweep_path
    )

    print(
        selected_path
    )

    print(
        threshold_npz_path
    )

    print(
        txt_path
    )

    # ==========================================================
    # COMPLETE
    # ==========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALIDATION THRESHOLD CALIBRATION COMPLETE"
    )

    print(
        "=" * 70
    )


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    main()