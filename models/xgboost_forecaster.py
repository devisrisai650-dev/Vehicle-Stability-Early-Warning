from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ==============================================================
# Project paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

TABLE_DIR = (
    RESULTS_DIR
    / "tables"
)

MODEL_DIR = (
    RESULTS_DIR
    / "models"
)

PREDICTION_DIR = (
    DATA_DIR
    / "predictions"
)


# ==============================================================
# Input files
# ==============================================================

TRAIN_PATH = (
    DATA_DIR
    / "train_sequences.npz"
)

VAL_PATH = (
    DATA_DIR
    / "validation_sequences.npz"
)

TEST_PATH = (
    DATA_DIR
    / "test_sequences.npz"
)


# ==============================================================
# Configuration
# ==============================================================

FEATURE_NAMES = [
    "yaw_rate_meas",
    "lateral_accel_meas",
    "speed_meas",
    "steering_angle_meas",
]

HORIZON_NAMES = [
    "0.5s",
    "1.0s",
    "2.0s",
]


# ==============================================================
# Load forecasting dataset
# ==============================================================

def load_sequence_file(path):

    data = np.load(path)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    return X, y


# ==============================================================
# Flatten temporal sequences
# ==============================================================

def flatten_sequences(X):
    """
    Convert:

        [samples, 20, 4]

    into:

        [samples, 80]

    The original temporal ordering is preserved.
    """

    n_samples = X.shape[0]

    return X.reshape(
        n_samples,
        -1
    )


# ==============================================================
# Calculate regression metrics
# ==============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    results = []

    for i, horizon in enumerate(
        HORIZON_NAMES
    ):

        mae = mean_absolute_error(
            y_true[:, i],
            y_pred[:, i]
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true[:, i],
                y_pred[:, i]
            )
        )

        r2 = r2_score(
            y_true[:, i],
            y_pred[:, i]
        )

        results.append(
            {
                "horizon": horizon,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    return pd.DataFrame(
        results
    )


# ==============================================================
# Main
# ==============================================================

def main():

    print("=" * 70)
    print("XGBOOST SENSOR-ONLY FORECASTER")
    print("=" * 70)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------

    X_train_raw, y_train = load_sequence_file(
        TRAIN_PATH
    )

    X_val_raw, y_val = load_sequence_file(
        VAL_PATH
    )

    X_test_raw, y_test = load_sequence_file(
        TEST_PATH
    )

    print("\nRAW SEQUENCE SHAPES")
    print("-" * 70)

    print(
        "X_train:",
        X_train_raw.shape
    )

    print(
        "X_val:",
        X_val_raw.shape
    )

    print(
        "X_test:",
        X_test_raw.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    # ----------------------------------------------------------
    # Flatten sequences
    # ----------------------------------------------------------

    X_train = flatten_sequences(
        X_train_raw
    )

    X_val = flatten_sequences(
        X_val_raw
    )

    X_test = flatten_sequences(
        X_test_raw
    )

    print("\nFLATTENED FEATURES")
    print("-" * 70)

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_val:",
        X_val.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    # ----------------------------------------------------------
    # XGBoost configuration
    # ----------------------------------------------------------

    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )

    # ----------------------------------------------------------
    # Validation phase
    # ----------------------------------------------------------

    validation_results = []

    print("\nVALIDATION PHASE")
    print("-" * 70)

    for horizon_index, horizon_name in enumerate(
        HORIZON_NAMES
    ):

        print(
            f"\nTraining XGBoost for {horizon_name}..."
        )

        model = XGBRegressor(
            **params
        )

        model.fit(
            X_train,
            y_train[:, horizon_index],

            eval_set=[
                (
                    X_val,
                    y_val[:, horizon_index]
                )
            ],

            verbose=False,
        )

        val_prediction = model.predict(
            X_val
        )

        mae = mean_absolute_error(
            y_val[:, horizon_index],
            val_prediction
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_val[:, horizon_index],
                val_prediction
            )
        )

        r2 = r2_score(
            y_val[:, horizon_index],
            val_prediction
        )

        validation_results.append(
            {
                "horizon": horizon_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        print(
            f"  MAE  = {mae:.6f}"
        )

        print(
            f"  RMSE = {rmse:.6f}"
        )

        print(
            f"  R2   = {r2:.6f}"
        )

    validation_df = pd.DataFrame(
        validation_results
    )

    print("\nVALIDATION RESULTS")
    print("-" * 70)

    print(
        validation_df.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Final training on TRAIN + VALIDATION
    # ----------------------------------------------------------

    print("\nFINAL TRAINING")
    print("-" * 70)

    X_train_final = np.concatenate(
        [
            X_train,
            X_val,
        ],
        axis=0
    )

    y_train_final = np.concatenate(
        [
            y_train,
            y_val,
        ],
        axis=0
    )

    print(
        "Final training samples:",
        X_train_final.shape[0]
    )

    # ----------------------------------------------------------
    # Create output directories
    # ----------------------------------------------------------

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------------------
    # Train final models and generate TEST predictions
    # ----------------------------------------------------------

    test_predictions = np.zeros(
        (
            y_test.shape[0],
            len(HORIZON_NAMES)
        ),
        dtype=np.float32
    )

    final_models = []

    for horizon_index, horizon_name in enumerate(
        HORIZON_NAMES
    ):

        print(
            f"\nTraining final model for {horizon_name}..."
        )

        model = XGBRegressor(
            **params
        )

        model.fit(
            X_train_final,
            y_train_final[:, horizon_index],

            verbose=False,
        )

        prediction = model.predict(
            X_test
        )

        test_predictions[
            :,
            horizon_index
        ] = prediction

        final_models.append(
            model
        )

    # ----------------------------------------------------------
    # IMPORTANT:
    # Save predictions AFTER all predictions are generated.
    # ----------------------------------------------------------

    prediction_path = (
        PREDICTION_DIR
        / "xgboost_test_predictions.npy"
    )

    np.save(
        prediction_path,
        test_predictions
    )

    # ----------------------------------------------------------
    # Verify prediction file
    # ----------------------------------------------------------

    saved_predictions = np.load(
        prediction_path
    )

    if not np.allclose(
        saved_predictions,
        test_predictions
    ):

        raise RuntimeError(
            "Saved XGBoost predictions do not "
            "match in-memory predictions."
        )

    print(
        "\nSaved test predictions:",
        prediction_path
    )

    print(
        "Prediction shape:",
        test_predictions.shape
    )

    # ----------------------------------------------------------
    # Test metrics
    # ----------------------------------------------------------

    test_results = calculate_metrics(
        y_test,
        test_predictions
    )

    print("\nTEST RESULTS")
    print("-" * 70)

    print(
        test_results.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Save validation metrics
    # ----------------------------------------------------------

    validation_output = (
        TABLE_DIR
        / "xgboost_validation_metrics.csv"
    )

    validation_df.to_csv(
        validation_output,
        index=False
    )

    # ----------------------------------------------------------
    # Save test metrics
    # ----------------------------------------------------------

    test_output = (
        TABLE_DIR
        / "xgboost_test_metrics.csv"
    )

    test_results.to_csv(
        test_output,
        index=False
    )

    # ----------------------------------------------------------
    # Save models
    # ----------------------------------------------------------

    for model, horizon_name in zip(
        final_models,
        HORIZON_NAMES
    ):

        model_path = (
            MODEL_DIR
            / f"xgboost_{horizon_name}.json"
        )

        model.save_model(
            model_path
        )

    # ----------------------------------------------------------
    # Final information
    # ----------------------------------------------------------

    print("\nSAVED FILES")
    print("-" * 70)

    print(
        "Validation metrics:",
        validation_output
    )

    print(
        "Test metrics:",
        test_output
    )

    print(
        "Test predictions:",
        prediction_path
    )

    print(
        "Models:",
        MODEL_DIR
    )

    print("\n" + "=" * 70)
    print("XGBOOST BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()