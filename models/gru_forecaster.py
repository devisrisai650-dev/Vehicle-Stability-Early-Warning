from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==============================================================
# Project paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TABLE_DIR = RESULTS_DIR / "tables"
MODEL_DIR = RESULTS_DIR / "models"
PREDICTION_DIR = DATA_DIR / "predictions"

TRAIN_PATH = DATA_DIR / "train_sequences.npz"
VAL_PATH = DATA_DIR / "validation_sequences.npz"
TEST_PATH = DATA_DIR / "test_sequences.npz"


# ==============================================================
# Configuration
# ==============================================================

RANDOM_SEED = 42

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

BATCH_SIZE = 128
EPOCHS = 50


# ==============================================================
# Reproducibility
# ==============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ==============================================================
# Load data
# ==============================================================

def load_sequence_file(path):

    data = np.load(path)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    return X, y


# ==============================================================
# Training-only normalization
# ==============================================================

def fit_normalizer(X_train):

    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True
    )

    std = X_train.std(
        axis=(0, 1),
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-8
    )

    return mean, std


def normalize(X, mean, std):

    return (
        X - mean
    ) / std


# ==============================================================
# Build GRU
# ==============================================================

def build_model(
    sequence_length,
    n_features,
    n_outputs,
):

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(
                    sequence_length,
                    n_features,
                )
            ),

            tf.keras.layers.GRU(
                64,
                return_sequences=False,
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                n_outputs
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),

        loss="mse",

        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
    )

    return model


# ==============================================================
# Metrics
# ==============================================================

def calculate_metrics(
    y_true,
    y_pred,
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
    print("GRU SENSOR-ONLY FORECASTER")
    print("=" * 70)

    # ----------------------------------------------------------
    # Load
    # ----------------------------------------------------------

    X_train, y_train = load_sequence_file(
        TRAIN_PATH
    )

    X_val, y_val = load_sequence_file(
        VAL_PATH
    )

    X_test, y_test = load_sequence_file(
        TEST_PATH
    )

    print("\nRAW DATA")
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

    print(
        "y_train:",
        y_train.shape
    )

    # ----------------------------------------------------------
    # Normalize using TRAINING data only
    # ----------------------------------------------------------

    mean, std = fit_normalizer(
        X_train
    )

    X_train_norm = normalize(
        X_train,
        mean,
        std
    )

    X_val_norm = normalize(
        X_val,
        mean,
        std
    )

    X_test_norm = normalize(
        X_test,
        mean,
        std
    )

    print("\nNORMALIZATION")
    print("-" * 70)

    for i, name in enumerate(
        FEATURE_NAMES
    ):

        print(
            f"{name}: "
            f"mean={mean[0, 0, i]:.6f}, "
            f"std={std[0, 0, i]:.6f}"
        )

    # ----------------------------------------------------------
    # Build model
    # ----------------------------------------------------------

    model = build_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2],
        n_outputs=y_train.shape[1],
    )

    print("\nMODEL")
    print("-" * 70)

    model.summary()

    # ----------------------------------------------------------
    # Directories
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

    best_model_path = (
        MODEL_DIR
        / "gru_best.keras"
    )

    # ----------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                best_model_path
            ),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    print("\nTRAINING")
    print("-" * 70)

    history = model.fit(
        X_train_norm,
        y_train,

        validation_data=(
            X_val_norm,
            y_val,
        ),

        epochs=EPOCHS,
        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        shuffle=True,
        verbose=1,
    )

    # ----------------------------------------------------------
    # Test prediction
    # ----------------------------------------------------------

    print("\nTEST EVALUATION")
    print("-" * 70)

    y_pred = model.predict(
        X_test_norm,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    print(
        "Prediction shape:",
        y_pred.shape
    )

    # ----------------------------------------------------------
    # Save TEST predictions
    # ----------------------------------------------------------

    prediction_path = (
        PREDICTION_DIR
        / "gru_test_predictions.npy"
    )

    np.save(
        prediction_path,
        y_pred
    )

    # ----------------------------------------------------------
    # Verify saved predictions
    # ----------------------------------------------------------

    saved_predictions = np.load(
        prediction_path
    )

    if not np.allclose(
        saved_predictions,
        y_pred
    ):

        raise RuntimeError(
            "Saved GRU predictions do not "
            "match in-memory predictions."
        )

    print(
        "Saved test predictions:",
        prediction_path
    )

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    test_results = calculate_metrics(
        y_test,
        y_pred
    )

    print("\nTEST RESULTS")
    print("-" * 70)

    print(
        test_results.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Save metrics
    # ----------------------------------------------------------

    test_output = (
        TABLE_DIR
        / "gru_test_metrics.csv"
    )

    test_results.to_csv(
        test_output,
        index=False
    )

    # ----------------------------------------------------------
    # Save normalization parameters
    # ----------------------------------------------------------

    scaler_output = (
        DATA_DIR
        / "gru_normalization.npz"
    )

    np.savez(
        scaler_output,
        mean=mean,
        std=std
    )

    # ----------------------------------------------------------
    # Save model
    # ----------------------------------------------------------

    final_model_path = (
        MODEL_DIR
        / "gru_forecaster.keras"
    )

    model.save(
        final_model_path
    )

    # ----------------------------------------------------------
    # Save training history
    # ----------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_output = (
        TABLE_DIR
        / "gru_training_history.csv"
    )

    history_df.to_csv(
        history_output,
        index=False
    )

    # ----------------------------------------------------------
    # Final output
    # ----------------------------------------------------------

    print("\nSAVED FILES")
    print("-" * 70)

    print(
        "Test metrics:",
        test_output
    )

    print(
        "Test predictions:",
        prediction_path
    )

    print(
        "Normalization:",
        scaler_output
    )

    print(
        "Model:",
        final_model_path
    )

    print(
        "Training history:",
        history_output
    )

    print("\n" + "=" * 70)
    print("GRU BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()