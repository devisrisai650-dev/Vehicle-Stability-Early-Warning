# ============================================================
# SAVE VALIDATION PREDICTIONS
# Vehicle Stability Early-Warning System
#
# Generates validation predictions for:
#   1. XGBoost
#   2. LSTM
#   3. GRU
#
# Output:
# data/processed/predictions/
#   xgboost_validation_predictions.npy
#   lstm_validation_predictions.npy
#   gru_validation_predictions.npy
# ============================================================

import os
import sys
import glob
import numpy as np
import xgboost as xgb

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

PREDICTIONS_DIR = os.path.join(
    PROCESSED_DIR,
    "predictions"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "models"
)

os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# ============================================================
# FILE PATHS
# ============================================================

VALIDATION_FILE = os.path.join(
    PROCESSED_DIR,
    "validation_sequences.npz"
)

NORMALIZATION_FILE = os.path.join(
    PROCESSED_DIR,
    "lstm_normalization.npz"
)


# ============================================================
# HORIZONS
# ============================================================

HORIZONS = [0.5, 1.0, 2.0]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 70)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError(
        f"\nValidation dataset not found:\n"
        f"{VALIDATION_FILE}"
    )

data = np.load(VALIDATION_FILE)

X_val = data["X"]
y_val = data["y"]

print("\nValidation data")
print("-" * 70)
print(f"X: {X_val.shape}")
print(f"y: {y_val.shape}")


# ============================================================
# BASIC VALIDATION
# ============================================================

if X_val.ndim != 3:
    raise ValueError(
        f"Expected X_val to have 3 dimensions "
        f"(samples, timesteps, features), got {X_val.shape}"
    )

if y_val.ndim != 2:
    raise ValueError(
        f"Expected y_val to have 2 dimensions "
        f"(samples, horizons), got {y_val.shape}"
    )

if X_val.shape[0] != y_val.shape[0]:
    raise ValueError(
        "Number of X and y samples does not match."
    )

print(
    f"Timesteps: {X_val.shape[1]}"
)

print(
    f"Features: {X_val.shape[2]}"
)

print(
    f"Horizons: {y_val.shape[1]}"
)


# ============================================================
# XGBOOST VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST")
print("=" * 70)

xgb_predictions = []

# XGBoost was trained using flattened sequences
X_val_flat = X_val.reshape(
    X_val.shape[0],
    -1
)

print(
    f"\nFlattened validation shape: "
    f"{X_val_flat.shape}"
)

for i, horizon in enumerate(HORIZONS):

    print(
        f"\nTraining horizon: {horizon:.1f}s"
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"xgboost_{horizon:.1f}s.json"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"\nXGBoost model not found:\n"
            f"{model_path}\n\n"
            f"Available model files:"
        )

    print(
        f"Loading:\n{model_path}"
    )

    model = xgb.XGBRegressor()

    model.load_model(model_path)

    prediction = model.predict(
        X_val_flat
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float32
    ).reshape(-1)

    print(
        f"Prediction shape: "
        f"{prediction.shape}"
    )

    if prediction.shape[0] != X_val.shape[0]:
        raise ValueError(
            f"XGBoost prediction size mismatch "
            f"for {horizon}s."
        )

    xgb_predictions.append(prediction)


# Stack horizons

xgb_predictions = np.column_stack(
    xgb_predictions
)

print(
    "\nXGBoost validation prediction shape:"
)

print(
    xgb_predictions.shape
)

xgb_output = os.path.join(
    PREDICTIONS_DIR,
    "xgboost_validation_predictions.npy"
)

np.save(
    xgb_output,
    xgb_predictions
)

print(
    "\nSaved XGBoost predictions:"
)

print(
    xgb_output
)


# ============================================================
# LOAD NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("LOADING NORMALIZATION")
print("=" * 70)

if not os.path.exists(NORMALIZATION_FILE):

    raise FileNotFoundError(
        f"\nNormalization file not found:\n"
        f"{NORMALIZATION_FILE}"
    )

norm = np.load(
    NORMALIZATION_FILE
)

print(
    "\nNormalization keys:"
)

print(
    list(norm.files)
)


# ------------------------------------------------------------
# Your file uses:
#
# mean
# std
#
# NOT:
# means
# stds
# ------------------------------------------------------------

if "mean" not in norm.files:

    raise KeyError(
        "Normalization file does not contain 'mean'."
    )

if "std" not in norm.files:

    raise KeyError(
        "Normalization file does not contain 'std'."
    )

means = norm["mean"].astype(
    np.float32
)

stds = norm["std"].astype(
    np.float32
)

print(
    "\nMeans:"
)

print(
    means
)

print(
    "\nStandard deviations:"
)

print(
    stds
)


# ============================================================
# NORMALIZATION SHAPE CHECK
# ============================================================

print(
    "\nNormalization shapes:"
)

print(
    f"mean: {means.shape}"
)

print(
    f"std : {stds.shape}"
)

print(
    f"X   : {X_val.shape}"
)


# ------------------------------------------------------------
# Expected:
#
# X:
#     (4683, 20, 4)
#
# mean:
#     (1, 1, 4)
#
# std:
#     (1, 1, 4)
#
# Direct broadcasting works:
#
# (4683,20,4) - (1,1,4)
# ---------------------
#        (1,1,4)
#
# Result:
# (4683,20,4)
# ------------------------------------------------------------

if means.shape != (1, 1, X_val.shape[2]):

    # Allow equivalent shapes such as (4,) or (1,4)
    # by reshaping them to (1,1,4).

    if means.size == X_val.shape[2]:

        means = means.reshape(
            1,
            1,
            X_val.shape[2]
        )

    else:

        raise ValueError(
            f"Cannot reshape normalization mean "
            f"{means.shape} to "
            f"(1,1,{X_val.shape[2]})"
        )


if stds.shape != (1, 1, X_val.shape[2]):

    if stds.size == X_val.shape[2]:

        stds = stds.reshape(
            1,
            1,
            X_val.shape[2]
        )

    else:

        raise ValueError(
            f"Cannot reshape normalization std "
            f"{stds.shape} to "
            f"(1,1,{X_val.shape[2]})"
        )


# Avoid division by zero

if np.any(stds == 0):

    raise ValueError(
        "Normalization contains zero standard deviation."
    )


# ============================================================
# NORMALIZE VALIDATION DATA
# ============================================================

X_val_float = X_val.astype(
    np.float32
)

X_val_norm = (
    X_val_float - means
) / stds


print(
    "\nNormalized validation shape:"
)

print(
    X_val_norm.shape
)


print(
    "\nNormalized feature means:"
)

print(
    np.mean(
        X_val_norm,
        axis=(0, 1)
    )
)


print(
    "\nNormalized feature standard deviations:"
)

print(
    np.std(
        X_val_norm,
        axis=(0, 1)
    )
)


# ============================================================
# LSTM VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("LSTM")
print("=" * 70)


lstm_model_path = os.path.join(
    MODEL_DIR,
    "lstm_forecaster.keras"
)

if not os.path.exists(lstm_model_path):

    raise FileNotFoundError(
        f"\nLSTM model not found:\n"
        f"{lstm_model_path}"
    )


print(
    f"\nLoading:\n{lstm_model_path}"
)

lstm_model = load_model(
    lstm_model_path,
    compile=False
)


print(
    "\nGenerating LSTM validation predictions..."
)

lstm_predictions = lstm_model.predict(
    X_val_norm,
    verbose=1
)

lstm_predictions = np.asarray(
    lstm_predictions,
    dtype=np.float32
)


print(
    "\nLSTM validation prediction shape:"
)

print(
    lstm_predictions.shape
)


if lstm_predictions.shape != y_val.shape:

    raise ValueError(
        "\nLSTM prediction shape does not match "
        f"target shape.\n"
        f"Prediction: {lstm_predictions.shape}\n"
        f"Target:     {y_val.shape}"
    )


lstm_output = os.path.join(
    PREDICTIONS_DIR,
    "lstm_validation_predictions.npy"
)

np.save(
    lstm_output,
    lstm_predictions
)

print(
    "\nSaved LSTM predictions:"
)

print(
    lstm_output
)


# ============================================================
# GRU VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GRU")
print("=" * 70)


# Try the most likely GRU model names

gru_candidates = [

    os.path.join(
        MODEL_DIR,
        "gru_forecaster.keras"
    ),

    os.path.join(
        MODEL_DIR,
        "gru_best.keras"
    ),

    os.path.join(
        MODEL_DIR,
        "gru_model.keras"
    )
]


gru_model_path = None

for candidate in gru_candidates:

    if os.path.exists(candidate):

        gru_model_path = candidate
        break


if gru_model_path is None:

    print(
        "\nCould not find GRU model."
    )

    print(
        "\nSearched:"
    )

    for candidate in gru_candidates:

        print(
            f"  {candidate}"
        )

    raise FileNotFoundError(
        "\nGRU model file not found."
    )


print(
    f"\nLoading:\n{gru_model_path}"
)

gru_model = load_model(
    gru_model_path,
    compile=False
)


print(
    "\nGenerating GRU validation predictions..."
)

gru_predictions = gru_model.predict(
    X_val_norm,
    verbose=1
)

gru_predictions = np.asarray(
    gru_predictions,
    dtype=np.float32
)


print(
    "\nGRU validation prediction shape:"
)

print(
    gru_predictions.shape
)


if gru_predictions.shape != y_val.shape:

    raise ValueError(
        "\nGRU prediction shape does not match "
        f"target shape.\n"
        f"Prediction: {gru_predictions.shape}\n"
        f"Target:     {y_val.shape}"
    )


gru_output = os.path.join(
    PREDICTIONS_DIR,
    "gru_validation_predictions.npy"
)

np.save(
    gru_output,
    gru_predictions
)

print(
    "\nSaved GRU predictions:"
)

print(
    gru_output
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)


files_to_check = {

    "XGBoost":
        xgb_output,

    "LSTM":
        lstm_output,

    "GRU":
        gru_output
}


all_good = True


for model_name, path in files_to_check.items():

    print(
        f"\n{model_name}:"
    )

    if not os.path.exists(path):

        print(
            "  ERROR: File not found"
        )

        all_good = False

        continue


    pred = np.load(path)

    print(
        f"  File: {path}"
    )

    print(
        f"  Shape: {pred.shape}"
    )

    print(
        f"  Expected: {y_val.shape}"
    )


    if pred.shape != y_val.shape:

        print(
            "  ERROR: Shape mismatch"
        )

        all_good = False

    else:

        print(
            "  OK"
        )


# ============================================================
# COMPLETE
# ============================================================

if not all_good:

    raise RuntimeError(
        "\nOne or more prediction files failed "
        "verification."
    )


print("\n" + "=" * 70)
print("ALL VALIDATION PREDICTIONS GENERATED SUCCESSFULLY")
print("=" * 70)

print(
    "\nPrediction directory:"
)

print(
    PREDICTIONS_DIR
)

print(
    "\nFiles:"
)

print(
    "  xgboost_validation_predictions.npy"
)

print(
    "  lstm_validation_predictions.npy"
)

print(
    "  gru_validation_predictions.npy"
)

print("=" * 70)