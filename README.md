# Vehicle Stability Early Warning

A research prototype for forecasting a simulated vehicle's future rear-axle tire-saturation margin from noisy, low-cost retrofit sensor channels. This is **not a validated road-safety product** and must not be used to control a real vehicle.

## Scope and claims

- Inputs: measured yaw rate, lateral acceleration, wheel-speed-derived longitudinal speed, and steering angle.
- Target: simulated rear-tire saturation margin at 0.5 s, 1.0 s, and 2.0 s ahead.
- Baselines: persistence, a sensor-only physics observer, XGBoost, LSTM, and GRU.
- Data: nonlinear bicycle-model simulations with illustrative MEMS-like noise. Noise parameters are not calibrated to a particular sensor datasheet.
- The current margin is a rear-axle slip/saturation proxy. It is **not** a complete ESC stability envelope, rollover model, or proof of safe handling.
- No real-vehicle, hardware-in-the-loop, or production deployment validation is implied.

## Repository layout

- `generate_dataset.py`, `vehicle_model.py`, `maneuvers.py`: simulator and synthetic data generation.
- `data/raw/`: ground-truth simulation labels, sensor measurements, and run metadata.
- `preprocessing/create_forecasting_dataset.py`: run-level train/validation/test split and temporal windows.
- `data/processed/`: sequence archives and prediction arrays.
- `models/`: forecasting baselines, physics observer, and experimental physics-plus-XGBoost residual hybrid.
- `physics/`: rear-slip boundary, margin, and phase-plane analyses.
- `evaluation/`: regression, threshold, event-level, and robustness evaluation.
- `results/tables/`, `results/figures/`, `results/models/`: generated reports and model artifacts.

## Windows / VS Code setup

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a Python version supported by the TensorFlow wheel installed on your machine. If TensorFlow cannot be installed in the current Python version, create the environment with a supported Python version rather than changing the model code.

## Reproduce the pipeline

Run commands from the repository root.

1. Generate the synthetic data:

   ```powershell
   python .\generate_dataset.py
   ```

   Output is written to project-relative `data/raw/` paths. This overwrites the generated raw CSV files, so back up any manually edited data first.

2. Build run-separated forecasting sequences:

   ```powershell
   python .\preprocessing\create_forecasting_dataset.py
   ```

   Splits are made by complete `Run_ID`, not by individual rows. Do not use ground-truth states as model input.

3. Train the baselines and save test predictions:

   ```powershell
   python .\models\xgboost_forecaster.py
   python .\models\lstm_forecaster.py
   python .\models\gru_forecaster.py
   python .\models\hybrid_model.py
   ```

4. Calibrate warning thresholds on validation predictions, then evaluate the untouched test set:

   ```powershell
   python .\evaluation\calibrate_threshold.py
   python .\evaluation\evaluate_test_calibrated.py
   python .\evaluation\event_lead_time.py
   python .\evaluation\robustness_generalization.py
   python -m pytest .\tests
   ```

   Check each script's output and expected input files before running the next one. The event-level evaluator currently reports a fixed zero-margin threshold as a transparent baseline; use the validation-calibrated evaluation for primary threshold-based results. Do not select thresholds from the test set.

## Data and label details

- The simulator integrates at 100 Hz and writes samples at 20 Hz.
- A sequence contains 20 past samples (1 second) of four sensor channels.
- Forecast targets correspond to 0.5 s, 1.0 s, and 2.0 s in the future.
- The instability flag requires three consecutive samples with negative margin. It is a smoothed event label; the continuous margin target and the event flag are not interchangeable.
- Sensor noise is illustrative and should be stress-tested. It is not a substitute for a sensor datasheet or measured noise characterization.
- The simulator omits several real-vehicle effects, including weight transfer, combined longitudinal/lateral tire forces, and roll/suspension dynamics.

## Evaluation principles

- Keep run-level split boundaries intact; never randomly split overlapping windows from the same run across train and test.
- Fit normalization and tune warning thresholds using training/validation data only.
- Report regression metrics per horizon and event metrics separately.
- Event lead time is defined as event onset minus warning onset. Warnings issued after event onset are late warnings, not successful advance detections.
- Report false/unassociated warning episodes alongside event detection rate and lead-time distributions.
- For research claims, regenerate all tables from the current committed prediction files and preserve the exact code/data revision used.

## Current limitations and research checklist

- The physics observer is an approximate sensor-only baseline; lateral-velocity integration can drift.
- The rear-slip saturation boundary is a proxy and should be cross-checked against a validated beta–yaw-rate stability envelope before stronger stability claims.
- `models/hybrid_model.py` implements an experimental current-physics-margin plus learned residual baseline. It must be run and compared against the same held-out test set before claiming it improves performance.
- Run multi-seed experiments and report uncertainty intervals before claiming robust superiority.
- Validate on measured vehicle data before making deployment or safety claims.

## Generated outputs and version control

Generated CSVs, prediction arrays, and trained models may be large. Commit them only when required for reproducibility and check that the artifacts correspond to the same data split and code revision. Never commit credentials or private data.
