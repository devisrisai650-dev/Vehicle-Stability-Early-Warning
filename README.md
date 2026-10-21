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

   Check each script's output and expected input files before running the next one. The event-level evaluator reads the per-model, per-horizon thresholds produced by validation calibration. If that file is missing, it prints a warning and falls back to the zero-margin physics boundary. Do not select thresholds from the test set.

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

## Latest event-level evaluation (20 October 2026)

The current simulated test evaluation contains 39 ground-truth instability events. Warning thresholds were selected using validation predictions. The table below summarizes early event detection, median lead time, and unassociated warning episodes; the full per-model/per-horizon metrics and thresholds are recorded in [`docs/event_level_results_2026-10-20.csv`](docs/event_level_results_2026-10-20.csv).

| Model | Horizon | Early detections | Detection rate | Median lead time (s) | Unassociated warning episodes |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.5 s | 26/39 | 66.7% | 0.85 | 20 |
| XGBoost | 1.0 s | 26/39 | 66.7% | 0.83 | 17 |
| XGBoost | 2.0 s | 25/39 | 64.1% | 0.85 | 17 |
| LSTM | 0.5 s | 24/39 | 61.5% | 0.50 | 7 |
| LSTM | 1.0 s | 25/39 | 64.1% | 0.60 | 6 |
| LSTM | 2.0 s | 25/39 | 64.1% | 0.65 | 9 |
| GRU | 0.5 s | 26/39 | 66.7% | 0.70 | 12 |
| GRU | 1.0 s | 25/39 | 64.1% | 0.85 | 18 |
| GRU | 2.0 s | 27/39 | 69.2% | 0.85 | 15 |
| Hybrid | 0.5 s | 26/39 | 66.7% | 0.80 | 14 |
| Hybrid | 1.0 s | 26/39 | 66.7% | 0.88 | 15 |
| Hybrid | 2.0 s | 29/39 | 74.4% | 0.90 | 17 |

These are simulation-only results from the current evaluation run, not real-vehicle validation. The Hybrid model at 2.0 s has the highest early-detection count in this run, but it also produces unassociated warnings; model choice should consider false warnings and lead-time distribution, not detection rate alone. Do not compare these event metrics with stale CSVs from earlier evaluator versions.

Validation performed for this result snapshot: event counts were consistent at 39 across all model/horizon rows; the detail CSV's early-detection counts agreed with non-empty lead-time records; false-warning detail counts matched the summary for all 12 model/horizon combinations; test timestamps were uniformly spaced at approximately 0.05 s; and the current event-lead-time regression tests passed (4 tests). These checks establish internal consistency, not independent validation of the simulator or warning policy.

## Current limitations and research checklist

- The physics observer is an approximate sensor-only baseline; lateral-velocity integration can drift.
- The rear-slip saturation boundary is a proxy and should be cross-checked against a validated beta–yaw-rate stability envelope before stronger stability claims.
- `models/hybrid_model.py` implements an experimental current-physics-margin plus learned residual baseline. It must be run and compared against the same held-out test set before claiming it improves performance.
- Run multi-seed experiments and report uncertainty intervals before claiming robust superiority.
- Validate on measured vehicle data before making deployment or safety claims.

## Generated outputs and version control

Generated CSVs, prediction arrays, and trained models may be large. Commit them only when required for reproducibility and check that the artifacts correspond to the same data split and code revision. Never commit credentials or private data.

## Reviewer revision experiments

After the existing processed train/validation/test sequence archives and sensor stream are available, run the reproducible repeated-seed regression/ablation experiment:

```powershell
python .\\evaluation\\repeated_seed_ablation_experiments.py
```

It trains sensor-only XGBoost and physics-plus-XGBoost residual models with five predeclared seeds (11, 23, 37, 42, 58), and compares both against a sensor-derived physics-only current-margin persistence comparator. It writes per-seed metrics and mean/standard-deviation summaries to `results/tables/reviewer_revision/repeated_seed_ablation_raw.csv` and `repeated_seed_ablation_summary.csv`. The author ran the script on the current 44-run/4,584-sequence test split; the console summary is preserved as `docs/reviewer_revision/repeated_seed_ablation_summary_2026-10-21.csv`. The per-seed raw CSV remains a local generated artifact and should be archived with the experiment outputs before final submission. It does not retrain LSTM/GRU, run event-warning threshold ablations, or measure hardware latency.

### Matched event-warning ablation

After validation and test prediction files exist, run:

```powershell
python .\\evaluation\\event_warning_ablation.py
```

The script compares sensor-only XGBoost, a sensor-derived physics-only observer, and physics-plus-residual Hybrid. It selects each method/horizon threshold on validation runs only, maximizing early-event detection subject to at most 0.5 unassociated warning episodes per validation run; ties are resolved by fewer unassociated episodes and then a higher threshold. It freezes those thresholds before test evaluation and writes validation threshold/grid and test results to `results/tables/reviewer_revision/event_warning_ablation_*.csv`. The script has been added but not yet executed locally; do not cite numeric results until outputs have been reviewed. If no threshold satisfies the warning-burden cap, the output explicitly marks the constraint infeasible and chooses the lowest-burden threshold.
