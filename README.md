# Near-Limit Vehicle Stability Dataset (synthetic)

Generated from a nonlinear 2-DOF bicycle model with a Pacejka "Magic Formula"
tire model, specifically designed to reach the tire-saturated / near-limit
regime that the original `ieee_bicycle_vehicle_dataset.csv` never entered
(that file only covers gentle highway cruising: steering under ~10°, narrow
speed band, no instability events).

## Why this exists
A stability early-warning model can't learn what instability looks like from
data that never goes unstable. This dataset was built to fix exactly that gap:
288 runs across 4 maneuver types, 3 road-friction levels, 4 speeds, and 6
steering-severity levels, ~63% of which cross into genuine loss-of-control
(rear-tire-saturation) territory.

## Files

### `ground_truth_dynamics.csv` (47,520 rows, 20 Hz) — the answer key
Columns: `Run_ID, Time, maneuver, mu_surface, mu_true, target_speed_kmh,
amplitude_deg, steer_true, yaw_rate, beta, lateral_accel, alpha_f, alpha_r,
stability_margin, instability_flag`

- `beta` — true sideslip angle (rad). This is what real sideslip-estimation
  systems can't measure directly and have to infer — it's your label, not an
  input feature.
- `stability_margin` — `(alpha_r_peak - |alpha_r|) / alpha_r_peak`, i.e.
  normalized distance from the rear axle's peak-grip slip angle. Positive =
  margin remaining, negative = rear tires already past peak grip (the
  classical oversteer/spin trigger in bicycle-model theory). `alpha_r_peak`
  ≈ 0.140 rad (8.0°) here, and is independent of `mu`/load for this tire
  model (a Magic-Formula property, not a coincidence).
- `instability_flag` — True once `stability_margin` has been negative for
  ≥3 consecutive samples (0.15s), to avoid flagging single-sample noise.
- **Never feed this file's `beta`, `stability_margin`, or `instability_flag`
  into a model as input** — they're the targets.

### `retrofit_sensor_stream.csv` (47,520 rows, 20 Hz) — the actual model input
Columns: `Run_ID, Time, yaw_rate_meas, lateral_accel_meas, speed_meas,
steering_angle_meas, maneuver, mu_surface`

This is what a low-cost retrofit sensor kit (MEMS gyro + accelerometer, wheel
-speed sensor, steering potentiometer, no GNSS/INS) would actually observe:
Gaussian noise plus a small per-run constant bias on yaw rate and lateral
accel, quantization on speed (~1 km/h steps), and noise on steering angle.
**These noise levels are illustrative, order-of-magnitude "cheap consumer
IMU"-style values — they are not calibrated against a specific sensor
datasheet.** If you have a target sensor in mind, replace the numbers in
`add_sensor_noise()` in `generate_dataset.py` with its actual noise-density
and bias-instability specs before trusting any results.

Join `retrofit_sensor_stream.csv` (inputs) to `ground_truth_dynamics.csv`
(labels) on `Run_ID` + `Time`.

### `run_metadata.csv` (288 rows) — one row per run
`Run_ID, maneuver, mu_surface, mu_true, target_speed_kmh, amplitude_deg,
duration_s, n_samples, reaches_instability, min_stability_margin`

**Split train/test by `Run_ID`, never by row.** Rows within a run are highly
autocorrelated (same physical trajectory) — splitting by row will leak and
inflate your reported accuracy, the same class of bug as the NGSIM
vehicle-mixing issue from the earlier version of this project.

## Physics model, in brief
- 2-DOF bicycle model (lateral velocity `vy`, yaw rate `r`), longitudinal
  speed held per-run (standard convention for handling tests).
- Pacejka Magic Formula tire model per axle, SAE sign convention (positive
  slip angle → restoring force). Peak lateral force is friction-limited
  (`D = mu × Fz`), which is what actually produces saturation/instability —
  a linear tire model (`Fy = -C·alpha`) can never do this, which is exactly
  why the original dataset couldn't either.
- Vehicle parameters (`vehicle_model.py`): m=1500 kg, Iz=2500 kg·m², lf=1.2 m,
  lr=1.6 m — representative passenger-car values, not fit to a real vehicle.
  Tire shape parameters (B, C, E) are likewise representative, not fit to a
  real tire. **Calibrate both against real vehicle/tire data before treating
  absolute numbers (not just qualitative trends) as meaningful.**
- No weight transfer (static axle loads only), no combined longitudinal+
  lateral load transfer, no roll dynamics, no suspension. This is intentional
  scope-limiting for a first-pass dataset — extending to a 3-DOF or full
  vehicle model is a natural next step if reviewers want it.
- Steering angle here (`steer_true`) is the **front road-wheel angle**, not
  steering-wheel angle. This is why it looks larger than the ~10° max in the
  original dataset — that file's "Steering Angle" was very likely the
  steering-wheel (driver input) angle, off by the steering ratio (~15–20×).

## Stability margin caveat
The rear-axle-saturation margin used here is one standard, defensible
formulation (classical bicycle-model theory: the rear axle saturating first
is the textbook trigger for oversteer/spin). It is **not** the only
convention in the ESC literature — the beta–r phase-plane boundary
(Inagaki et al., 1994) is a commonly cited alternative that also accounts for
yaw-rate limits, not just rear slip angle. Worth implementing as a
cross-check before finalizing which definition you report against.

## Regenerating / modifying
Run `generate_dataset.py` (depends on `vehicle_model.py` and `maneuvers.py`,
NumPy + pandas only). Everything is seeded (`RNG`, `noise_rng`) for
reproducibility. To change scope: `MU_LEVELS`, `SPEED_LEVELS_KMH`,
`AMPLITUDE_DEG`, `MANEUVERS` in `generate_dataset.py` control the run grid.

## Quick start
```python
import pandas as pd
gt = pd.read_csv('ground_truth_dynamics.csv')
sensor = pd.read_csv('retrofit_sensor_stream.csv')
meta = pd.read_csv('run_metadata.csv')

data = sensor.merge(gt[['Run_ID','Time','beta','stability_margin','instability_flag']],
                     on=['Run_ID','Time'])

train_runs = meta.sample(frac=0.8, random_state=0)['Run_ID']
train = data[data['Run_ID'].isin(train_runs)]
test  = data[~data['Run_ID'].isin(train_runs)]
```
