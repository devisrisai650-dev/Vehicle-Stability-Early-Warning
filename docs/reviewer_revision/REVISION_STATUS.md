# CICT-2026 Reviewer Revision Status

Paper ID: 693  
Title: *Predictive Vehicle Stability Early Warning Using Low-Cost Retrofit Sensors*  
Reviewers: #4 and #6

## Purpose

This document tracks the revision against the actual repository implementation and current event-level output. It deliberately separates completed code/evaluation evidence from experiments that have not yet been run. Do not submit this checklist as evidence that all requested experiments are complete.

## Verified from the current repository / latest local console output

- Synthetic-data pipeline: nonlinear 2-DOF bicycle-model simulation with representative, not vehicle-fitted, Pacejka tire curves.
- The simulator writes at 20 Hz after internal 100 Hz integration.
- Inputs: `yaw_rate_meas`, `lateral_accel_meas`, `speed_meas`, `steering_angle_meas`.
- Targets: simulated rear-axle stability-margin proxy at 0.5, 1.0 and 2.0 s.
- Run-level split: 201 train, 43 validation, 44 test runs; test sequence count reported as 4,584.
- Validation-only threshold calibration exists; event warning policy uses two consecutive warning samples, a 0.20 s episode gap, and a maximum 5 s pre-event association window.
- Current event evaluator reports 39 ground-truth events and 12 model/horizon combinations.
- The four event-lead-time unit tests passed in the user's latest local run.
- A local surface-stratified run completed for XGBoost, LSTM, GRU, and Hybrid: dry 10 runs/1,010 sequences; ice/snow 16 runs/1,696 sequences; wet 18 runs/1,878 sequences. Its 36 metric rows are preserved in `surface_stratified_metrics_2026-10-21.csv`; they are provisional until the updated script reruns with persistence and sensor-only physics baselines included.
- The reviewer analysis script now builds persistence from the current simulated ground-truth margin and computes a sensor-only physics-observer baseline. Persistence is a reference baseline using simulator state, not an operational sensor-only method.
- The author successfully ran the optimized run-cluster bootstrap for 2,000 resamples per model/horizon. Output is committed as `results/tables/reviewer_revision/run_cluster_bootstrap_intervals.csv`.
- The author successfully ran the full surface-stratified comparison including persistence and sensor-only physics observer. Output is committed as `results/tables/reviewer_revision/surface_stratified_forecasting_metrics.csv`.
- The Hybrid model is experimental: sensor-derived current-margin estimate plus XGBoost residual forecast. It is not a validated vehicle controller.
- The repository documentation explicitly says the simulator's sensor-noise parameters are illustrative and not calibrated against a sensor datasheet.

## Latest event-level snapshot (local run reported by the author)

| Model | Horizon | Early detections / 39 | Missed | Late-only | Median lead (s) | Unassociated warnings |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.5 s | 26 | 13 | 9 | 0.850 | 20 |
| XGBoost | 1.0 s | 26 | 13 | 9 | 0.825 | 17 |
| XGBoost | 2.0 s | 25 | 14 | 10 | 0.850 | 17 |
| LSTM | 0.5 s | 24 | 15 | 9 | 0.500 | 7 |
| LSTM | 1.0 s | 25 | 14 | 9 | 0.600 | 6 |
| LSTM | 2.0 s | 25 | 14 | 9 | 0.650 | 9 |
| GRU | 0.5 s | 26 | 13 | 9 | 0.700 | 12 |
| GRU | 1.0 s | 25 | 14 | 9 | 0.850 | 18 |
| GRU | 2.0 s | 27 | 12 | 9 | 0.850 | 15 |
| Hybrid | 0.5 s | 26 | 13 | 9 | 0.800 | 14 |
| Hybrid | 1.0 s | 26 | 13 | 9 | 0.875 | 15 |
| Hybrid | 2.0 s | 29 | 10 | 9 | 0.900 | 17 |

Rates are counts divided by 39; the table is a simulation-only snapshot, not field validation. Hybrid at 2 s has the largest early-detection count in this snapshot, but its warning burden is non-trivial. LSTM at 1 s has fewer unassociated warning episodes (6) but detects 25/39 events early. Do not call any model universally best based on one metric.

## Important reconciliation issue

The submitted manuscript's abstract/conclusion report approximately 97% event detection based on 35–36 events and short mean lead times. The current evaluator uses 39 extracted events and a stricter definition: a warning counts as early only if its onset is no later than event onset and within the preceding 5 s; post-onset warnings are late-only. These figures are not directly comparable until the older evaluation code/output and the current evaluator's definitions are reconciled. The manuscript must not mix them. Use the current 39-event snapshot for provisional discussion, and explicitly state the evaluation definition/version.

The current event-level output is available locally in `results/tables/event_lead_time_metrics.csv`; generated CSVs and prediction arrays are not all tracked in GitHub, so this branch does not claim that the local binary artifacts were independently rerun by CI.

## Experiments still required before claiming full reviewer closure

1. The single-source comparison has now run for persistence, sensor-only physics observer, XGBoost, LSTM, GRU, and Hybrid. Results are committed; inspect target-scale consistency before final paper submission.
2. Surface-stratified metrics are committed for all six methods and three horizons.
3. The 2,000-resample run-level bootstrap completed and its confidence-interval CSV is committed. Interpret intervals as fixed-model test-run sampling uncertainty only.
4. Repeat training with several random seeds and report variability. Do not treat bootstrap resampling as a substitute for retraining across seeds.
5. Evaluate a physics-rule baseline only if the required beta/yaw-rate signals and boundary are available. An LTR baseline is not supported by the current 2-DOF lateral model unless roll/load-transfer data or a defensible simulated extension is added.
6. Run ablations that isolate the physics margin and residual learner using the same split and thresholds calibrated on validation data.
7. Measure inference latency and memory on specified low-cost hardware. No such measurement is currently verified.
8. Add sensor-noise robustness results and include Hybrid consistently if its model artifacts and inference path are available.
9. Update the paper tables, figures, abstract and conclusion only after those result files have been regenerated and checked.

## Manuscript claim boundary

Permitted: “simulation-based study,” “rear-axle tire-saturation-margin proxy,” “illustrative sensor noise,” “run-level held-out test set,” “experimental Hybrid baseline.”

Not supported: “validated road-safety system,” “deploy-ready,” “sufficient time for driver intervention,” “real-world sensor performance,” or “superior overall performance” without further evidence.

## Execution checklist

- [ ] Reconcile old and current event definitions/results.
- [ ] Regenerate a single-source forecasting comparison for all models/horizons.
- [ ] Run surface-stratified evaluation.
- [ ] Run run-level bootstrap intervals.
- [ ] Run multiple training seeds.
- [ ] Run physics-rule baseline or document why the available data/model do not support it.
- [ ] Run ablation study.
- [ ] Benchmark hardware latency/memory.
- [ ] Update manuscript from verified output tables.
- [ ] Run full tests and confirm reproducibility.
- [ ] Finalize point-by-point response with page/section locations.
