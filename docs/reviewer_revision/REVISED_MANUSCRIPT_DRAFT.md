# Predictive Vehicle Stability Early Warning Using Low-Cost Retrofit Sensors

## Revised manuscript working draft

**Status:** This is a text-first revision draft. It is not yet a submission-ready replacement PDF: numerical regression tables, surface-stratified results, uncertainty intervals, physics-rule baseline, ablations, and hardware latency measurements remain to be generated and verified.

## Abstract

Vehicle lateral instability can develop rapidly, while a warning issued only after an instability indicator crosses its limit may provide little time for a driver to respond. This paper studies a simulation-based, warning-only framework for forecasting a rear-axle tire-saturation-margin proxy from four low-cost retrofit sensor channels. The framework uses a nonlinear two-degree-of-freedom bicycle-model simulator with representative tire-force curves, 20 Hz sensor-stream output, one-second input histories, and forecast horizons of 0.5, 1.0, and 2.0 s. The inputs are measured yaw rate, lateral acceleration, longitudinal speed, and steering angle. XGBoost, LSTM, and GRU forecasters are compared with persistence; an experimental physics-plus-XGBoost residual model is also evaluated. Complete runs are separated into 201 training, 43 validation, and 44 test runs, and warning thresholds are selected using validation data. In the latest event-level evaluation, 39 simulated instability events were extracted. Across the tested model/horizon combinations, 24–29 events were detected before onset (61.5%–74.4%), median lead time among early detections was 0.50–0.90 s, and 6–20 warning episodes were not associated with an event. The Hybrid model at 2.0 s had the highest early-detection count in this snapshot (29/39), but it also produced 17 unassociated warning episodes; therefore, no model is best on every safety-related metric. These results are limited to the current simulator and operating conditions. The sensor noise is illustrative, and no real-vehicle, hardware-in-the-loop, or driver-response validation has been performed.

**Index Terms—** vehicle stability, early warning, retrofit sensors, multi-horizon forecasting, beta–yaw-rate representation, XGBoost, LSTM, GRU, event-level evaluation.

## I. Introduction

Vehicle lateral stability is affected by speed, steering input, tire–road friction, and vehicle dynamics. Near the tire-force limit, a warning based only on the current state may provide limited advance notice. A predictive system could estimate a future stability-related quantity from recent sensor measurements and present a warning before a defined event begins. However, predictive accuracy alone is not sufficient: the warning must precede the event by a useful interval, and excessive unassociated warnings can reduce trust and usability.

Vehicle stability and rollover research includes analytical stability indices, load-transfer measures, vehicle-dynamics models, and data-driven prediction. These methods address different vehicle classes and operating assumptions; their results should not be compared numerically unless the datasets, targets, and evaluation protocols are comparable. Temporal models such as recurrent neural networks can represent sequential dependence, while tree-based models provide a different accuracy–complexity trade-off. None of these model families is novel by itself.

This study evaluates a warning-oriented workflow rather than proposing a new general-purpose forecasting algorithm. The workflow combines (i) a simulated low-cost sensor stream, (ii) a normalized rear-axle tire-saturation-margin proxy, (iii) multi-horizon forecasts, (iv) validation-only warning-threshold calibration, and (v) event-level evaluation that separates advance detections, late warnings, and unassociated warning episodes. An experimental Hybrid comparator adds a sensor-derived current-margin estimate to an XGBoost residual forecaster. The contribution is the reproducible comparison and explicit warning-burden analysis under the stated simulation assumptions; it is not proof of road-safety performance.

The study addresses four questions:
1. How do persistence, XGBoost, LSTM, GRU, and the experimental Hybrid model compare when forecasting the same margin target at 0.5, 1.0, and 2.0 s?
2. How do sample-wise forecasting and classification metrics relate to event-level advance detection?
3. What trade-off exists between early event detection, lead time, and unassociated warning episodes when thresholds are calibrated on validation data?
4. Which limitations prevent the current simulation results from supporting real-vehicle deployment claims?

## II. Related Work

Vehicle-dynamics and stability research uses simplified and high-fidelity models to study sideslip, yaw response, tire saturation, rollover, and load transfer. The beta–yaw-rate (β–r) phase plane is a common way to visualize lateral stability behaviour, while load-transfer ratio and rollover indices address different aspects of vehicle response. A lateral two-degree-of-freedom model does not capture the full roll and load-transfer dynamics needed to claim rollover prediction.

Data-driven methods have been studied for vehicle-state forecasting and rollover-warning tasks, including neural networks and recurrent models. XGBoost, LSTM, and GRU are established algorithms and are used here as comparison methods, not as algorithmic novelties. The relevant methodological distinction in this work is the use of complete-run splits, validation-only threshold selection, and event-level warning metrics alongside regression metrics. Prior studies differ in vehicle type, simulator, target definition, and warning protocol; the comparison should therefore remain qualitative unless a matched benchmark is available.

## III. Methodology

### A. Simulation and sensor assumptions

The data are generated using a nonlinear two-degree-of-freedom bicycle model with representative Pacejka-style tire curves. Longitudinal speed is treated as approximately constant within a simulated run. The model parameters and tire curves are representative assumptions, not fitted to a named production vehicle or measured tire. The simulator integrates at 100 Hz and exports aligned ground-truth and noisy sensor streams at 20 Hz.

The four model inputs are:
- `yaw_rate_meas`: simulated noisy/quantized yaw-rate measurement;
- `lateral_accel_meas`: simulated noisy/quantized lateral acceleration;
- `speed_meas`: simulated longitudinal-speed measurement, representing a wheel-speed-derived channel;
- `steering_angle_meas`: simulated steering-angle measurement.

The sensor noise is illustrative and has not been calibrated against a particular sensor datasheet. The manuscript must list the exact noise and quantization parameters from the source code in the final version. No real sensor hardware or vehicle logging was used for the results reported here.

### B. Target and physical interpretation

The forecast target is a dimensionless rear-axle tire-saturation-margin proxy. In the simulator, rear slip angle α_r is compared with the positive slip angle at which the chosen tire-force curve reaches its peak, α_{r,peak}. The margin is:

[
m(t)=rac{alpha_{r,mathrm{peak}}-|alpha_r(t)|}{alpha_{r,mathrm{peak}}}.
]

A positive margin indicates that the simulated rear slip angle is below the chosen peak-slip reference; a negative margin indicates that it has passed that reference. This is a simulator-specific proxy, not a validated complete vehicle-stability envelope, rollover index, or probability of a crash. The continuous target is forecast at +0.5, +1.0, and +2.0 s.

The β–r state uses sideslip angle β and yaw rate r. In simulation, β is computed from the simulator's lateral and longitudinal velocities. Those ground-truth states are used to construct labels and evaluate the proxy; they are not among the four model inputs. Any sideslip estimate reconstructed from low-cost measurements is approximate and can drift, particularly if lateral acceleration is integrated over time. The paper must not imply direct measurement of lateral velocity by the stated retrofit sensors.

### C. Forecasting setup

Each input sequence contains 20 samples (one second at 20 Hz) from the four sensor channels. The targets are the future margin values at the three specified horizons. Complete runs, not overlapping windows, are split into training, validation, and test partitions: 201, 43, and 44 runs, respectively. This reduces leakage between windows from the same simulated trajectory. The current held-out test archive contains 4,584 sequences.

The compared methods are persistence, XGBoost, LSTM, and GRU. The current repository also contains an experimental Hybrid model: a sensor-derived current-margin estimate is combined with an XGBoost prediction of the future-margin residual. Because numerical performance must be compared using the same target arrays, horizon alignment, and metric implementation, all final MAE/RMSE/R² values must be regenerated together before the final manuscript table is frozen.

MAE is the mean absolute error in the dimensionless margin; RMSE is the square root of the mean squared error in the same units and gives greater weight to large errors; R² compares the residual sum of squares with target variance and may be negative when a model is worse than predicting the test mean. These metrics describe regression fit, not warning utility or safety.

### D. Warning calibration and event evaluation

Warning thresholds are calibrated using validation predictions only and are then frozen for test evaluation. The calibration implementation searches candidate thresholds from −0.50 to +0.50 in increments of 0.05. The final paper must state the exact objective and tie-breaking rule from the calibration code; this draft does not assume that a threshold maximizes a particular metric until that code path is verified.

The simulator's instability flag uses a sustained negative-margin condition. At 20 Hz, three consecutive samples correspond to 0.15 s. In the current event evaluator, contiguous true-flag samples within each run form a ground-truth event. A warning episode is formed from predicted margins below the calibrated threshold, requiring two consecutive warning samples. The evaluator groups warning samples using a 0.20 s gap parameter. A warning counts as an early detection only if it starts at or before the event onset and within the 5 s pre-event association window. Warnings starting after onset but before the event end are counted separately as late-only. Each warning episode can be matched to at most one event; any unmatched episode is counted as unassociated.

Point-wise metrics count individual samples. Event-level detection asks whether a warning episode precedes a ground-truth event. These metrics can differ because a model can classify many samples correctly yet issue its warning too close to or after event onset; conversely, a small number of warning episodes may precede events even if point-wise recall is modest. Event-level and point-wise tables should not be interpreted as interchangeable measures.

## IV. Experimental Setup

The dataset contains 47,520 synchronized simulated samples across 288 runs. The run-level split comprises 201 training runs, 43 validation runs, and 44 test runs. The held-out test set contains 4,584 temporal sequences. The simulation varies maneuver, nominal road-friction category (dry, wet, and ice/snow), target speed, and steering amplitude. Final results should include the exact number of held-out runs and events for each surface; those counts must be computed from the current metadata rather than inferred from the overall split.

The test set is not a sample of independent real vehicles. It does not establish transfer across vehicle geometries, payloads, production tire characteristics, road conditions, or actual retrofit sensor hardware. The simulator does not model dynamic load transfer, combined longitudinal/lateral tire forces, or roll/suspension dynamics. For these reasons, an LTR threshold cannot be claimed as a validated baseline from the present lateral model alone.

## V. Results and Discussion

### A. Event-level results

The latest local event-level evaluation extracts 39 simulated ground-truth events. The early-detection results are:

| Model | Horizon | Early detections | Early detection rate | Median lead (s) | Unassociated warning episodes |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.5 s | 26/39 | 66.7% | 0.850 | 20 |
| XGBoost | 1.0 s | 26/39 | 66.7% | 0.825 | 17 |
| XGBoost | 2.0 s | 25/39 | 64.1% | 0.850 | 17 |
| LSTM | 0.5 s | 24/39 | 61.5% | 0.500 | 7 |
| LSTM | 1.0 s | 25/39 | 64.1% | 0.600 | 6 |
| LSTM | 2.0 s | 25/39 | 64.1% | 0.650 | 9 |
| GRU | 0.5 s | 26/39 | 66.7% | 0.700 | 12 |
| GRU | 1.0 s | 25/39 | 64.1% | 0.850 | 18 |
| GRU | 2.0 s | 27/39 | 69.2% | 0.850 | 15 |
| Hybrid | 0.5 s | 26/39 | 66.7% | 0.800 | 14 |
| Hybrid | 1.0 s | 26/39 | 66.7% | 0.875 | 15 |
| Hybrid | 2.0 s | 29/39 | 74.4% | 0.900 | 17 |

These counts indicate a trade-off, not a universal ranking. Hybrid at 2.0 s has the largest early-detection count in this snapshot, but 17 unassociated warning episodes remain. LSTM at 1.0 s has fewer unassociated episodes (6) but detects 25/39 events early. The median lead times of 0.50–0.90 s are short and may be insufficient for dependable driver intervention. They are conditional on early-detected events; they do not mean that all events receive a warning.

The earlier manuscript reported approximately 97% event detection using an older event count and evaluation definition. That result is not combined with the current table. Before submission, the previous evaluator and output must be archived or explicitly identified so the change in event extraction, warning association, and denominator is transparent.

### B. Forecasting metrics and persistence

The older forecasting comparison table is dated 22 September 2026, whereas the current event-level output was regenerated on 20 October 2026. The old regression table must be treated as provisional until MAE, RMSE, and R² are recomputed for persistence, XGBoost, LSTM, GRU, and Hybrid from the same held-out target array and horizon indices. MAE and RMSE can rank models differently because RMSE penalizes large residuals more strongly; this explains how rankings can differ in principle, but the calculation itself must still be checked.

No claim that learned models consistently outperform persistence is justified. Where persistence has lower RMSE or R² is better for persistence, the paper should report that directly and discuss the limited incremental value at that horizon.

### C. Statistical uncertainty and robustness

The current event-level table is one evaluation snapshot from a fixed test split. It does not yet contain run-level bootstrap confidence intervals or multiple-seed retraining variability. The revised experiment should resample test runs as clusters to preserve within-run temporal dependence and should repeat training with several predeclared seeds. Surface-stratified and sensor-noise robustness results should be reported by model and horizon after the relevant scripts have been run and their output checked. Until then, no interval estimates or robust-superiority claims are made.

### D. Physics baselines, ablations and latency

The current Hybrid is a sensor-only approximate current-margin estimator plus an XGBoost residual predictor. It is not equivalent to an independent, validated β–r boundary classifier. A defensible physics-rule comparison requires a stated boundary selected independently of test outcomes and consistent event-level calibration. A load-transfer-ratio baseline is not directly supported by the current 2-DOF lateral simulator because it lacks roll/suspension dynamics. Ablation experiments should isolate the physics estimate and residual learner using the same run split. No low-cost onboard hardware latency or memory measurement is currently verified.

## VI. Limitations

This study uses synthetic data from a simplified lateral vehicle model. The tire parameters are representative rather than fitted to measured tires; sensor noise is illustrative rather than datasheet-calibrated; and the model omits dynamic load transfer, combined tire forces, and roll/suspension effects. The target is a rear-axle slip/saturation-margin proxy, not a validated complete stability envelope or rollover measure. The 44-run test partition represents held-out simulator runs, not independent real vehicles. No real-vehicle, hardware-in-the-loop, driver-response, or onboard compute validation has been performed. Short median warning lead times and unassociated warning episodes further limit claims of practical usefulness. The system should therefore be described as a research prototype for warning evaluation, not a deploy-ready safety product.

## VII. Conclusion

This paper evaluates multi-horizon prediction of a simulated rear-axle tire-saturation-margin proxy from four noisy retrofit-style sensor channels. Run-level splitting and validation-only threshold selection reduce leakage risk, while event-level evaluation distinguishes early warnings from late and unassociated episodes. In the latest 39-event evaluation snapshot, early detection ranges from 61.5% to 74.4%, with median lead times of 0.50–0.90 s and non-negligible unassociated warning counts. Hybrid at the 2.0 s horizon has the largest early-detection count in this snapshot, but it does not dominate every warning metric. These results do not establish sufficient driver-reaction time, transfer to real vehicles, or safety improvement. Future work must include repeated-seed experiments, run-level uncertainty intervals, physics-rule baselines, surface-stratified analysis, ablations, hardware latency tests, and validation with measured vehicle data.

## References

Retain and verify the references from the source manuscript. Expand the related-work section with carefully checked sources on beta–yaw-rate stability boundaries, load-transfer/rollover indices, vehicle stability forecasting, and event-based early-warning evaluation. Do not add citations unless their bibliographic details and relevance have been checked.
