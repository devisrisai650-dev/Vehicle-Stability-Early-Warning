# Predictive Vehicle Stability Early Warning Using Low-Cost Retrofit Sensors

## I. Abstract

Vehicle lateral instability can develop rapidly, while a warning issued only after an instability indicator crosses its limit may provide little time for a driver to respond. This paper studies a simulation-based, warning-only framework for forecasting a rear-axle tire-saturation-margin proxy from four low-cost retrofit sensor channels. The framework uses a nonlinear two-degree-of-freedom bicycle-model simulator with representative tire-force curves, 20 Hz sensor-stream output, one-second input histories, and forecast horizons of 0.5, 1.0, and 2.0 s. The inputs are simulated noisy/quantized measurements of yaw rate, lateral acceleration, longitudinal speed, and steering angle. XGBoost, LSTM, and GRU forecasters are compared with persistence; an experimental physics-plus-XGBoost residual model is also evaluated. Complete runs are separated into 201 training, 43 validation, and 44 test runs, and warning thresholds are selected using validation data. In the latest event-level evaluation, 39 simulated instability events were extracted. Across the tested model/horizon combinations, 24–29 events were detected before onset (61.5%–74.4%), median lead time among early detections was 0.50–0.90 s, and 6–20 warning episodes were not associated with an event. The Hybrid model at 2.0 s had the highest early-detection count in this snapshot (29/39), but it also produced 17 unassociated warning episodes; therefore, no model is best on every safety-related metric. These results are limited to the current simulator and operating conditions. The sensor noise is illustrative, and no real-vehicle, hardware-in-the-loop, or driver-response validation has been performed.

**Index Terms—** vehicle stability, early warning, retrofit sensors, multi-horizon forecasting, beta–yaw-rate representation, XGBoost, LSTM, GRU, event-level evaluation.

## II. Introduction

Vehicle lateral stability is affected by speed, steering input, tire–road friction, and vehicle dynamics [1], [2]. Near the tire-force limit, a warning based only on the current state may provide limited advance notice. A predictive system could estimate a future stability-related quantity from recent sensor measurements and present a warning before a defined event begins. However, predictive accuracy alone is not sufficient: the warning must precede the event by a useful interval, and excessive unassociated warnings can reduce trust and usability.

Vehicle stability research includes analytical stability indices, phase-plane methods, load-transfer measures, vehicle-dynamics models, and data-driven prediction [1]–[3]. These methods address different vehicle classes and operating assumptions; their results should not be compared numerically unless the datasets, targets, and evaluation protocols are comparable. Temporal models such as LSTM and GRU networks can represent sequential dependence [5], [6], while tree-based boosting and ensemble methods provide different accuracy–complexity trade-offs [4], [7], [8]. None of these model families is novel by itself.

This study evaluates a warning-oriented workflow rather than proposing a new general-purpose forecasting algorithm. The workflow combines (i) a simulated low-cost sensor stream, (ii) a normalized rear-axle tire-saturation-margin proxy, (iii) multi-horizon forecasts, (iv) validation-only warning-threshold calibration, and (v) event-level evaluation that separates advance detections, late warnings, and unassociated warning episodes. An experimental Hybrid comparator adds a sensor-derived current-margin estimate to an XGBoost residual forecaster. The contribution is the reproducible comparison and explicit warning-burden analysis under the stated simulation assumptions; it is not proof of road-safety performance.

The study addresses four questions:
1. How do persistence, XGBoost, LSTM, GRU, and the experimental Hybrid model compare when forecasting the same margin target at 0.5, 1.0, and 2.0 s?
2. How do sample-wise forecasting and classification metrics relate to event-level advance detection?
3. What trade-off exists between early event detection, lead time, and unassociated warning episodes when thresholds are calibrated on validation data?
4. Which limitations prevent the current simulation results from supporting real-vehicle deployment claims?

## III. Related Works

Vehicle-dynamics and stability research uses simplified and high-fidelity models to study sideslip, yaw response, tire saturation, rollover, and load transfer. The beta–yaw-rate (β–r) phase plane is a common way to visualize lateral stability behaviour, while load-transfer ratio and rollover indices address different aspects of vehicle response. A lateral two-degree-of-freedom model does not capture the full roll and load-transfer dynamics needed to claim rollover prediction.

Data-driven methods have been studied for vehicle-state forecasting and rollover-warning tasks, including neural networks and recurrent models. XGBoost, LSTM, and GRU are established algorithms and are used here as comparison methods, not as algorithmic novelties. The relevant methodological distinction in this work is the use of complete-run splits, validation-only threshold selection, and event-level warning metrics alongside regression metrics. Prior studies differ in vehicle type, simulator, target definition, and warning protocol; the comparison should therefore remain qualitative unless a matched benchmark is available.

## IV. Methods

### A. Simulation and sensor assumptions

The data are generated using a nonlinear two-degree-of-freedom bicycle model with representative Pacejka-style tire curves [1], [2]. Longitudinal speed is treated as approximately constant within a simulated run. The model parameters and tire curves are representative assumptions, not fitted to a named production vehicle or measured tire. The simulator integrates at 100 Hz and exports aligned ground-truth and noisy sensor streams at 20 Hz.

The four model inputs are:
- `yaw_rate_meas`: simulated noisy/quantized yaw-rate measurement;
- `lateral_accel_meas`: simulated noisy/quantized lateral acceleration;
- `speed_meas`: simulated longitudinal-speed measurement, representing a wheel-speed-derived channel;
- `steering_angle_meas`: simulated steering-angle measurement.

The sensor noise is illustrative and has not been calibrated against a particular sensor datasheet. The manuscript must list the exact noise and quantization parameters from the source code in the final version. No real sensor hardware or vehicle logging was used for the results reported here.

### B. Target and physical interpretation

The forecast target is a dimensionless rear-axle tire-saturation-margin proxy. In the simulator, rear slip angle α_r is compared with the positive slip angle at which the chosen tire-force curve reaches its peak, α_{r,peak}. The margin is:

\[
m(t)=\frac{\alpha_{r,\mathrm{peak}}-|\alpha_r(t)|}{\alpha_{r,\mathrm{peak}}}.
\]

A positive margin indicates that the simulated rear slip angle is below the chosen peak-slip reference; a negative margin indicates that it has passed that reference. This is a simulator-specific proxy, not a validated complete vehicle-stability envelope, rollover index, or probability of a crash. The continuous target is forecast at +0.5, +1.0, and +2.0 s.

The β–r state uses sideslip angle β and yaw rate r; phase-plane methods have been used to characterize vehicle behaviour near critical cornering [3]. In simulation, β is computed from the simulator's lateral and longitudinal velocities. Those ground-truth states are used to construct labels and evaluate the proxy; they are not among the four model inputs. Any sideslip estimate reconstructed from low-cost measurements is approximate and can drift, particularly if lateral acceleration is integrated over time. The paper must not imply direct measurement of lateral velocity by the stated retrofit sensors.

### C. Forecasting setup

Each input sequence contains 20 samples (one second at 20 Hz) from the four sensor channels. The targets are the future margin values at the three specified horizons. Complete runs, not overlapping windows, are split into training, validation, and test partitions: 201, 43, and 44 runs, respectively. Grouping by run reduces leakage risk from correlated windows and is consistent with the need for careful out-of-sample evaluation in time-series prediction [9]. This reduces leakage between windows from the same simulated trajectory. The current held-out test archive contains 4,584 sequences.

The compared methods are persistence, XGBoost, LSTM, and GRU. The current repository also contains an experimental Hybrid model: a sensor-derived current-margin estimate is combined with an XGBoost prediction of the future-margin residual. The implementation uses established model families and is not presented as a new learning algorithm [4]–[8], [12]–[14]. The five-seed experiment trains sensor-only XGBoost and this Hybrid implementation with seeds 11, 23, 37, 42, and 58, while keeping the same held-out test split. The physics-only comparator holds the sensor-derived current-margin estimate constant across horizons. These repeated-seed results are separate from the previously reported event-level results, which used fixed model artifacts and thresholds.

MAE is the mean absolute error in the dimensionless margin; RMSE is the square root of the mean squared error in the same units and gives greater weight to large errors; R² compares the residual sum of squares with target variance and may be negative when a model is worse than predicting the test mean. Error metrics must be interpreted in the context of the target and comparison design [10]. These metrics describe regression fit, not warning utility or safety.

### D. Warning calibration and event evaluation

Warning thresholds are calibrated using validation predictions only and are then frozen for test evaluation. The calibration implementation searches candidate thresholds from −0.50 to +0.50 in increments of 0.05. The matched event-warning ablation separately calibrates a threshold for each method and horizon on validation runs. It selects the candidate with the highest early-event detection rate among candidates with no more than 0.5 unassociated warning episodes per validation run; ties are resolved by fewer unassociated episodes and then the higher threshold. If no candidate meets the burden cap, the implementation selects the lowest-burden candidate and flags infeasibility. In the latest run, all nine method/horizon settings met the validation cap. The frozen thresholds were then applied to the 44-run test set.

The simulator's instability flag uses a sustained negative-margin condition. At 20 Hz, three consecutive samples correspond to 0.15 s. In the current event evaluator, contiguous true-flag samples within each run form a ground-truth event. A warning episode is formed from predicted margins below the calibrated threshold, requiring two consecutive warning samples. The evaluator groups warning samples using a 0.20 s gap parameter. A warning counts as an early detection only if it starts at or before the event onset and within the 5 s pre-event association window. Warnings starting after onset but before the event end are counted separately as late-only. Each warning episode can be matched to at most one event; any unmatched episode is counted as unassociated.

Point-wise metrics count individual samples. Since event-warning evaluation is sensitive to the balance between detected events and unassociated alarms, reporting several complementary measures is preferable to relying on a single classification score [11]. Event-level detection asks whether a warning episode precedes a ground-truth event. These metrics can differ because a model can classify many samples correctly yet issue its warning too close to or after event onset; conversely, a small number of warning episodes may precede events even if point-wise recall is modest. Event-level and point-wise tables should not be interpreted as interchangeable measures.

### D. Dataset and experimental setup

The dataset contains 47,520 synchronized simulated samples across 288 runs. The run-level split comprises 201 training runs, 43 validation runs, and 44 test runs. The held-out test set contains 4,584 temporal sequences. The simulation varies maneuver, nominal road-friction category (dry, wet, and ice/snow), target speed, and steering amplitude. In the author's local surface-stratified run, the test partition contained 10 dry runs (1,010 sequences), 16 ice/snow runs (1,696 sequences), and 18 wet runs (1,878 sequences). The 36 provisional XGBoost/LSTM/GRU/Hybrid metric rows are recorded in `surface_stratified_metrics_2026-10-21.csv`. The updated script includes persistence and the sensor-only physics observer; the final local run regenerated these metrics.

The test set is not a sample of independent real vehicles. It does not establish transfer across vehicle geometries, payloads, production tire characteristics, road conditions, or actual retrofit sensor hardware. The simulator does not model dynamic load transfer, combined longitudinal/lateral tire forces, or roll/suspension dynamics. For these reasons, an LTR threshold cannot be claimed as a validated baseline from the present lateral model alone.

## V. Results and Discussion

### A. Event-level results

The latest local event-level evaluation extracts 39 simulated ground-truth events. The event-level snapshot below uses the model thresholds in the existing evaluation artifacts; it is distinct from the new validation-calibrated event-warning ablation table later in this section. The early-detection results are:

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

The updated local evaluation computes MAE, RMSE, and R² for persistence, XGBoost, LSTM, GRU, Hybrid, and a sensor-only physics observer on the same held-out target array, and saves the 54 surface/model/horizon rows in `results/tables/reviewer_revision/surface_stratified_forecasting_metrics.csv`. Across the surface-stratified output, the Hybrid model has the lowest MAE on ice/snow and wet surfaces at all three horizons, and on dry at 0.5 s and 1.0 s; at dry 2.0 s, LSTM has slightly lower MAE than Hybrid (0.5616 vs. 0.5631). The persistence reference has lower MAE than the learned models on dry at 0.5 s (0.2989), but its MAE rises with horizon and is worse than the best learned model on ice/snow and wet conditions. This persistence reference uses the simulator's current ground-truth margin, so it is not a sensor-available deployment baseline. The physics observer is sensor-only but approximate; its error is similar to persistence and grows substantially at longer horizons. The complete table should be used rather than presenting a universal model winner.

### C. Statistical uncertainty and robustness

The event-level estimates now have run-cluster bootstrap intervals based on 2,000 resamples of the 44 held-out test runs; bootstrap methods provide a way to estimate sampling uncertainty from resampling units [16]. The output is saved in `results/tables/reviewer_revision/run_cluster_bootstrap_intervals.csv`. Hybrid at 2.0 s has an early-detection estimate of 0.7436 (95% bootstrap interval 0.5952–0.8781), 0.3864 unassociated warning episodes per test run (interval 0.2040–0.5909), and median lead time 0.90 s (interval 0.50–1.50 s). LSTM at 1.0 s has an early-detection estimate of 0.6410 (interval 0.4889–0.7956) and the lowest unassociated-episode estimate among these configurations, 0.1364 per test run (interval 0.0455–0.2500). Intervals overlap across several model/horizon pairs, so the results do not establish a clear statistical winner. These intervals represent test-run resampling for fixed trained models and thresholds only; they do not quantify retraining-seed variability or real-vehicle generalization. Repeated-seed training remains unrun.




### D. Validation-calibrated event-warning ablation

A matched event-warning ablation compared the sensor-derived physics-only observer, sensor-only XGBoost, and physics-plus-residual Hybrid. For each method and horizon, the warning threshold was selected on validation runs only by maximizing early-event detection subject to at most 0.5 unassociated warning episodes per validation run; ties were resolved by fewer unassociated episodes and then a higher threshold. All nine configurations met this validation constraint. The selected thresholds and test results were saved to `event_warning_ablation_validation_thresholds.csv` and `event_warning_ablation_test_results.csv` in the local `results/tables/reviewer_revision/` directory.

| Method | Horizon | Frozen threshold | Early detections | Rate | Median lead (s) | Late-only events | Unassociated episodes | Unassociated episodes/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Physics-only sensor observer | 0.5 s | 0.35 | 26/39 | 66.7% | 0.400 | 9 | 9 | 0.205 |
| Physics-only sensor observer | 1.0 s | 0.35 | 26/39 | 66.7% | 0.400 | 9 | 9 | 0.205 |
| Physics-only sensor observer | 2.0 s | 0.35 | 26/39 | 66.7% | 0.400 | 9 | 9 | 0.205 |
| Sensor-only XGBoost | 0.5 s | 0.45 | 26/39 | 66.7% | 0.850 | 9 | 20 | 0.455 |
| Sensor-only XGBoost | 1.0 s | 0.30 | 25/39 | 64.1% | 0.800 | 9 | 21 | 0.477 |
| Sensor-only XGBoost | 2.0 s | 0.10 | 25/39 | 64.1% | 0.850 | 10 | 17 | 0.386 |
| Hybrid physics + residual | 0.5 s | 0.50 | 26/39 | 66.7% | 0.800 | 9 | 14 | 0.318 |
| Hybrid physics + residual | 1.0 s | 0.45 | 26/39 | 66.7% | 0.875 | 9 | 15 | 0.341 |
| Hybrid physics + residual | 2.0 s | 0.20 | 29/39 | 74.4% | 0.900 | 9 | 17 | 0.386 |

Under this calibration rule, the Hybrid at 2.0 s produced the highest early-detection count (29/39) among the compared configurations. The physics-only observer produced fewer unassociated episodes but shorter median lead (0.40 s). XGBoost at 1.0 s had 21 unassociated episodes and a test burden of 0.477 per run, while the Hybrid at 2.0 s had 17 and 0.386 per run. These results illustrate a detection/lead-time/warning-burden trade-off; they do not establish a universal winner. The validation cap is a threshold-selection constraint, not a guarantee that the test burden will remain below 0.5 in future conditions. The ablation is simulation-only and uses the same held-out split and simulator-derived event labels.

### E. Repeated-seed regression and ablation

A follow-up experiment trained sensor-only XGBoost and the physics-plus-XGBoost residual Hybrid with five predeclared random seeds (11, 23, 37, 42, and 58), using the same 201/43/44 run-level split and the 4,584-sequence held-out test set. The sensor-derived physics-only comparator repeats its current-margin estimate across the forecast horizons and is deterministic, so its across-seed standard deviation is zero by construction. These are regression results only; thresholds were not recalibrated and event-warning metrics were not recomputed for each seed.

| Method | Horizon | MAE mean ± SD | RMSE mean ± SD | R² mean ± SD |
|---|---:|---:|---:|---:|
| Physics-only | 0.5 s | 0.29494 ± 0.00000 | 0.39180 ± 0.00000 | 0.95693 ± 0.00000 |
| Sensor-only XGBoost | 0.5 s | 0.23461 ± 0.00164 | 0.41850 ± 0.00904 | 0.95084 ± 0.00213 |
| Hybrid physics + residual | 0.5 s | 0.07532 ± 0.00042 | 0.12810 ± 0.00185 | 0.99540 ± 0.00013 |
| Physics-only | 1.0 s | 0.55813 ± 0.00000 | 0.75136 ± 0.00000 | 0.86784 ± 0.00000 |
| Sensor-only XGBoost | 1.0 s | 0.27200 ± 0.00144 | 0.46229 ± 0.00171 | 0.94997 ± 0.00037 |
| Hybrid physics + residual | 1.0 s | 0.14962 ± 0.00156 | 0.24548 ± 0.00411 | 0.98589 ± 0.00047 |
| Physics-only | 2.0 s | 0.98927 ± 0.00000 | 1.39343 ± 0.00000 | 0.66929 ± 0.00000 |
| Sensor-only XGBoost | 2.0 s | 0.34623 ± 0.00240 | 0.54425 ± 0.00597 | 0.94954 ± 0.00111 |
| Hybrid physics + residual | 2.0 s | 0.29044 ± 0.00409 | 0.46743 ± 0.00754 | 0.96278 ± 0.00119 |

In this five-seed regression experiment, the Hybrid had the lowest MAE and RMSE and highest R² of these three methods at each horizon, and its variation across these five seeds was small. This supports a limited statement about regression performance on this particular simulated split. It does not establish statistical superiority over all models, because LSTM/GRU were not retrained across seeds, event-warning thresholds were not recalibrated per seed, and the same held-out simulated runs were used for all comparisons. The physics-only comparator is sensor-derived and approximate; its deterministic standard deviation is not evidence of model robustness.

### F. Physics baselines, ablations and latency

The current Hybrid is a sensor-only approximate current-margin estimator plus an XGBoost residual predictor. It is not equivalent to an independent, validated β–r boundary classifier. A defensible physics-rule comparison requires a stated boundary selected independently of test outcomes and consistent event-level calibration. A load-transfer-ratio baseline is not directly supported by the current 2-DOF lateral simulator because it lacks roll/suspension dynamics. The completed event-warning ablation isolates the sensor-derived physics observer, sensor-only XGBoost, and Hybrid under validation-calibrated thresholds; the five-seed regression ablation covers XGBoost and Hybrid but does not retrain LSTM/GRU. No low-cost onboard hardware latency or memory measurement is currently verified.

### G. Limitations

This study uses synthetic data from a simplified lateral vehicle model. The tire parameters are representative rather than fitted to measured tires; sensor noise is illustrative rather than datasheet-calibrated; and the model omits dynamic load transfer, combined tire forces, and roll/suspension effects. The target is a rear-axle slip/saturation-margin proxy, not a validated complete stability envelope or rollover measure. The 44-run test partition represents held-out simulator runs, not independent real vehicles. No real-vehicle, hardware-in-the-loop, driver-response, or onboard compute validation has been performed. Short median warning lead times and unassociated warning episodes further limit claims of practical usefulness. The system should therefore be described as a research prototype for warning evaluation, not a deploy-ready safety product.

## VI. Conclusion

This paper evaluates multi-horizon prediction of a simulated rear-axle tire-saturation-margin proxy from four noisy retrofit-style sensor channels. Run-level splitting and validation-only threshold selection reduce leakage risk, while event-level evaluation distinguishes early warnings from late and unassociated episodes. In the latest 39-event evaluation snapshot, early detection ranges from 61.5% to 74.4%, with median lead times of 0.50–0.90 s and non-negligible unassociated warning counts. Hybrid at the 2.0 s horizon has the largest early-detection count in this snapshot, but it does not dominate every warning metric. These results do not establish sufficient driver-reaction time, transfer to real vehicles, or safety improvement. Future work must include multi-seed LSTM/GRU experiments, surface-specific event/class counts, exact sensor noise/quantization values and calibration, hardware latency tests, and validation with measured vehicle data. The completed matched event-warning ablation and five-seed regression experiment remain simulation-only; the five-seed retraining covers XGBoost and Hybrid, not LSTM/GRU.

## VII. References

[1] R. Rajamani, *Vehicle Dynamics and Control*, 2nd ed. New York, NY, USA: Springer, 2012, doi: 10.1007/978-1-4614-1433-9.

[2] H. B. Pacejka, *Tire and Vehicle Dynamics*, 3rd ed. Oxford, U.K.: Butterworth-Heinemann, 2012, doi: 10.1016/C2010-0-68548-8.

[3] S. Inagaki, I. Kushiro, and M. Yamamoto, “Analysis on vehicle stability in critical cornering using phase-plane method,” in *Proc. AVEC ’94*, 1994, pp. 287–292.

[4] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[5] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997, doi: 10.1162/neco.1997.9.8.1735.

[6] K. Cho et al., “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in *Proc. 2014 Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2014, pp. 1724–1734, doi: 10.3115/v1/D14-1179.

[7] L. Breiman, “Random forests,” *Machine Learning*, vol. 45, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.

[8] J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” *The Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001, doi: 10.1214/aos/1013203451.

[9] C. Bergmeir, R. J. Hyndman, and B. Koo, “A note on the validity of cross-validation for evaluating autoregressive time series prediction,” *Computational Statistics & Data Analysis*, vol. 120, pp. 70–83, 2018, doi: 10.1016/j.csda.2017.11.003.

[10] R. J. Hyndman and A. B. Koehler, “Another look at measures of forecast accuracy,” *International Journal of Forecasting*, vol. 22, no. 4, pp. 679–688, 2006, doi: 10.1016/j.ijforecast.2006.03.001.

[11] T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” *PLOS ONE*, vol. 10, no. 3, Art. no. e0118432, 2015, doi: 10.1371/journal.pone.0118432.

[12] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, 2015.

[13] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016.

[14] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[15] C. M. Bishop, *Pattern Recognition and Machine Learning*. New York, NY, USA: Springer, 2006.

[16] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993.


