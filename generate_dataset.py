"""
Generates the full near-limit vehicle stability dataset:

  1. ground_truth_dynamics.csv  -- clean simulated states + derived labels
     (beta, stability_margin, instability_flag). This is the "answer key" --
     never feed it to a model as input.
  2. retrofit_sensor_stream.csv -- what a low-cost retrofit sensor kit would
     actually measure: noisy/quantized yaw rate, lateral accel, speed,
     steering angle, at a lower sample rate. No beta, no margin, no flag.
     This is the actual model INPUT.
  3. run_metadata.csv -- one row per run, for run-level (not row-level)
     train/test splitting.

Join key: Run_ID + Time (both files share the same downsampled time grid).
"""
import numpy as np
import pandas as pd

from vehicle_model import rk4_step, dynamics, ALPHA_R_PEAK, VEHICLE
from maneuvers import PROFILES, DURATIONS

RNG = np.random.default_rng(42)

DT_SIM = 0.01           # 100 Hz internal integration
OUT_HZ = 20              # 20 Hz output grid (both files)
DECIMATE = int(round((1.0 / DT_SIM) / OUT_HZ))

MU_LEVELS = {'dry': 0.9, 'wet': 0.5, 'ice_snow': 0.2}
SPEED_LEVELS_KMH = [50, 70, 90, 110]
AMPLITUDE_DEG = [1, 2, 3, 5, 7, 10]
MANEUVERS = list(PROFILES.keys())

MIN_SUSTAIN_SAMPLES = 3  # at 20Hz, ~0.15s sustained past-peak to count as an instability event


def simulate_run(maneuver, mu_name, mu, speed_kmh, amp_deg, run_id):
    vx = speed_kmh / 3.6
    amplitude = np.radians(amp_deg)
    duration = DURATIONS[maneuver]
    n_steps = int(duration / DT_SIM)
    t_fine = np.arange(n_steps) * DT_SIM
    delta_fine = PROFILES[maneuver](t_fine, amplitude)

    state = np.array([0.0, 0.0])  # vy, r
    rows = []
    for i in range(n_steps):
        d, af, ar, Fyf, Fyr = dynamics(np.array([state[0], state[1], vx]), delta_fine[i], mu)
        ay = d[0] + vx * state[1]  # body-frame lateral accel measured at CG
        beta = np.arctan2(state[0], vx)
        margin = (ALPHA_R_PEAK - abs(ar)) / ALPHA_R_PEAK
        rows.append((t_fine[i], state[0], state[1], beta, delta_fine[i], ay, af, ar, margin))
        state = rk4_step(state, delta_fine[i], vx, mu, DT_SIM)
        if not np.isfinite(state).all() or abs(state[0]) > 200:
            break  # numerical safety net only

    df = pd.DataFrame(rows, columns=['Time', 'vy', 'yaw_rate', 'beta', 'steer_true',
                                      'lateral_accel', 'alpha_f', 'alpha_r', 'stability_margin'])
    df = df.iloc[::DECIMATE].reset_index(drop=True)  # decimate to OUT_HZ

    # sustained instability flag (avoid single-sample noise triggers)
    past_peak = (df['stability_margin'] < 0).astype(int).values
    sustained = np.zeros(len(df), dtype=bool)
    run_len = 0
    for i, p in enumerate(past_peak):
        run_len = run_len + 1 if p else 0
        if run_len >= MIN_SUSTAIN_SAMPLES:
            sustained[max(0, i - run_len + 1):i + 1] = True
    df['instability_flag'] = sustained

    df.insert(0, 'Run_ID', run_id)
    df['maneuver'] = maneuver
    df['mu_surface'] = mu_name
    df['mu_true'] = mu
    df['target_speed_kmh'] = speed_kmh
    df['amplitude_deg'] = amp_deg
    return df


def add_sensor_noise(gt, rng):
    """Low-cost retrofit-grade sensor model applied to ground truth.
    Noise levels are illustrative (MEMS-consumer-grade order of magnitude),
    NOT calibrated against a specific datasheet -- documented as such."""
    n = len(gt)
    yaw_rate_meas = gt['yaw_rate'].values + rng.normal(0, np.radians(0.5), n) \
        + rng.normal(0, np.radians(0.1))  # per-run constant bias term
    lat_accel_meas = gt['lateral_accel'].values + rng.normal(0, 0.15, n) \
        + rng.normal(0, 0.05)
    speed_ms = gt['target_speed_kmh'].values / 3.6
    speed_meas = speed_ms + rng.normal(0, 0.1, n)
    speed_meas = np.round(speed_meas / 0.28) * 0.28  # ~1 km/h quantization, typical cheap wheel-speed resolution
    steer_meas = gt['steer_true'].values + rng.normal(0, np.radians(0.2), n)

    out = pd.DataFrame({
        'Run_ID': gt['Run_ID'].values,
        'Time': gt['Time'].values,
        'yaw_rate_meas': yaw_rate_meas,
        'lateral_accel_meas': lat_accel_meas,
        'speed_meas': speed_meas,
        'steering_angle_meas': steer_meas,
        'maneuver': gt['maneuver'].values,
        'mu_surface': gt['mu_surface'].values,
    })
    return out


def main():
    all_gt, all_meta, run_id = [], [], 0
    for maneuver in MANEUVERS:
        for mu_name, mu in MU_LEVELS.items():
            for speed_kmh in SPEED_LEVELS_KMH:
                for amp_deg in AMPLITUDE_DEG:
                    run_id += 1
                    df = simulate_run(maneuver, mu_name, mu, speed_kmh, amp_deg, run_id)
                    all_gt.append(df)
                    all_meta.append(dict(
                        Run_ID=run_id, maneuver=maneuver, mu_surface=mu_name, mu_true=mu,
                        target_speed_kmh=speed_kmh, amplitude_deg=amp_deg,
                        duration_s=df['Time'].iloc[-1] if len(df) else 0.0,
                        n_samples=len(df),
                        reaches_instability=bool(df['instability_flag'].any()),
                        min_stability_margin=float(df['stability_margin'].min()) if len(df) else np.nan,
                    ))

    gt = pd.concat(all_gt, ignore_index=True)
    meta = pd.DataFrame(all_meta)

    noise_rng = np.random.default_rng(7)
    sensor = add_sensor_noise(gt, noise_rng)

    print(f"Total runs: {meta.shape[0]}")
    print(f"Runs reaching instability: {meta['reaches_instability'].sum()} "
          f"({100 * meta['reaches_instability'].mean():.1f}%)")
    print(f"Total rows (ground truth): {len(gt)}")
    print(meta.groupby(['mu_surface'])['reaches_instability'].mean())

    gt_out = gt[['Run_ID', 'Time', 'maneuver', 'mu_surface', 'mu_true', 'target_speed_kmh',
                 'amplitude_deg', 'steer_true', 'yaw_rate', 'beta', 'lateral_accel',
                 'alpha_f', 'alpha_r', 'stability_margin', 'instability_flag']]

    gt_out.to_csv('/home/claude/dataset_gen/ground_truth_dynamics.csv', index=False)
    sensor.to_csv('/home/claude/dataset_gen/retrofit_sensor_stream.csv', index=False)
    meta.to_csv('/home/claude/dataset_gen/run_metadata.csv', index=False)
    print("\nSaved: ground_truth_dynamics.csv, retrofit_sensor_stream.csv, run_metadata.csv")


if __name__ == '__main__':
    main()
