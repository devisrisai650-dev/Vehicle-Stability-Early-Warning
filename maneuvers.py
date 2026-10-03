import numpy as np


def step_steer(t, amplitude, ramp_time=0.3):
    return amplitude * np.clip(t / ramp_time, 0, 1)


def sine_sweep(t, amplitude, freq_hz=0.3):
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def double_lane_change(t, amplitude):
    # breakpoints (s): 0-1 hold 0 | 1-1.5 ramp +A | 1.5-2.5 hold +A |
    # 2.5-3.5 ramp -A | 3.5-4.5 hold -A | 4.5-5.0 ramp 0 | 5.0+ hold 0
    d = np.zeros_like(t)
    d = np.where((t >= 1.0) & (t < 1.5), amplitude * (t - 1.0) / 0.5, d)
    d = np.where((t >= 1.5) & (t < 2.5), amplitude, d)
    d = np.where((t >= 2.5) & (t < 3.5), amplitude - 2 * amplitude * (t - 2.5) / 1.0, d)
    d = np.where((t >= 3.5) & (t < 4.5), -amplitude, d)
    d = np.where((t >= 4.5) & (t < 5.0), -amplitude + amplitude * (t - 4.5) / 0.5, d)
    d = np.where(t >= 5.0, 0.0, d)
    return d


def ramp_steer(t, amplitude, ramp_end=8.0):
    return amplitude * np.clip(t / ramp_end, 0, 1)


PROFILES = {
    'step_steer': lambda t, a: step_steer(t, a),
    'sine_sweep': lambda t, a: sine_sweep(t, a),
    'double_lane_change': lambda t, a: double_lane_change(t, a),
    'ramp_steer': lambda t, a: ramp_steer(t, a),
}

DURATIONS = {
    'step_steer': 8.0,
    'sine_sweep': 8.0,
    'double_lane_change': 7.0,
    'ramp_steer': 10.0,
}
