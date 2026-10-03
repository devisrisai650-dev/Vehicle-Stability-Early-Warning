"""
Nonlinear 2-DOF bicycle model with a Pacejka "Magic Formula" tire model.
Used to generate vehicle lateral/yaw dynamics data that actually reaches the
tire-saturated, near-limit-of-control regime -- unlike a linear bicycle model,
which can never produce a genuine loss-of-control event.

States integrated: vy (lateral speed), r (yaw rate). vx (longitudinal speed)
is treated as slowly varying / approximately held per run (standard
convention for vehicle-dynamics handling tests).

References (for parameter/structure sanity, not exact calibration):
  - Pacejka, "Tire and Vehicle Dynamics" (Magic Formula tire model)
  - Rajamani, "Vehicle Dynamics and Control" (2-DOF bicycle model)
  - Inagaki et al. 1994 (beta-r phase plane stability boundary -- an
    alternative stability-margin definition to the rear-saturation one
    used here; worth implementing as a cross-check later)
"""
import numpy as np

G = 9.81

VEHICLE = dict(
    m=1500.0,      # kg
    Iz=2500.0,     # kg*m^2 yaw inertia
    lf=1.2,        # m, CG to front axle
    lr=1.6,        # m, CG to rear axle
)

# Simplified Magic Formula shape parameters (front/rear), typical
# passenger-tire-like values -- NOT fit to a specific real tire.
TIRE_F = dict(B=9.0, C=1.5, E=-1.0)
TIRE_R = dict(B=9.5, C=1.5, E=-1.0)


def _peak_slip_angle(tire, grid=np.linspace(0, 0.5, 5001)):
    y = np.sin(tire['C'] * np.arctan(tire['B'] * grid - tire['E'] * (tire['B'] * grid - np.arctan(tire['B'] * grid))))
    return grid[np.argmax(y)]


ALPHA_R_PEAK = _peak_slip_angle(TIRE_R)   # ~0.140 rad (~8.0 deg) -- independent of mu, Fz


def magic_formula_Fy(alpha, mu, Fz, tire):
    """SAE-style convention: positive slip angle -> restoring (negative) force."""
    D = mu * Fz
    B, C, E = tire['B'], tire['C'], tire['E']
    return -D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))


def static_normal_loads(veh=VEHICLE):
    m, lf, lr = veh['m'], veh['lf'], veh['lr']
    L = lf + lr
    Fzf = m * G * lr / L
    Fzr = m * G * lf / L
    return Fzf, Fzr


def dynamics(state, delta, mu, veh=VEHICLE):
    """state = [vy, r], vx passed separately (quasi-constant per step)."""
    vy, r, vx = state
    m, Iz, lf, lr = veh['m'], veh['Iz'], veh['lf'], veh['lr']
    Fzf, Fzr = static_normal_loads(veh)

    vx_safe = max(vx, 1.0)
    alpha_f = np.arctan2(vy + lf * r, vx_safe) - delta
    alpha_r = np.arctan2(vy - lr * r, vx_safe)

    Fyf = magic_formula_Fy(alpha_f, mu, Fzf, TIRE_F)
    Fyr = magic_formula_Fy(alpha_r, mu, Fzr, TIRE_R)

    vy_dot = (Fyf * np.cos(delta) + Fyr) / m - vx_safe * r
    r_dot = (lf * Fyf * np.cos(delta) - lr * Fyr) / Iz
    return np.array([vy_dot, r_dot]), alpha_f, alpha_r, Fyf, Fyr


def rk4_step(state2, delta, vx, mu, dt, veh=VEHICLE):
    def f(s):
        d, *_ = dynamics(np.array([s[0], s[1], vx]), delta, mu, veh)
        return d
    k1 = f(state2)
    k2 = f(state2 + 0.5 * dt * k1)
    k3 = f(state2 + 0.5 * dt * k2)
    k4 = f(state2 + dt * k3)
    return state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
