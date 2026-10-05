"""
Nonlinear 2-DOF bicycle model with a Pacejka "Magic Formula" tire model.

Used to generate vehicle lateral/yaw dynamics data that reaches the
tire-saturated / near-limit regime.

States integrated:
    vy : lateral velocity [m/s]
    r  : yaw rate [rad/s]

Longitudinal speed vx is passed separately and treated as approximately
constant for each run.

References for model structure/sanity:
    - Pacejka, Tire and Vehicle Dynamics
    - Rajamani, Vehicle Dynamics and Control
    - Inagaki et al. 1994, beta-r phase-plane stability boundary
"""

import numpy as np


# ==============================================================
# Constants
# ==============================================================

G = 9.81  # gravitational acceleration [m/s^2]


# ==============================================================
# Vehicle parameters
# ==============================================================

VEHICLE = dict(
    m=1500.0,      # vehicle mass [kg]
    Iz=2500.0,     # yaw moment of inertia [kg m^2]
    lf=1.2,        # CG -> front axle [m]
    lr=1.6,        # CG -> rear axle [m]
)


# ==============================================================
# Tire parameters
# ==============================================================

# Representative passenger-car-like tire shape parameters.
# These are NOT fitted to a specific real tire.

TIRE_F = dict(
    B=9.0,
    C=1.5,
    E=-1.0,
)

TIRE_R = dict(
    B=9.5,
    C=1.5,
    E=-1.0,
)


# ==============================================================
# Peak slip-angle calculation
# ==============================================================

def _peak_slip_angle(
    tire,
    grid=np.linspace(0.0, 0.5, 5001)
):
    """
    Find the positive slip angle at which the normalized
    Pacejka lateral-force shape reaches its maximum.

    Returns:
        Peak slip angle [rad]
    """

    y = np.sin(
        tire["C"]
        * np.arctan(
            tire["B"] * grid
            - tire["E"]
            * (
                tire["B"] * grid
                - np.arctan(tire["B"] * grid)
            )
        )
    )

    return grid[np.argmax(y)]


# Exact peak rear slip angle used by the simulator.
ALPHA_R_PEAK = _peak_slip_angle(TIRE_R)


# ==============================================================
# Pacejka Magic Formula
# ==============================================================

def magic_formula_Fy(
    alpha,
    mu,
    Fz,
    tire
):
    """
    Calculate lateral tire force.

    Sign convention:
        Positive slip angle -> restoring negative lateral force.

    Parameters
    ----------
    alpha : float or np.ndarray
        Tire slip angle [rad]

    mu : float
        Road friction coefficient

    Fz : float
        Normal load [N]

    tire : dict
        Tire parameters B, C, E

    Returns
    -------
    Fy : float or np.ndarray
        Lateral tire force [N]
    """

    # Peak lateral force
    D = mu * Fz

    B = tire["B"]
    C = tire["C"]
    E = tire["E"]

    Fy = -D * np.sin(
        C
        * np.arctan(
            B * alpha
            - E
            * (
                B * alpha
                - np.arctan(B * alpha)
            )
        )
    )

    return Fy


# ==============================================================
# Static axle loads
# ==============================================================

def static_normal_loads(
    veh=VEHICLE
):
    """
    Calculate static front and rear axle normal loads.

    Returns
    -------
    Fzf : float
        Front axle normal load [N]

    Fzr : float
        Rear axle normal load [N]
    """

    m = veh["m"]
    lf = veh["lf"]
    lr = veh["lr"]

    L = lf + lr

    Fzf = m * G * lr / L
    Fzr = m * G * lf / L

    return Fzf, Fzr


# ==============================================================
# Vehicle dynamics
# ==============================================================

def dynamics(
    state,
    delta,
    mu,
    veh=VEHICLE
):
    """
    Nonlinear 2-DOF bicycle-model dynamics.

    Parameters
    ----------
    state : array-like
        [vy, r]

    delta : float
        Front road-wheel steering angle [rad]

    mu : float
        Road friction coefficient

    veh : dict
        Vehicle parameters

    Returns
    -------
    derivative : np.ndarray
        [vy_dot, r_dot]

    alpha_f : float
        Front tire slip angle [rad]

    alpha_r : float
        Rear tire slip angle [rad]

    Fyf : float
        Front lateral force [N]

    Fyr : float
        Rear lateral force [N]
    """

    vy, r, vx = state

    m = veh["m"]
    Iz = veh["Iz"]
    lf = veh["lf"]
    lr = veh["lr"]

    # Static axle loads
    Fzf, Fzr = static_normal_loads(veh)

    # Protect against unrealistically small / zero longitudinal speed
    vx_safe = max(vx, 1.0)

    # ----------------------------------------------------------
    # Tire slip angles
    # ----------------------------------------------------------

    alpha_f = (
        np.arctan2(
            vy + lf * r,
            vx_safe
        )
        - delta
    )

    alpha_r = np.arctan2(
        vy - lr * r,
        vx_safe
    )

    # ----------------------------------------------------------
    # Tire lateral forces
    # ----------------------------------------------------------

    Fyf = magic_formula_Fy(
        alpha_f,
        mu,
        Fzf,
        TIRE_F
    )

    Fyr = magic_formula_Fy(
        alpha_r,
        mu,
        Fzr,
        TIRE_R
    )

    # ----------------------------------------------------------
    # Equations of motion
    # ----------------------------------------------------------

    vy_dot = (
        Fyf * np.cos(delta)
        + Fyr
    ) / m - vx_safe * r

    r_dot = (
        lf * Fyf * np.cos(delta)
        - lr * Fyr
    ) / Iz

    derivative = np.array([
        vy_dot,
        r_dot,
    ])

    return (
        derivative,
        alpha_f,
        alpha_r,
        Fyf,
        Fyr,
    )


# ==============================================================
# RK4 integration
# ==============================================================

def rk4_step(
    state2,
    delta,
    vx,
    mu,
    dt,
    veh=VEHICLE
):
    """
    Perform one fourth-order Runge-Kutta integration step.

    Parameters
    ----------
    state2 : np.ndarray
        Current [vy, r]

    delta : float
        Steering angle [rad]

    vx : float
        Longitudinal speed [m/s]

    mu : float
        Road friction coefficient

    dt : float
        Integration time step [s]

    veh : dict
        Vehicle parameters

    Returns
    -------
    np.ndarray
        Updated [vy, r]
    """

    def f(s):

        derivative, _, _, _, _ = dynamics(
            np.array([
                s[0],
                s[1],
                vx
            ]),
            delta,
            mu,
            veh
        )

        return derivative

    # RK4 stages
    k1 = f(state2)

    k2 = f(
        state2
        + 0.5 * dt * k1
    )

    k3 = f(
        state2
        + 0.5 * dt * k2
    )

    k4 = f(
        state2
        + dt * k3
    )

    # Final RK4 update
    next_state = (
        state2
        + (dt / 6.0)
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )

    return next_state