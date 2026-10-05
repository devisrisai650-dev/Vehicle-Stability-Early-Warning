import numpy as np


def rear_slip_from_beta_r(
    beta,
    yaw_rate,
    speed,
    lr
):
    """
    Compute rear tire slip angle from beta-r phase-plane variables.

    beta:
        Vehicle sideslip angle [rad]

    yaw_rate:
        Yaw rate r [rad/s]

    speed:
        Longitudinal vehicle speed [m/s]

    lr:
        Distance from CG to rear axle [m]
    """

    vx = np.asarray(speed)
    beta = np.asarray(beta)
    yaw_rate = np.asarray(yaw_rate)

    vy = vx * np.tan(beta)

    alpha_r = np.arctan2(
        vy - lr * yaw_rate,
        vx
    )

    return alpha_r


def rear_saturation_boundary(
    beta,
    speed,
    lr,
    alpha_r_peak
):
    """
    Compute the yaw-rate boundaries corresponding to
    rear tire peak slip.

    The boundary satisfies:

        |alpha_r| = alpha_r_peak

    using the beta-r relationship.
    """

    vx = np.asarray(speed)
    beta = np.asarray(beta)

    vy = vx * np.tan(beta)

    alpha = alpha_r_peak

    # alpha_r = atan2(vy - lr*r, vx)
    #
    # Therefore:
    #
    # vy - lr*r = vx*tan(alpha_r)

    r_upper = (
        vy - vx * np.tan(-alpha)
    ) / lr

    r_lower = (
        vy - vx * np.tan(alpha)
    ) / lr

    return r_lower, r_upper