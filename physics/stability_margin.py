import numpy as np


def compute_stability_margin(alpha_r, alpha_r_peak):
    """
    Compute normalized rear-tire stability margin.

    M = (alpha_r_peak - |alpha_r|) / alpha_r_peak

    Positive:
        Rear tire is below peak slip.

    Zero:
        Rear tire is at peak slip.

    Negative:
        Rear tire has exceeded peak slip.
    """

    alpha_r = np.asarray(alpha_r)

    margin = (
        alpha_r_peak - np.abs(alpha_r)
    ) / alpha_r_peak

    return margin


def compute_instability_flag(
    stability_margin,
    consecutive_samples=3
):
    """
    Reproduce the instability-flag logic used by the dataset generator.

    A contiguous negative-margin segment is considered an instability
    event when it contains at least `consecutive_samples` samples.

    Once such a segment reaches the required length, the complete
    negative segment is marked True, including the first samples.
    """

    margin = np.asarray(stability_margin)

    negative = margin < 0

    flag = np.zeros(
        len(margin),
        dtype=bool
    )

    run_len = 0

    for i, is_negative in enumerate(negative):

        if is_negative:
            run_len += 1
        else:
            run_len = 0

        if run_len >= consecutive_samples:

            start = max(
                0,
                i - run_len + 1
            )

            flag[start:i + 1] = True

    return flag