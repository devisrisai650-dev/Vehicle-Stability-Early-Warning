import pandas as pd

from evaluation.event_lead_time import match_warnings_to_events


def _event(event_id, onset, end, run_id=1):
    return {
        "Run_ID": run_id,
        "event_id": event_id,
        "onset_time": onset,
        "end_time": end,
        "duration_s": end - onset,
        "minimum_margin": -0.2,
        "max_abs_beta": 0.1,
        "max_abs_yaw_rate": 0.2,
    }


def _warning(start, end, run_id=1):
    return {
        "Run_ID": run_id,
        "warning_start": start,
        "warning_end": end,
        "duration_s": end - start,
        "minimum_prediction": -0.1,
    }


def test_warning_before_event_counts_as_early_detection():
    events = pd.DataFrame([_event(1, 5.0, 6.0)])
    warnings = pd.DataFrame([_warning(4.0, 4.2)])
    matched, false = match_warnings_to_events(events, warnings)
    assert bool(matched.loc[0, "early_warning_detected"])
    assert matched.loc[0, "lead_time_s"] == 1.0
    assert not bool(matched.loc[0, "late_warning_only"])
    assert false.empty


def test_warning_after_event_onset_is_not_early_detection():
    events = pd.DataFrame([_event(1, 5.0, 6.0)])
    warnings = pd.DataFrame([_warning(5.2, 5.4)])
    matched, false = match_warnings_to_events(events, warnings)
    assert not bool(matched.loc[0, "early_warning_detected"])
    assert bool(matched.loc[0, "late_warning_only"])
    assert pd.isna(matched.loc[0, "lead_time_s"])
    assert false.empty


def test_warning_outside_relevance_window_is_unassociated():
    events = pd.DataFrame([_event(1, 10.0, 11.0)])
    warnings = pd.DataFrame([_warning(1.0, 1.2)])
    matched, false = match_warnings_to_events(events, warnings)
    assert not bool(matched.loc[0, "early_warning_detected"])
    assert len(false) == 1
    assert false.iloc[0]["classification"] == "unassociated_warning"


def test_one_warning_cannot_detect_two_events():
    events = pd.DataFrame([
        _event(1, 5.0, 5.5),
        _event(2, 5.8, 6.2),
    ])
    warnings = pd.DataFrame([_warning(4.8, 5.0)])
    matched, false = match_warnings_to_events(events, warnings)
    assert int(matched["early_warning_detected"].sum()) == 1
    assert len(false) == 0
