# -*- coding: utf-8 -*-
"""The VA pass rule (ntuh/vavf/hold_tracker.py): only valid, fresh gaze on the target's side
counts toward the hold. Synthetic samples; the replay of the 2026-09-18 field data at the end
runs only when NTUH_VA_0918_ROOT points at it.
"""
import os
from types import SimpleNamespace

import pytest

from ntuh.tracking.sol_quality import sol_sample_usable
from ntuh.vavf.hold_tracker import (
    PASS_RULE_VERSION, TRIAL_QUALITY_COLUMNS, GazeSample, HoldTracker, trial_quality_row)

HZ = 120            # the Sol gaze rate; one update per sample keeps the times exact
PASS_DUR = 2.0


def run(tracker, sample_at, until):
    """Update the tracker every 1/120 s from 0 to `until` with sample_at(t). Returns the time
    of the PASS, or None."""
    for k in range(int(round(until * HZ)) + 1):
        t = k / HZ
        if tracker.update(t, sample_at(t)):
            return t
    return None


def on_target(t, valid=True):
    return GazeSample(t, valid, True)


def sdk_sample(valid=True, left=2, right=2):
    """What the Sol SDK delivers, reduced to what the validity check reads (2 = NORMAL eye)."""
    eye = lambda status: SimpleNamespace(eye_status=SimpleNamespace(value=status))
    return SimpleNamespace(combined=SimpleNamespace(gaze_3d=SimpleNamespace(validity=valid)),
                           left_eye=eye(left), right_eye=eye(right))


# ---------- the five cases of the audit ----------
def test_two_seconds_of_valid_gaze_on_the_target_side_pass_at_two_seconds():
    assert run(HoldTracker(PASS_DUR), on_target, until=3.0) == 2.0


def test_invalid_samples_on_the_target_side_never_pass():
    """Validity 0 with both eyes NO_EYE and the gaze point frozen on the target's side: the
    0%-valid catch trial of 2026-09-18 that passed."""
    frozen = sdk_sample(valid=False, left=0, right=0)
    tracker = HoldTracker(PASS_DUR)
    assert run(tracker, lambda t: GazeSample(t, sol_sample_usable(frozen), True), until=6.0) is None
    assert not tracker.in_progress and tracker.held_s == 0.0


def test_one_sample_repeated_with_no_new_ones_never_passes():
    tracker = HoldTracker(PASS_DUR)
    only = GazeSample(0.0, True, True)                      # received once, then the stream stops
    assert run(tracker, lambda t: only, until=6.0) is None
    assert not tracker.in_progress                          # 0.2 s counted, then it started over


def test_a_stream_that_freezes_mid_hold_does_not_carry_the_hold_to_a_pass():
    tracker = HoldTracker(PASS_DUR)
    assert run(tracker, lambda t: on_target(min(t, 1.0)), until=6.0) is None
    assert not tracker.in_progress


def test_a_200_ms_blink_delays_the_pass_by_200_ms():
    blink = lambda t: on_target(t, valid=not (1.0 <= t < 1.2))
    assert run(HoldTracker(PASS_DUR), blink, until=4.0) == pytest.approx(2.2, abs=2 / HZ)


def test_a_500_ms_loss_starts_the_hold_over():
    lost = lambda t: on_target(t, valid=not (1.0 <= t < 1.5))
    tracker = HoldTracker(PASS_DUR)
    assert run(tracker, lost, until=1.25) is None
    assert tracker.in_progress and tracker.held_s == pytest.approx(1.0, abs=2 / HZ)    # 0.25 s in: paused
    tracker = HoldTracker(PASS_DUR)
    assert run(tracker, lost, until=1.4) is None
    assert not tracker.in_progress and tracker.held_s == 0.0                           # 0.4 s in: started over
    assert run(HoldTracker(PASS_DUR), lost, until=5.0) == pytest.approx(3.5, abs=2 / HZ)


# ---------- the rest of the rule ----------
def test_valid_gaze_on_the_other_side_starts_the_hold_over_at_once():
    tracker = HoldTracker(PASS_DUR)
    away = lambda t: GazeSample(t, True, not (1.0 <= t < 1.0 + 1 / HZ))     # one sample away
    assert run(tracker, away, until=1.0) is None
    assert not tracker.in_progress
    assert run(HoldTracker(PASS_DUR), away, until=5.0) == pytest.approx(3.0, abs=2 / HZ)


def test_a_sample_counts_only_while_it_is_fresh():
    tracker = HoldTracker(PASS_DUR, max_sample_age_s=0.2, blink_grace_s=0.3)
    sample = GazeSample(10.0, True, True)
    assert not tracker.update(10.0, sample) and not tracker.update(10.19, sample)
    assert tracker.held_s == pytest.approx(0.19)            # still fresh: it counts
    tracker.update(10.25, sample)
    assert tracker.in_progress and tracker.held_s == pytest.approx(0.19)    # too old: paused
    tracker.update(10.31, sample)
    assert not tracker.in_progress                          # 0.3 s since it was received: over


def test_no_sample_at_all_is_no_gaze():
    tracker = HoldTracker(PASS_DUR)
    assert run(tracker, lambda t: None, until=3.0) is None and not tracker.in_progress


def test_the_thresholds_are_parameters():
    strict = HoldTracker(PASS_DUR, max_sample_age_s=0.05, blink_grace_s=0.1)
    blink = lambda t: on_target(t, valid=not (1.0 <= t < 1.2))
    assert run(strict, blink, until=5.0) == pytest.approx(3.2, abs=2 / HZ)   # the blink is a loss now


def test_hold_start_is_when_the_passing_hold_began():
    tracker = HoldTracker(PASS_DUR)
    late = lambda t: GazeSample(t, True, t >= 0.5)          # looks at the target from 0.5 s on
    assert run(tracker, late, until=4.0) == pytest.approx(2.5, abs=2 / HZ)
    assert tracker.hold_start == pytest.approx(0.5, abs=1 / HZ) and tracker.passed


# ---------- which Sol samples are evidence ----------
@pytest.mark.parametrize("sample, usable", [
    (sdk_sample(), True),
    (sdk_sample(left=0), True),                             # one eye is enough
    (sdk_sample(left=1, right=1), True),                    # blinking but valid: the SDK decides
    (sdk_sample(left=0, right=0), False),                   # no eye at all, whatever validity says
    (sdk_sample(valid=False), False),
    (SimpleNamespace(combined=SimpleNamespace(gaze_3d=SimpleNamespace(validity=True))), True),
    (None, False),
])
def test_a_sol_sample_is_evidence_when_valid_and_an_eye_is_seen(sample, usable):
    assert sol_sample_usable(sample) is usable


# ---------- trial_quality.csv ----------
def test_trial_quality_columns_are_the_agreed_ones():
    assert TRIAL_QUALITY_COLUMNS == ["trial_number", "n_sol_samples", "valid_fraction_trial",
                                     "valid_fraction_final_window", "hold_start_s",
                                     "max_sample_age_ms", "n_gaps_gt_300ms", "pass_rule_version"]


def test_trial_quality_row_of_a_trial_with_a_blink_and_a_dropout():
    """A 3 s trial at 100 Hz: invalid from 1.0 to 1.5 s, no samples from 2.0 to 2.4 s."""
    start = 100.0
    times = [k / 100 for k in range(300) if not 2.0 <= k / 100 < 2.4]
    samples = [(start + t, not 1.0 <= t < 1.5) for t in times]
    row = trial_quality_row(7, samples, start, start + 3.0, final_window_s=2.0, hold_start=start + 2.4)
    assert list(row) == TRIAL_QUALITY_COLUMNS
    assert (row["trial_number"], row["n_sol_samples"]) == (7, 260)
    assert row["valid_fraction_trial"] == round(210 / 260, 4)
    assert row["valid_fraction_final_window"] == round(110 / 160, 4)    # of the samples from 1.0 s on
    assert row["hold_start_s"] == 2.4
    assert row["max_sample_age_ms"] == 410                              # the dropout
    assert row["n_gaps_gt_300ms"] == 2                                  # the blink and the dropout
    assert row["pass_rule_version"] == PASS_RULE_VERSION == "2"


def test_trial_quality_row_without_any_sample():
    row = trial_quality_row(1, [], 50.0, 55.0, final_window_s=2.0)
    assert (row["n_sol_samples"], row["valid_fraction_trial"], row["valid_fraction_final_window"]) == (0, "", "")
    assert (row["hold_start_s"], row["max_sample_age_ms"], row["n_gaps_gt_300ms"]) == ("", 5000, 1)


# ---------- the 2026-09-18 field data (opt-in, never in CI) ----------
@pytest.mark.skipif(not os.environ.get("NTUH_VA_0918_ROOT"),
                    reason="set NTUH_VA_0918_ROOT to the folder of the 2026-09-18 VA sessions")
def test_the_0918_catch_trial_without_valid_gaze_no_longer_passes():
    from tools import replay_va_hold
    results = replay_va_hold.replay_root(os.environ["NTUH_VA_0918_ROOT"])
    outcome = {(r["session"], r["trial"]): (r["trial_type"], r["was"], r["now"]) for r in results}
    assert outcome[("s02_c5", 5)] == ("catch", "PASS", "FAIL")
