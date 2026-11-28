# -*- coding: utf-8 -*-
"""The shared synthetic builders (tests/helpers/synthetic.py) do what their docstrings say,
and what they build is accepted by the real code it stands in for: the VA staircase, the
Sol IPC sample shape and the Sol quality accounting.
"""
import pytest

from helpers.synthetic import NO_EYE, NORMAL, make_sol_stream, parse_outcomes
from ntuh.sol import sol_ipc
from ntuh.tracking.sol_quality import SolQualityTracker, sol_sample_is_valid
from ntuh.vavf.stimuli import Staircase


# ---------- parse_outcomes ----------
def test_parse_outcomes_returns_cpd_and_passed_pairs():
    assert parse_outcomes("P2 P4 F6") == [(2.0, True), (4.0, True), (6.0, False)]


def test_parse_outcomes_accepts_any_whitespace_and_fractional_levels():
    assert parse_outcomes("  P2.5\nF12 ") == [(2.5, True), (12.0, False)]
    assert parse_outcomes("") == []


@pytest.mark.parametrize("seq", ["P2 X4", "P2 4", "P2 F", "P2 Fsix", "p2"])
def test_parse_outcomes_rejects_a_malformed_token(seq):
    with pytest.raises(ValueError):
        parse_outcomes(seq)


def test_a_parsed_sequence_replays_on_the_va_staircase():
    # The levels written in the sequence are the ones the VA staircase (start 2, step 2,
    # 2..20 cpd) really presents for these outcomes, so the sequence is self-consistent.
    stair = Staircase(start=2.0, step=2.0, minv=2.0, maxv=20.0)
    for cpd, passed in parse_outcomes("P2 P4 P6 F8 F6 P4 P6 P8 F10 P8"):
        assert stair.freq == cpd
        stair.update(passed)
    assert stair.reversals == [8.0, 4.0, 10.0, 8.0]


# ---------- make_sol_stream ----------
def test_stream_is_evenly_spaced_from_zero_and_excludes_the_end():
    stream = make_sol_stream(2.0, 120)
    times = [t for t, _ in stream]
    assert len(stream) == 240
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(2.0 - 1 / 120)
    assert [b - a for a, b in zip(times, times[1:])] == pytest.approx([1 / 120] * 239)


def test_samples_are_valid_with_both_eyes_normal_by_default():
    _, sample = make_sol_stream(0.1, 120)[0]
    assert sol_sample_is_valid(sample)
    assert sample.left_eye.eye_status == sample.right_eye.eye_status == NORMAL
    assert (sample.combined.gaze_2d.x, sample.combined.gaze_2d.y) == (640.0, 360.0)


def test_a_gap_removes_the_samples_and_nothing_else():
    stream = make_sol_stream(2.0, 120, gaps=[(0.5, 1.0)])
    times = [t for t, _ in stream]
    assert len(stream) == 240 - 60
    assert not [t for t in times if 0.5 <= t < 1.0]
    assert max(b - a for a, b in zip(times, times[1:])) == pytest.approx(0.5 + 1 / 120)
    assert all(sol_sample_is_valid(s) for _, s in stream)


def test_an_invalid_span_keeps_the_samples_but_marks_them_invalid():
    stream = make_sol_stream(2.0, 120, invalid=[(1.0, 1.5)], xy=(100.0, 200.0))
    assert len(stream) == 240
    for t, s in stream:
        bad = 1.0 <= t < 1.5
        assert sol_sample_is_valid(s) is not bad
        assert s.combined.gaze_2d.validity is not bad
        assert s.left_eye.eye_status == s.right_eye.eye_status == (NO_EYE if bad else NORMAL)
        assert s.left_eye.confidence == (0.0 if bad else 1.0)
        # Invalid samples still carry coordinates.
        assert (s.combined.gaze_2d.x, s.combined.gaze_2d.y) == (100.0, 200.0)


def test_gaps_and_invalid_spans_combine():
    stream = make_sol_stream(1.0, 100, gaps=[(0.2, 0.4)], invalid=[(0.3, 0.6), (0.9, 1.0)])
    assert len(stream) == 80
    assert sum(not sol_sample_is_valid(s) for _, s in stream) == 20 + 10


def test_xy_may_be_a_function_of_time():
    stream = make_sol_stream(0.5, 100, xy=lambda t: (100.0 + 1000.0 * t, 50.0))
    for t, s in stream:
        assert s.combined.gaze_2d.x == pytest.approx(100.0 + 1000.0 * t)
        assert s.combined.gaze_2d.y == 50.0


@pytest.mark.parametrize("invalid", [(), [(0.0, 1.0)]])
def test_sample_has_exactly_the_shape_the_sol_child_sends(invalid):
    # gaze_to_dict() is what the child runs on an SDK sample and reconstruct() what the parent
    # runs on the result. A builder sample must survive that round trip unchanged: a missing
    # field would come back filled with a default, an extra one would be dropped.
    _, sample = make_sol_stream(0.1, 120, invalid=invalid)[0]
    assert sol_ipc.reconstruct(sol_ipc.gaze_to_dict(sample)) == sample


def test_stream_feeds_the_sol_quality_tracker():
    quality = SolQualityTracker()
    for t, sample in make_sol_stream(2.0, 100, invalid=[(0.5, 1.0)]):
        quality.add_sample(sample, now=t)
    report = quality.validity_report(trial_only=False)
    assert report["sol_combined"] == (75.0, 200)
    assert report["sol_left"] == report["sol_right"] == (75.0, 200)
