# -*- coding: utf-8 -*-
"""Stop rules of the VA staircase (ntuh.vavf.stimuli.Staircase.done).

Uses the parameters the VA test builds its staircase with (ntuh/flows/va_test.py:
start 2, step 2, min 2, max 20 cpd). Sequences are synthetic PASS/FAIL runs, not field data.

Not tested here: which cpd the VA test reports as the result and how it maps to a VA score.
That is ntuh.vavf.va_scoring (REQ-0025, REQ-0026), tested in tests/test_va_scoring.py. Only the
stop rules are pinned:
  - 4 reversals, or
  - at the ceiling, once the run has had >= 3 correct in a row at any point
    (max_correct_streak, not necessarily the last 3 trials), or
  - 4 incorrect in a row.
"""
import pytest

from ntuh.vavf.stimuli import Staircase

P, F = True, False


def _va_staircase():
    return Staircase(start=2.0, step=2.0, minv=2.0, maxv=20.0)


def _run(responses):
    """Feed responses one by one; return the staircase and done() after every step."""
    stair = _va_staircase()
    done_after = []
    for correct in responses:
        stair.update(correct)
        done_after.append(stair.done())
    return stair, done_after


def test_fresh_staircase_is_not_done():
    assert _va_staircase().done() is False


def test_stops_on_the_fourth_reversal():
    stair, done_after = _run([P, F, P, F, P])
    assert len(stair.reversals) == 4
    assert done_after == [False, False, False, False, True]


def test_reversals_record_the_level_where_direction_changed():
    stair, _ = _run([P, P, F, P, F])        # 2->4->6, F at 6 ->4, P at 4 ->6, F at 6
    assert stair.reversals == [6.0, 4.0, 6.0]


def test_stops_at_the_ceiling_after_three_consecutive_passes():
    # 2 -> 20 cpd takes 9 passes; the rule fires the moment the ceiling is reached.
    stair, done_after = _run([P] * 9)
    assert stair.freq == 20.0
    assert done_after == [False] * 8 + [True]


def test_stops_after_four_consecutive_fails():
    stair, done_after = _run([F, F, F, F])
    assert stair.incorrect_streak == 4
    assert done_after == [False, False, False, True]


def test_a_pass_resets_the_fail_streak():
    # 6 fails in total, never 4 in a row, and only 2 reversals: must keep going.
    stair, done_after = _run([F, F, F, P, F, F, F])
    assert stair.incorrect_streak == 3
    assert not any(done_after)


@pytest.mark.parametrize("responses", [[F] * 6, [P] * 12, [P, F] * 3])
def test_level_stays_within_floor_and_ceiling(responses):
    stair = _va_staircase()
    for correct in responses:
        stair.update(correct)
        assert stair.minv <= stair.freq <= stair.maxv
