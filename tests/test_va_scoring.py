# -*- coding: utf-8 -*-
"""VA scoring (ntuh/vavf/va_scoring.py): the threshold a staircase run actually reached.

Sequences are written as "P2 F4": PASS at 2 cpd, FAIL at 4 cpd, on the VA test's staircase
(start 2, step 2, 2-20 cpd). They are synthetic outcome patterns, named by pattern.
The replay of the 2026-09-18 field runs is opt-in: set NTUH_VA_0918_ROOT to that VA dataset
folder. It reads trial_events.csv only and is skipped when the variable is unset.
"""
import csv
import json
import os
import random
import re
from pathlib import Path

import pytest

from ntuh.vavf import va_scoring as vs
from ntuh.vavf.stimuli import Staircase


def seq(text):
    """"P2 F4" -> [(2.0, True), (4.0, False)]"""
    return [(float(token[1:]), token[0] == "P") for token in text.split()]


def replay(history):
    """The real staircase after the same outcomes, checking each level is the one it would show."""
    stair = Staircase(**vs.STAIRCASE)
    for cpd, passed in history:
        assert cpd == stair.freq, f"{cpd} cpd is not on the staircase path (expected {stair.freq})"
        stair.update(passed)
    return stair


def stop_reason_of(stair):
    """Staircase.done(), spelled out in its own order."""
    if len(stair.reversals) >= 4:
        return "four_reversals"
    if stair.freq >= stair.maxv and stair.max_correct_streak >= 3:
        return "max_streak"
    if stair.incorrect_streak >= 4:
        return "four_consecutive_fails"
    return None


# name, sequence, legacy next level, highest passed, reversals, last-4 mean, stop reason, floor
PATTERNS = [
    ("ends_on_a_pass_one_step_below_the_legacy_value", "P2 P4 P6 F8 F6 P4 P6 P8 F10 P8",
     10.0, 8.0, [8.0, 4.0, 10.0, 8.0], 7.5, "four_reversals", False),
    ("oscillates_between_two_levels", "P2 P4 P6 P8 P10 P12 F14 P12 F14 P12",
     14.0, 12.0, [14.0, 12.0, 14.0, 12.0], 13.0, "four_reversals", False),
    ("legacy_value_happens_to_be_a_passed_level", "P2 P4 P6 P8 P10 F12 P10 P12 P14 P16 F18 F16 P14",
     16.0, 16.0, [12.0, 10.0, 18.0, 14.0], 13.5, "four_reversals", False),
    ("four_fails_in_a_row_at_the_floor", "F2 P2 F4 F2 F2 F2",
     2.0, 2.0, [2.0, 4.0], None, "four_consecutive_fails", True),
    ("four_reversals_near_the_floor", "F2 P2 F4 P2 F4",
     2.0, 2.0, [2.0, 4.0, 2.0, 4.0], 3.0, "four_reversals", False),
    ("passes_all_the_way_up", "P2 P4 P6 P8 P10 P12 P14 P16 P18",
     20.0, 18.0, [], None, "max_streak", False),
    ("four_fails_in_a_row_above_the_floor", "P2 P4 P6 P8 P10 F12 F10 F8 F6",
     4.0, 10.0, [12.0], None, "four_consecutive_fails", False),
    ("stopped_by_the_operator", "P2 P4 F6",
     4.0, 4.0, [6.0], None, None, False),
]


@pytest.mark.parametrize("name, text, legacy, highest, reversals, mean, stop, floor", PATTERNS,
                         ids=[p[0] for p in PATTERNS])
def test_candidate_thresholds_of_a_run(name, text, legacy, highest, reversals, mean, stop, floor):
    history = seq(text)
    assert vs.score_staircase(history, **vs.STAIRCASE) == {
        "legacy_next_level_cpd": legacy, "highest_passed_cpd": highest,
        "last4_reversal_mean_cpd": mean, "reversals": reversals, "stop_reason": stop,
        "floor": floor, "n_normal_trials": len(history)}


def _random_run(rng, p_pass):
    """Outcomes drawn at random, levels taken from the real staircase, until it stops."""
    stair, history = Staircase(**vs.STAIRCASE), []
    while not stair.done():
        passed = rng.random() < p_pass
        history.append((stair.freq, passed))
        stair.update(passed)
    return history


RANDOM_RUNS = [_random_run(random.Random(seed), p) for seed in range(60) for p in (0.2, 0.5, 0.8, 0.97)]


def test_scoring_agrees_with_the_real_staircase():
    """Same next level, same reversals and same stop reason as Staircase itself, on the named
    patterns and on 240 random runs."""
    for history in [seq(p[1]) for p in PATTERNS] + RANDOM_RUNS:
        stair, score = replay(history), vs.score_staircase(history, **vs.STAIRCASE)
        assert score["legacy_next_level_cpd"] == stair.freq
        assert score["reversals"] == stair.reversals
        assert score["stop_reason"] == stop_reason_of(stair)
        assert (score["stop_reason"] is not None) == stair.done()


def test_an_empty_run_has_no_threshold():
    score = vs.score_staircase([], **vs.STAIRCASE)
    assert score["legacy_next_level_cpd"] == 2.0 and score["highest_passed_cpd"] is None
    assert score["stop_reason"] is None and score["floor"] is False


# ---------- the value that is reported ----------
def test_highest_passed_never_reports_a_level_that_was_not_passed():
    for history in RANDOM_RUNS:
        reported = vs.report_threshold(vs.score_staircase(history, **vs.STAIRCASE), "highest_passed")
        if reported["measurable"]:
            assert reported["threshold_cpd"] in {cpd for cpd, passed in history if passed}
            assert reported["rule_used"] == "highest_passed"


def test_last4_rule_is_the_mean_of_exactly_those_four_reversals():
    seen = 0
    for history in RANDOM_RUNS:
        score = vs.score_staircase(history, **vs.STAIRCASE)
        reported = vs.report_threshold(score, "last4_reversal_mean")
        if len(score["reversals"]) >= 4:
            seen += 1
            assert reported["threshold_cpd"] == sum(score["reversals"][-4:]) / 4
            assert reported["rule_used"] == "last4_reversal_mean" and not reported["ceiling"]
    assert seen > 50


def test_legacy_rule_keeps_the_old_value():
    reported = vs.report_threshold(vs.score_staircase(seq(PATTERNS[0][1]), **vs.STAIRCASE), "legacy_next_level")
    assert reported["threshold_cpd"] == 10.0          # a level that was never passed: why it is legacy


@pytest.mark.parametrize("rule", vs.THRESHOLD_RULES)
def test_a_floor_run_is_not_measurable_under_any_rule(rule):
    reported = vs.report_threshold(vs.score_staircase(seq("F2 P2 F4 F2 F2 F2"), **vs.STAIRCASE), rule)
    assert reported == {"threshold_cpd": None, "rule_used": rule, "measurable": False,
                        "not_measurable_reason": "floor", "ceiling": False}


def test_a_run_without_four_reversals_falls_back_to_the_highest_level_passed():
    all_passed = vs.report_threshold(vs.score_staircase(seq(PATTERNS[5][1]), **vs.STAIRCASE))
    assert all_passed == {"threshold_cpd": 18.0, "rule_used": "highest_passed", "measurable": True,
                          "not_measurable_reason": None, "ceiling": True}       # 20 cpd was never shown
    fails_above_floor = vs.report_threshold(vs.score_staircase(seq(PATTERNS[6][1]), **vs.STAIRCASE))
    assert (fails_above_floor["threshold_cpd"], fails_above_floor["rule_used"]) == (10.0, "highest_passed")
    assert fails_above_floor["ceiling"] is False


def test_an_unfinished_run_is_not_measurable():
    reported = vs.report_threshold(vs.score_staircase(seq("P2 P4 F6"), **vs.STAIRCASE))
    assert (reported["measurable"], reported["not_measurable_reason"]) == (False, "not_finished")


def test_unknown_rule_is_an_error():
    with pytest.raises(ValueError):
        vs.report_threshold(vs.score_staircase(seq("P2"), **vs.STAIRCASE), "best_guess")


# ---------- conversion ----------
@pytest.mark.parametrize("cpd, conversion, decimal", [
    (7.5, "cpd_div_30", 0.25), (7.5, "cpd_div_20_legacy", 0.375), (30.0, "cpd_div_30", 1.0),
    (7.5, "none", None), (None, "cpd_div_30", None),
])
def test_conversion(cpd, conversion, decimal):
    assert vs.convert(cpd, conversion) == decimal


def test_defaults_are_the_documented_placeholders():
    assert vs.DEFAULT_THRESHOLD_RULE == "last4_reversal_mean" and vs.DEFAULT_CONVERSION == "none"
    assert vs.DEFAULT_THRESHOLD_RULE in vs.THRESHOLD_RULES and vs.DEFAULT_CONVERSION in vs.CONVERSIONS


# ---------- va_result.json and the result screen ----------
def with_catch(text):
    """All trials of a run: the normal ones, with a catch trial (an unresolvable 60 cpd) after the 3rd."""
    trials = [(cpd, passed, "normal") for cpd, passed in seq(text)]
    return trials[:3] + [(60.0, False, "catch")] + trials[3:]


def test_result_file_content():
    result = vs.build_result(with_catch(PATTERNS[0][1]), app_version="1.8.0")
    assert json.loads(json.dumps(result)) == result          # plain JSON
    assert (result["schema"], result["schema_version"], result["provisional"]) == ("ntuh.va_result", 1, True)
    assert (result["app"], result["app_version"]) == ("VA_center_opt", "1.8.0")
    assert (result["threshold_rule"], result["threshold_rule_used"], result["threshold_cpd"]) == (
        "last4_reversal_mean", "last4_reversal_mean", 7.5)
    assert result["rules"] == {"last4_reversal_mean_cpd": 7.5, "highest_passed_cpd": 8.0,
                               "legacy_next_level_cpd": 10.0}
    assert result["reversals"] == [8.0, 4.0, 10.0, 8.0]
    assert (result["stop_reason"], result["staircase_finished"], result["floor"]) == ("four_reversals", True, False)
    assert (result["n_trials"], result["n_normal_trials"]) == (11, 10)       # the catch trial is listed, not scored
    assert result["trials"][3] == {"trial": 4, "cpd": 60.0, "result": "FAIL", "trial_type": "catch"}
    assert result["trials"][0] == {"trial": 1, "cpd": 2.0, "result": "PASS", "trial_type": "normal"}
    assert (result["conversion"], result["decimal_va"]) == ("none", None)
    assert result["conversions"] == {"cpd_div_30": 0.25, "cpd_div_20_legacy": 0.375}   # both, whatever is shown
    assert result["legacy"] == {"next_level_cpd": 10.0, "va_score": 0.5}     # what 1.5.0 displayed
    assert result["staircase"] == {"start": 2.0, "step": 2.0, "minv": 2.0, "maxv": 20.0}


def test_a_catch_trial_does_not_change_the_score():
    assert (vs.build_result(with_catch(PATTERNS[1][1]))["rules"]
            == vs.build_result([(c, p, "normal") for c, p in seq(PATTERNS[1][1])])["rules"])


def test_unknown_settings_fall_back_to_the_defaults():
    result = vs.build_result(with_catch(PATTERNS[0][1]), rule="best_guess", conversion="snellen")
    assert (result["threshold_rule"], result["conversion"]) == ("last4_reversal_mean", "none")


def test_result_screen_default_shows_cpd_rule_and_provisional():
    assert vs.result_lines(vs.build_result(with_catch(PATTERNS[0][1]))) == [
        "Threshold: 7.50 cpd", "Rule: mean of the last 4 reversals", "PROVISIONAL - not for clinical use"]


def test_result_screen_with_a_conversion_names_the_convention():
    lines = vs.result_lines(vs.build_result(with_catch(PATTERNS[0][1]), "highest_passed", "cpd_div_30"))
    assert lines == ["Threshold: 8.00 cpd", "Rule: highest level passed", "Decimal VA: 0.27 (cpd / 30)",
                     "PROVISIONAL - not for clinical use"]
    legacy = vs.result_lines(vs.build_result(with_catch(PATTERNS[0][1]), "legacy_next_level", "cpd_div_20_legacy"))
    assert legacy[:3] == ["Threshold: 10.00 cpd", "Rule: next level (legacy, before 1.8.0)",
                          "Decimal VA: 0.50 (cpd / 20, legacy)"]


@pytest.mark.parametrize("conversion", vs.CONVERSIONS)
def test_result_screen_at_the_floor_says_not_measurable_never_a_score(conversion):
    result = vs.build_result(with_catch("F2 P2 F4 F2 F2 F2"), conversion=conversion)
    lines = vs.result_lines(result)
    assert lines[0] == "Not measurable (<= 2 cpd floor)"
    assert not any("0.1" in line or "Decimal" in line for line in lines)
    assert (result["threshold_cpd"], result["decimal_va"]) == (None, None)
    assert result["legacy"] == {"next_level_cpd": 2.0, "va_score": 0.1}      # the old value stays in the file


def test_result_screen_when_the_subject_passed_every_level():
    lines = vs.result_lines(vs.build_result(with_catch(PATTERNS[5][1]), conversion="cpd_div_30"))
    assert lines[:3] == ["Threshold: >= 18.00 cpd", "Rule: highest level passed (passed every level shown)",
                         "Decimal VA: >= 0.60 (cpd / 30)"]


def test_result_screen_says_when_the_rule_could_not_be_applied():
    lines = vs.result_lines(vs.build_result(with_catch(PATTERNS[6][1])))
    assert lines[:2] == ["Threshold: 10.00 cpd", "Rule: highest level passed (fewer than 4 reversals)"]


# ---------- opt-in: the 2026-09-18 field runs ----------
FIELD_ROOT = os.environ.get("NTUH_VA_0918_ROOT")
# (last-4 mean, highest passed) by User Name or its subject code; "floor" = not measurable.
FIELD_EXPECTED = {
    "s01": (13.5, 16.0), "s02": (5.5, 6.0), "s03": (7.5, 8.0), "s04": (15.5, 16.0), "s06": (13.0, 12.0),
    "s05_c5": (3.0, 2.0), "s00": "floor", "s05_c5-1": "floor", "s07": "floor",
}


@pytest.mark.skipif(not FIELD_ROOT, reason="set NTUH_VA_0918_ROOT to the 2026-09-18 VA dataset folder")
def test_replay_of_the_20260918_field_runs():
    logs = sorted(Path(FIELD_ROOT).glob("VA_*/trial_events.csv"))
    assert logs, f"no VA_*/trial_events.csv under {FIELD_ROOT}"
    print(f"\n{'user':<10} {'legacy':>6} {'highest':>7} {'last4':>6}  stop / reported")
    checked = set()
    for log in logs:
        user = re.fullmatch(r"VA_(.+)_sess\d+_\d{8}_\d{6}", log.parent.name)[1]
        with open(log, newline="", encoding="utf-8") as f:
            trials = [(float(r["cpd"]), r["result"] == "PASS", r.get("trial_type") or "normal")
                      for r in csv.DictReader(f)]
        normal = [(cpd, passed) for cpd, passed, kind in trials if kind == "normal"]
        stair = replay(normal)                              # every logged level is on the staircase path
        result = vs.build_result(trials)
        rules = result["rules"]
        assert rules["legacy_next_level_cpd"] == stair.freq, user
        print(f"{user:<10} {rules['legacy_next_level_cpd']:>6g} {str(rules['highest_passed_cpd']):>7} "
              f"{str(rules['last4_reversal_mean_cpd']):>6}  {result['stop_reason']} / {vs.result_lines(result)[0]}")
        key = user if user in FIELD_EXPECTED else user.split("_")[0]
        if key in FIELD_EXPECTED:
            checked.add(key)
            if FIELD_EXPECTED[key] == "floor":
                assert result["floor"] and not result["measurable"], user
            else:
                assert (rules["last4_reversal_mean_cpd"], rules["highest_passed_cpd"]) == FIELD_EXPECTED[key], user
                assert not result["floor"], user
    assert checked == set(FIELD_EXPECTED), f"runs not found: {sorted(set(FIELD_EXPECTED) - checked)}"
