# -*- coding: utf-8 -*-
"""VA scoring: which threshold a staircase run actually reached, and how it is reported.

The staircase procedure itself (ntuh.vavf.stimuli.Staircase) is not changed here. This module
only reads what a run did. Pure: no pygame, no files.

Until 1.8.0 the test reported the level the staircase would have shown NEXT, which after a final
PASS is one step above anything the subject passed, and converted it with cpd / 20. Both the
rule and the conversion are still to be agreed with the clinicians (REQ-0025, REQ-0026), so
every result is provisional and the two choices are settings.
"""
from ntuh.vavf.stimuli import get_va_result_cpd

# The staircase the VA test runs (va_test.py builds it from this).
STAIRCASE = {"start": 2.0, "step": 2.0, "minv": 2.0, "maxv": 20.0}

THRESHOLD_RULES = ("last4_reversal_mean", "highest_passed", "legacy_next_level")
# PLACEHOLDER, pending Edan and the clinicians. This constant is the one place to change it.
DEFAULT_THRESHOLD_RULE = "last4_reversal_mean"
RULE_LABELS = {
    "last4_reversal_mean": "mean of the last 4 reversals",
    "highest_passed": "highest level passed",
    "legacy_next_level": "next level (legacy, before 1.8.0)",
}

CONVERSIONS = ("none", "cpd_div_30", "cpd_div_20_legacy")
DEFAULT_CONVERSION = "none"
CONVERSION_DIVISORS = {"cpd_div_30": 30.0, "cpd_div_20_legacy": 20.0}
CONVERSION_LABELS = {
    "cpd_div_30": "cpd / 30",
    "cpd_div_20_legacy": "cpd / 20, legacy",
}

PROVISIONAL_TEXT = "PROVISIONAL - not for clinical use"
RESULT_FILE = "va_result.json"


def score_staircase(history, start=2.0, step=2.0, minv=2.0, maxv=20.0):
    """Every candidate threshold of one run.

    history: ordered (cpd, passed) of the NORMAL trials. Catch trials never enter the staircase.
    A reversal is the level of a trial whose outcome differs from the previous trial's, as in
    Staircase.update(). The stop reasons mirror Staircase.done(), in its order; None means the
    staircase had not finished (the operator stopped the test).
    """
    hist = [(float(cpd), bool(passed)) for cpd, passed in history]
    reversals = [cpd for (cpd, passed), (_, before) in zip(hist[1:], hist[:-1]) if passed != before]
    passed_levels = [cpd for cpd, passed in hist if passed]

    best_pass_streak = streak = 0
    for _, passed in hist:
        streak = streak + 1 if passed else 0
        best_pass_streak = max(best_pass_streak, streak)
    fails_at_end = 0
    for _, passed in reversed(hist):
        if passed:
            break
        fails_at_end += 1

    if hist:
        last_cpd, last_passed = hist[-1]
        next_level = min(maxv, max(minv, last_cpd + (step if last_passed else -step)))
    else:
        next_level = float(start)

    if len(reversals) >= 4:
        stop_reason = "four_reversals"
    elif hist and next_level >= maxv and best_pass_streak >= 3:
        stop_reason = "max_streak"
    elif fails_at_end >= 4:
        stop_reason = "four_consecutive_fails"
    else:
        stop_reason = None

    last4 = reversals[-4:]
    return {
        "legacy_next_level_cpd": next_level,
        "highest_passed_cpd": max(passed_levels) if passed_levels else None,
        "last4_reversal_mean_cpd": sum(last4) / 4.0 if len(last4) == 4 else None,
        "reversals": reversals,
        "stop_reason": stop_reason,
        # Stopped on consecutive FAILs with the last one at the lowest level: below what the test shows.
        "floor": stop_reason == "four_consecutive_fails" and hist[-1][0] <= minv,
        "n_normal_trials": len(hist),
    }


def report_threshold(score, rule=DEFAULT_THRESHOLD_RULE):
    """The one value to report for a run under `rule`.

    Returns threshold_cpd (None when not measurable), rule_used, measurable,
    not_measurable_reason (floor | not_finished | nothing_passed | None) and ceiling (the test
    ended at the top of its range, so the threshold is a lower bound).

    The mean of the last 4 reversals does not exist for a run that ended before its 4th reversal
    (the subject passed all the way up, or failed 4 times in a row above the floor). Such a run
    reports the highest level passed instead, and rule_used says so.
    """
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"unknown threshold rule {rule!r}")
    out = {"threshold_cpd": None, "rule_used": rule, "measurable": False,
           "not_measurable_reason": None, "ceiling": False}
    if score["stop_reason"] is None:
        out["not_measurable_reason"] = "not_finished"
        return out
    if score["floor"]:
        out["not_measurable_reason"] = "floor"
        return out

    if rule == "legacy_next_level":
        value = score["legacy_next_level_cpd"]
    elif rule == "last4_reversal_mean" and score["last4_reversal_mean_cpd"] is not None:
        value = score["last4_reversal_mean_cpd"]
    else:
        out["rule_used"] = "highest_passed"
        value = score["highest_passed_cpd"]
    if value is None:
        out["not_measurable_reason"] = "nothing_passed"
        return out
    out.update(threshold_cpd=value, measurable=True, ceiling=score["stop_reason"] == "max_streak")
    return out


def convert(cpd, conversion):
    """Decimal acuity under one convention; None for no conversion or no threshold."""
    if cpd is None or conversion not in CONVERSION_DIVISORS:
        return None
    return round(cpd / CONVERSION_DIVISORS[conversion], 3)


def build_result(trials, rule=None, conversion=None, app_version=None):
    """The content of va_result.json, which is also what the result screen shows.

    trials: ordered (cpd, passed, trial_type) of EVERY trial, trial_type "normal" or "catch".
    An unknown rule or conversion (for example from an old settings file) falls back to the
    default.
    """
    rule = rule if rule in THRESHOLD_RULES else DEFAULT_THRESHOLD_RULE
    conversion = conversion if conversion in CONVERSIONS else DEFAULT_CONVERSION
    trials = [(float(cpd), bool(passed), str(kind)) for cpd, passed, kind in trials]
    score = score_staircase([(cpd, passed) for cpd, passed, kind in trials if kind == "normal"],
                            **STAIRCASE)
    reported = report_threshold(score, rule)
    threshold = reported["threshold_cpd"]
    return {
        "schema": "ntuh.va_result", "schema_version": 1,
        "provisional": True,                    # until the rule and the conversion are agreed
        "app": "VA_center_opt", "app_version": app_version,
        "threshold_rule": rule,
        "threshold_rule_used": reported["rule_used"],
        "threshold_cpd": threshold,
        "measurable": reported["measurable"],
        "not_measurable_reason": reported["not_measurable_reason"],
        "floor": score["floor"],
        "ceiling": reported["ceiling"],
        "staircase_finished": score["stop_reason"] is not None,
        "stop_reason": score["stop_reason"],
        "rules": {key: score[key] for key in ("last4_reversal_mean_cpd", "highest_passed_cpd",
                                              "legacy_next_level_cpd")},
        "reversals": score["reversals"],
        "n_trials": len(trials),
        "n_normal_trials": score["n_normal_trials"],
        "trials": [{"trial": i, "cpd": cpd, "result": "PASS" if passed else "FAIL", "trial_type": kind}
                   for i, (cpd, passed, kind) in enumerate(trials, start=1)],
        "conversion": conversion,
        "decimal_va": convert(threshold, conversion),
        "conversions": {name: convert(threshold, name) for name in CONVERSION_DIVISORS},
        # What versions before 1.8.0 displayed, to compare with scores already written on forms.
        "legacy": {"next_level_cpd": score["legacy_next_level_cpd"],
                   "va_score": get_va_result_cpd(score["legacy_next_level_cpd"])},
        "staircase": dict(STAIRCASE),
    }


def result_lines(result):
    """The lines of the result screen, top to bottom."""
    at_least = ">= " if result["ceiling"] else ""       # the subject passed everything shown
    if result["measurable"]:
        first = f"Threshold: {at_least}{result['threshold_cpd']:.2f} cpd"
    elif result["floor"]:
        first = f"Not measurable (<= {result['staircase']['minv']:g} cpd floor)"
    elif result["not_measurable_reason"] == "not_finished":
        first = "Not measurable (test not finished)"
    else:
        first = "Not measurable (no level passed)"
    rule_line = f"Rule: {RULE_LABELS[result['threshold_rule_used']]}"
    if result["ceiling"]:
        rule_line += " (passed every level shown)"
    elif result["threshold_rule_used"] != result["threshold_rule"]:
        rule_line += " (fewer than 4 reversals)"
    lines = [first, rule_line]
    if result["decimal_va"] is not None:
        lines.append(f"Decimal VA: {at_least}{result['decimal_va']:.2f} "
                     f"({CONVERSION_LABELS[result['conversion']]})")
    lines.append(PROVISIONAL_TEXT)
    return lines
