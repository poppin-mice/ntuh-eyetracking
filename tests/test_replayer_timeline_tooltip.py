# -*- coding: utf-8 -*-
"""Hover tooltip text of the replayer timeline (ntuh/replayer/widgets/timeline.py).

Replayer 1.2.0 gave every timeline trial a 7th field (trial_type), but the tooltip still
unpacked 6, so hovering any trial raised inside paintEvent and PyQt6 closed the app. The
text now comes from trial_tooltip_text(), which takes the full trial.

timeline.py imports PyQt6 at module level and CI does not install it, so the function is
compiled from the source file with the load_functions fixture instead of being imported.
The trials below are synthetic.
"""
import pytest


@pytest.fixture(scope="module")
def tooltip(load_functions):
    (fn,) = load_functions("ntuh/replayer/widgets/timeline.py", ["trial_tooltip_text"])
    return fn


def _trial(result="PASS", cpd=6.0, label=None, ttype="normal"):
    """A trial as TimelineWidget.set_trials() stores it (7 fields)."""
    return (12.5, 17.5, result, 3, cpd, label, ttype)


def test_normal_trial_without_review_label(tooltip):
    assert tooltip(_trial()) == "Trial 3  CPD:6.0  PASS"


def test_review_label_is_appended(tooltip):
    assert tooltip(_trial(result="FAIL", label="pass")) == "Trial 3  CPD:6.0  FAIL -> pass"


def test_catch_trial_is_tagged(tooltip):
    assert tooltip(_trial(cpd=29.1, ttype="catch")) == "Trial 3  CPD:29.1  CATCH  PASS"


def test_labelled_catch_trial_keeps_both_marks(tooltip):
    text = tooltip(_trial(cpd=29.1, label="discard", ttype="catch"))
    assert text == "Trial 3  CPD:29.1  CATCH  PASS -> discard"


@pytest.mark.parametrize("label", [None, ""])
def test_missing_review_label_adds_no_arrow(tooltip, label):
    assert "->" not in tooltip(_trial(label=label))
