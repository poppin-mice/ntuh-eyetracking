# -*- coding: utf-8 -*-
"""tools/rescore_session.py: scoring a session again with unchanged code reproduces the result it
saved, and never touches the session folder. Also the VA result itself (build_va_result).
Synthetic sessions and participant codes only.
"""
import csv
import json

import pytest

from ntuh.recording import session_bundle as sb
from ntuh.sol import accuracy_test as acc
from ntuh.vavf.stimuli import VA_RESULT_FILE, build_va_result
from ntuh.version import get_version
from tools import check_session_bundle as chk
from tools import rescore_session as rs

# Passes 2 and 4, a catch trial, then fails 6 / passes 4 twice: four reversals, next level 6.
TRIALS = [(2.0, True, "normal"), (4.0, True, "normal"), (38.4, True, "catch"), (6.0, False, "normal"),
          (4.0, True, "normal"), (6.0, False, "normal"), (4.0, True, "normal")]
TRIAL_COLUMNS = ["trial_number", "cpd", "side", "start_timestamp", "end_timestamp", "result",
                 "stim_x", "stim_y", "eval_source", "trial_type"]          # as recorder.py writes them


def va_session(root, trials=TRIALS, saved=None):
    """A finished VA session folder: trial_events.csv and va_result.json as the test writes them."""
    folder = root / "VA_output" / "VA_s01_c3_sess1_20261122_100305"
    bundle = sb.start(folder, "va", user_name="s01_c3")
    with open(folder / "trial_events.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(TRIAL_COLUMNS)
        for i, (cpd, passed, kind) in enumerate(trials, start=1):
            writer.writerow([i, cpd, "left", 100.0 + i, 105.0 + i, "PASS" if passed else "FAIL",
                             960, 540, "Sol", kind])
    bundle.write_json(VA_RESULT_FILE, saved or build_va_result(trials, get_version("VA_center_opt")))
    bundle.finish()
    return folder


def accuracy_session(root):
    folder = root / "accuracy_test" / "s01_c3_20261122_100305"
    bundle = sb.start(folder, "accuracy_test", user_name="s01_c3")
    targets = [{"name": "center", "group": "center", "ecc_deg": 0.0, "x": 1920, "y": 1080},
               {"name": "ring10_0", "group": "10deg", "ecc_deg": 10.0, "x": 2603, "y": 1080},
               {"name": "ring10_1", "group": "10deg", "ecc_deg": 10.0, "x": 1237, "y": 1080}]
    records = [acc.compute_point_record(
        t, [(t["x"] + 31.7 + i, t["y"] - 12.3 + 0.7 * i) for i in range(30)],
        [(t["x"] + 3.1 + i / 3, t["y"] - 1.9 + 0.2 * i) for i in range(30)], 60.0, 3840, 59.5)
        for t in targets]
    acc.save_accuracy_report(folder, records, {"user": "s01_c3", "n_points": len(records),
                                               "result": acc.result_block(records)})
    bundle.finish()
    return folder


def _snapshot(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


# ---------- rescoring ----------
def test_rescoring_a_va_session_with_unchanged_code_reproduces_va_result(bundle_app):
    folder = va_session(bundle_app)
    before = _snapshot(folder)
    out, record = rs.rescore(folder)

    saved = json.loads((folder / VA_RESULT_FILE).read_text(encoding="utf-8"))
    assert record["identical_to_saved"] is True
    assert record["result"] == saved                                    # every value, exactly
    assert (saved["stop_reason"], saved["legacy"]) == ("four_reversals", {"next_level_cpd": 6.0, "va_score": 0.3})
    version = get_version("VA_center_opt")
    assert out == bundle_app / "VA_output" / "_derived" / folder.name / f"rescore_{version}.json"
    assert json.loads(out.read_text(encoding="utf-8")) == record
    assert (record["session_id"], record["session_kind"]) == (folder.name, "va")
    assert record["session_app_version"] == record["rescored_with"]["app_version"] == version
    assert {"build_id", "dirty"} <= set(record["rescored_with"])        # which code did the rescoring
    assert record["inputs"] == {"trial_events.csv": {"sha256": sb.sha256_file(folder / "trial_events.csv")}}
    assert _snapshot(folder) == before                                  # the session folder is only read
    assert chk.check_session(folder)["verdict"] == "PASS"


def test_rescoring_an_accuracy_test_reproduces_its_summary_and_result(bundle_app):
    folder = accuracy_session(bundle_app)
    before = _snapshot(folder)
    out, record = rs.rescore(folder, bundle_app / "derived")

    report = json.loads((folder / "accuracy_report.json").read_text(encoding="utf-8"))
    assert record["identical_to_saved"] is True
    assert record["result"] == {"summary": report["summary"], "result": report["meta"]["result"]}
    assert record["result"]["result"]["n"] == 3
    assert out.parent == bundle_app / "derived" / folder.name
    assert _snapshot(folder) == before


def test_a_result_the_current_code_would_not_give_is_reported_as_different(bundle_app, capsys):
    wrong = dict(build_va_result(TRIALS, get_version("VA_center_opt")), stop_reason="max_streak")
    folder = va_session(bundle_app, saved=wrong)
    assert rs.main([str(folder)]) == 1
    assert "DIFFERENT from the saved result" in capsys.readouterr().out
    out, record = rs.rescore(folder)
    assert record["identical_to_saved"] is False and record["result"]["stop_reason"] == "four_reversals"


def test_a_result_saved_by_another_version_still_compares_equal(bundle_app):
    folder = va_session(bundle_app, saved=build_va_result(TRIALS, "0.0.1"))
    assert rs.main([str(folder)]) == 0                                   # app_version is not the result


def test_what_cannot_be_rescored_is_refused(bundle_app, tmp_path, capsys):
    folder = va_session(bundle_app)
    assert rs.main([str(tmp_path)]) == 2                                 # not a session bundle
    assert rs.main([str(folder), "--out", str(folder / "derived")]) == 2
    assert "outside the session folder" in capsys.readouterr().err
    vf = bundle_app / "VF_output" / "VF_s01_c3_sess1_20261122_100305"
    sb.start(vf, "vf", user_name="s01_c3").finish()
    assert rs.main([str(vf)]) == 2
    assert "nothing to rescore for a 'vf' session" in capsys.readouterr().err


# ---------- the VA result ----------
@pytest.mark.parametrize("outcomes, stop_reason, next_level, va_score", [
    ([True] * 9, "max_streak", 20.0, 0.9),                    # passed 2..18: the next level is the top
    ([False] * 4, "four_consecutive_fails", 2.0, 0.1),        # failed at the floor four times
    ([True, True, False], None, 4.0, 0.2),                    # the operator stopped the test
])
def test_the_va_result_names_how_the_staircase_ended(outcomes, stop_reason, next_level, va_score):
    level, trials = 2.0, []
    for passed in outcomes:                                    # the levels a real run would show
        trials.append((level, passed, "normal"))
        level = min(20.0, max(2.0, level + (2.0 if passed else -2.0)))
    result = build_va_result(trials, "1.9.0")
    assert (result["schema"], result["schema_version"], result["app_version"]) == ("ntuh.va_result", 1, "1.9.0")
    assert (result["stop_reason"], result["staircase_finished"]) == (stop_reason, stop_reason is not None)
    assert result["legacy"] == {"next_level_cpd": next_level, "va_score": va_score}
    assert (result["n_trials"], result["n_normal_trials"]) == (len(outcomes), len(outcomes))


def test_a_catch_trial_is_counted_but_never_moves_the_staircase():
    with_catch = build_va_result(TRIALS)
    without = build_va_result([t for t in TRIALS if t[2] == "normal"])
    assert (with_catch["n_trials"], with_catch["n_normal_trials"]) == (7, 6)
    assert with_catch["trials"][2] == {"trial": 3, "cpd": 38.4, "result": "PASS", "trial_type": "catch"}
    for key in ("stop_reason", "reversals", "legacy"):
        assert with_catch[key] == without[key]
    assert with_catch["reversals"] == [6.0, 4.0, 6.0, 4.0]
