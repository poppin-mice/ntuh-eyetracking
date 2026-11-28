# -*- coding: utf-8 -*-
"""Start-screen identity (B-D09 part a): subject-code check, code cleaning, derived User Name,
folder names, run number, the subject / condition lock and where outputs go.
Participant codes only."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ntuh.common import identity as idn

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------- subject-code check ----------
@pytest.mark.parametrize("code", ["s01", "K03", "A02", "abc123", "S105"])
def test_subject_codes_are_accepted(code):
    assert idn.subject_code_error(code) is None


@pytest.mark.parametrize("code", ["", "s 01", " s01", "s01 ", "王小明", "s０1", "ｓ01", "s01_c3",
                                  "s1", "abcd01", "s0001", "01"])
def test_names_spaces_and_other_shapes_are_refused(code):
    assert idn.subject_code_error(code)


def test_spaces_and_non_ascii_are_refused_even_by_a_looser_pattern():
    assert idn.subject_code_error("s 01", r".+")
    assert idn.subject_code_error("王01", r".+")
    assert idn.subject_code_error("x_7", r".+") is None


def test_a_broken_pattern_from_the_settings_falls_back_to_the_default():
    assert idn.subject_code_error("s01", r"([") is None
    assert idn.subject_code_error("s01_c3", r"([")


# ---------- code cleaning ----------
@pytest.mark.parametrize("raw, clean", [
    ("s01_c3", "s01_c3"), ("s05_c5-1", "s05_c5-1"), ("A02_05", "A02_05"),   # field shapes stay
    ("  s01  ", "s01"),             # outer blanks dropped
    ("s01 c3", "s01_c3"),           # inner whitespace -> '_' (the webcam profile folder rule)
    ("a  b", "a__b"),               # one '_' per whitespace character, as before
    ("../../s01", "....s01"), ("sub/dir", "subdir"), ("C:\\s01", "Cs01"),  # never a path
    ("", "anonymous"), (None, "anonymous"), ("///", "anonymous"),
])
def test_safe_code(raw, clean):
    assert idn.safe_code(raw) == clean


def test_safe_code_default_is_per_site():
    assert idn.safe_code("  ", default="default") == "default"


# ---------- derived User Name and folder names ----------
def test_user_name_follows_the_field_naming():
    assert idn.user_name("s01", "C3") == "s01_c3"
    assert idn.user_name("K03", "") == "K03"


@pytest.mark.parametrize("kind, run, name", [
    ("va", 2, "VA_s01_c3_sess2_20261124_101500"),
    ("vf", 1, "VF_s01_c3_sess1_20261124_101500"),
    ("accuracy_test", 3, "s01_c3_20261124_101500"),
    ("sol2d_calibration", None, "s01_c3_20261124_101500"),
])
def test_session_folder_names_keep_todays_pattern(kind, run, name):
    assert idn.session_folder_name(kind, "s01_c3", run, "20261124_101500") == name


# ---------- run number ----------
def test_run_number_counts_todays_sessions_of_the_same_subject_condition_and_task(tmp_path):
    root = tmp_path / "VA_output"
    root.mkdir()
    for name in ["VA_s01_c3_sess1_20261124_090000",      # counted
                 "VA_s01_c5_sess1_20261124_093000",      # other condition
                 "VA_s02_c3_sess1_20261124_094000",      # other subject
                 "VA_s01_c3_sess1_20261123_090000",      # other day
                 "last_settings_opt.json"]:
        (root / name).mkdir()
    assert idn.next_run("va", "s01_c3", "20261124", app_dir=tmp_path) == 2
    assert idn.next_run("vf", "s01_c3", "20261124", app_dir=tmp_path) == 1     # other task
    assert idn.next_run("va", "s09_c3", "20261124", app_dir=tmp_path) == 1


def test_run_number_never_repeats_a_number_used_that_day():
    names = ["VA_s01_c3_sess1_20261124_090000", "VA_s01_c3_sess4_20261124_100000"]
    assert idn.next_repeat_index(names, "va", "s01_c3", "20261124") == 5
    # Older builds wrote sess1 every time: three of them that day -> 4.
    old = [f"VA_s01_c3_sess1_20261124_0{h}0000" for h in (8, 9)] + ["VA_s01_c3_sess1_20261124_100000"]
    assert idn.next_repeat_index(old, "va", "s01_c3", "20261124") == 4


def test_run_number_of_tasks_without_sessN_counts_the_folders():
    names = ["s01_c3_20261124_090000", "s01_c3_20261124_091500", "s01_c3_c3_20261124_092000"]
    assert idn.next_repeat_index(names, "accuracy_test", "s01_c3", "20261124") == 3


def test_missing_output_folder_means_run_one(tmp_path):
    assert idn.next_run("va", "s01_c3", "20261124", app_dir=tmp_path) == 1


# ---------- lock ----------
def test_lock_holds_from_calibration_until_that_conditions_test_completes():
    lock = idn.IdentityLock()
    assert not lock.locked
    lock.calibration_started("s01", "C3")
    assert lock.held == ("s01", "C3")
    assert not lock.accuracy_test_ended("C3", completed=False)     # aborted: run it again
    assert not lock.accuracy_test_ended("C5", completed=True)      # another condition
    assert lock.locked
    assert lock.accuracy_test_ended("C3", completed=True)
    assert not lock.locked


def test_unlock_is_the_explicit_way_out():
    lock = idn.IdentityLock()
    lock.calibration_started("s01", "C5")
    assert lock.unlock() == ("s01", "C5")
    assert not lock.locked


# ---------- output location ----------
def _app_dir_seen_by(entry_script, cwd, env_extra=None):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    env.pop("NTUH_APP_DIR", None)
    env.update(env_extra or {})
    out = subprocess.run([sys.executable, str(entry_script)], cwd=str(cwd), env=env,
                         capture_output=True, text=True, timeout=60, check=True)
    return Path(out.stdout.strip())


def test_outputs_go_under_the_program_folder_whatever_the_working_directory(tmp_path):
    app, elsewhere = tmp_path / "app", tmp_path / "elsewhere"
    app.mkdir()
    elsewhere.mkdir()
    entry = app / "entry.py"
    entry.write_text("from ntuh.common import identity\nprint(identity.output_root('va'))\n",
                     encoding="utf-8")
    assert _app_dir_seen_by(entry, elsewhere) == (app / "VA_output").resolve()


def test_ntuh_app_dir_overrides_the_program_folder_of_a_source_run(tmp_path):
    app, data = tmp_path / "app", tmp_path / "data"
    app.mkdir()
    entry = app / "entry.py"
    entry.write_text("from ntuh.common.app_env import APP_DIR\nprint(APP_DIR)\n", encoding="utf-8")
    assert _app_dir_seen_by(entry, tmp_path, {"NTUH_APP_DIR": str(data)}) == data.resolve()
