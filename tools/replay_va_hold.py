# -*- coding: utf-8 -*-
"""Replay recorded VA sessions through the current pass rule (ntuh/vavf/hold_tracker.py) and list
the trials whose PASS / FAIL would change.

    python tools/replay_va_hold.py <folder of VA session folders> [--pass-dur 2.0] [--stim-dur 5.0]
                                   [--max-age 0.2] [--blink-grace 0.3]

For each trial scored on Sol gaze, the rows of sol_gaze_data.csv between the trial's start and end
are fed to a HoldTracker, one update per 1/60 s, and the outcome is compared with the result in
trial_events.csv. The session folders are only read; the output names sessions by the code in the
folder name and trials by number. Outcome under the current rule:

    PASS   the hold reaches the pass duration (to within the time resolution of the file)
    FAIL   the stimulus time is up and no hold is in progress: the trial ends without a pass
    OPEN   the recording stops first. The old rule ended the trial there with a PASS; under the
           current rule the trial would have gone on, and the file cannot say how it ends

Approximate. sol_gaze_data.csv holds one row per rendered frame that had a new mapped sample
(about a sixth of the samples the test received), stamped when the frame was recorded and not when
the hold was evaluated. A hold is therefore accepted when it falls short of the pass duration by
no more than one recorded frame plus two replay steps.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntuh.vavf.hold_tracker import (  # noqa: E402  (pure, stdlib only)
    DEFAULT_BLINK_GRACE_S, DEFAULT_MAX_SAMPLE_AGE_S, GazeSample, HoldTracker)

STEP_S = 1 / 60.0       # the test evaluates the hold once per rendered frame, at most 60 a second


def _sol_rows(folder):
    """(receive time s, usable, x, y) of every row of sol_gaze_data.csv, in file order. Usable as
    in ntuh.tracking.sol_quality.sol_sample_usable: is_valid and not both eyes NO_EYE."""
    rows = []
    with open(folder / "sol_gaze_data.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            no_eye = r.get("left_eye_status") == "0" and r.get("right_eye_status") == "0"
            rows.append((int(r["pc_timestamp_ms"]) / 1000.0, r["is_valid"] == "1" and not no_eye,
                         float(r["mapped_gaze_x"]), float(r["mapped_gaze_y"])))
    return rows


def replay_session(folder, pass_dur=2.0, stim_dur=5.0, max_age=DEFAULT_MAX_SAMPLE_AGE_S,
                   blink_grace=DEFAULT_BLINK_GRACE_S):
    """One dict per Sol-scored trial of a session folder: session, trial, trial_type, was
    (trial_events.csv), now (PASS / FAIL / OPEN), held (seconds that counted in the hold in
    progress when the replay stops) and valid_final (fraction of usable rows in the last pass_dur
    seconds; None without rows)."""
    folder = Path(folder)
    meta = json.loads((folder / "screen_meta.json").read_text(encoding="utf-8"))
    width, height = meta["screen_width"], meta["screen_height"]
    rows = _sol_rows(folder)
    steps = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
    tolerance = (statistics.median(steps) if steps else 0.0) + 2 * STEP_S
    session = folder.name.split("_sess")[0].removeprefix("VA_")
    out = []
    with open(folder / "trial_events.csv", newline="", encoding="utf-8") as f:
        trials = list(csv.DictReader(f))
    for trial in trials:
        if trial.get("eval_source") != "Sol":
            continue
        start, end = float(trial["start_timestamp"]), float(trial["end_timestamp"])
        in_trial = [r for r in rows if start <= r[0] <= end]
        tracker, newest, i, now, verdict = HoldTracker(pass_dur, max_age, blink_grace), None, 0, start, "OPEN"
        while now <= end + STEP_S:
            while i < len(in_trial) and in_trial[i][0] <= now:
                t, usable, x, y = in_trial[i]
                on_side = 0 <= x < width and 0 <= y < height and (
                    (x >= width // 2) if trial["side"] == "right" else (x < width // 2))
                newest = GazeSample(t, usable, on_side)
                i += 1
            if tracker.update(now, newest):
                break
            if now - start > stim_dur and not tracker.in_progress:
                verdict = "FAIL"
                break
            now += STEP_S
        if tracker.held_s >= pass_dur - tolerance:
            verdict = "PASS"
        final = [r[1] for r in in_trial if r[0] >= end - pass_dur]
        out.append({"session": session, "trial": int(trial["trial_number"]),
                    "trial_type": trial.get("trial_type") or "normal", "was": trial["result"],
                    "now": verdict, "held": tracker.held_s,
                    "valid_final": sum(final) / len(final) if final else None})
    return out


def replay_root(root, **rule):
    """replay_session() over every session folder at or below root."""
    results = []
    for events in sorted(Path(root).rglob("trial_events.csv")):
        folder = events.parent
        if (folder / "sol_gaze_data.csv").exists() and (folder / "screen_meta.json").exists():
            results += replay_session(folder, **rule)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay VA sessions through the current pass rule.")
    ap.add_argument("root", help="a folder of VA session folders (or one session folder)")
    ap.add_argument("--pass-dur", type=float, default=2.0, help="pass duration the sessions ran with (s)")
    ap.add_argument("--stim-dur", type=float, default=5.0, help="stimulus duration they ran with (s)")
    ap.add_argument("--max-age", type=float, default=DEFAULT_MAX_SAMPLE_AGE_S)
    ap.add_argument("--blink-grace", type=float, default=DEFAULT_BLINK_GRACE_S)
    args = ap.parse_args(argv)
    results = replay_root(args.root, pass_dur=args.pass_dur, stim_dur=args.stim_dur,
                          max_age=args.max_age, blink_grace=args.blink_grace)
    changed = [r for r in results if r["was"] != r["now"]]
    passes = sum(r["was"] == "PASS" for r in results)
    count = lambda verdict: sum(r["now"] == verdict for r in changed)
    print(f"{len(results)} Sol-scored trial(s), {passes} recorded PASS. Changed: {len(changed)} "
          f"({count('FAIL')} FAIL, {count('OPEN')} OPEN, {count('PASS')} PASS)")
    print(f"{'session':<12}{'trial':<7}{'type':<8}{'was':<6}{'now':<6}{'held s':<8}valid rows, final {args.pass_dur:g} s")
    for r in changed:
        valid = "-" if r["valid_final"] is None else f"{r['valid_final'] * 100:.0f}%"
        print(f"{r['session']:<12}{r['trial']:<7}{r['trial_type']:<8}{r['was']:<6}{r['now']:<6}"
              f"{r['held']:<8.2f}{valid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
