# -*- coding: utf-8 -*-
"""Recompute an accuracy test from its own folder (bundle v2, doc/dev/LOGGING_SPEC.md 8).

    python tools/check_accuracy_replay.py <accuracy session folder>

R1  raw_screen in sol_gaze_data.csv = raw gaze through the logged homography (homography_log.csv)
R2  corr_screen = the applied profile (calib_profile_snapshot.json) on raw_screen; corr = raw
    when no profile was applied
R3  the headline of points.csv / accuracy_data.csv (raw_x/y, corr_x/y: mean of the 30 render-frame
    samples) = recomputed from the samples and homographies collect_end lists in events.csv

Each check passes when no value differs by more than 1e-6 px. Exit code 0 when all three pass,
1 when one fails, 2 when the folder is not a bundle v2 accuracy session. Only reads the folder;
uses the program's own mapping and offset code.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntuh.sol import accuracy_test as acc  # noqa: E402  (numpy only)

TOL_PX = 1e-6
SNAPSHOT = "calib_profile_snapshot.json"


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _xy(row, prefix):
    x, y = row[f"{prefix}_x"], row[f"{prefix}_y"]
    return None if x == "" or y == "" else (float(x), float(y))


def _diff(a, b):
    if a is None or b is None:
        return 0.0 if a is None and b is None else math.inf
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def load(folder):
    """Everything R1-R3 read from a session folder. Raises OSError / KeyError on a folder that
    lacks the bundle v2 files."""
    folder = Path(folder)
    model = None
    if (folder / SNAPSHOT).is_file():   # written only when the test applied a profile
        from ntuh.sol.offset_calibration_2d import Sol2DOffsetModel
        data = json.loads((folder / SNAPSHOT).read_text(encoding="utf-8"))
        model = Sol2DOffsetModel.from_dict(data.get("model"))
    homs = {}
    for r in _read(folder / "homography_log.csv"):
        homs[int(r["h_id"])] = (None if r["h00"] == "" else
                                [[float(r[f"h{i}{j}"]) for j in range(3)] for i in range(3)])
    return {"gaze": _read(folder / "sol_gaze_data.csv"), "homs": homs, "model": model,
            "ends": [e for e in _read(folder / "events.csv") if e["event"] == "collect_end"],
            "points": _read(folder / "points.csv"), "legacy": _read(folder / "accuracy_data.csv")
            if (folder / "accuracy_data.csv").is_file() else []}


def _result(name, compared, max_diff, note=""):
    return {"check": name, "compared": compared, "max_diff_px": max_diff,
            "ok": max_diff <= TOL_PX, "note": note}


def check_r1(s):
    """raw_screen = raw gaze through the homography of its h_id. Rows without a homography are
    counted, not compared."""
    n = skipped = 0
    worst = 0.0
    for r in s["gaze"]:
        H = s["homs"].get(int(r["h_id"])) if r["h_id"] != "" else None
        if H is None:
            skipped += 1
            worst = max(worst, _diff(None, _xy(r, "raw_screen")))   # no H must mean no raw_screen
            continue
        n += 1
        raw = acc._apply_homography(H, (float(r["raw_gaze_x"]), float(r["raw_gaze_y"])))
        worst = max(worst, _diff(raw, _xy(r, "raw_screen")))
    return _result("R1", n, worst, f"{skipped} row(s) without a homography")


def check_r2(s):
    """corr_screen = the snapshot profile applied to raw_screen (screen-mode profile), or to the
    raw gaze before the homography (legacy camera-mode profile); = raw_screen without a profile."""
    model, n, worst = s["model"], 0, 0.0
    for r in s["gaze"]:
        raw = _xy(r, "raw_screen")
        if raw is None:
            continue
        n += 1
        if model is None:
            expect = raw
        elif getattr(model, "offset_mode", "") == "screen":
            expect = tuple(model.predict(raw))
        else:
            expect = acc.map_raw_and_corrected(s["homs"][int(r["h_id"])],
                                               (float(r["raw_gaze_x"]), float(r["raw_gaze_y"])), model)[1]
        worst = max(worst, _diff(expect, _xy(r, "corr_screen")))
    return _result("R2", n, worst, "profile applied" if model else "no profile: corr must equal raw")


def check_r3(s):
    """Each point of points.csv (and the same row of accuracy_data.csv) from the render-frame
    samples its collect_end event lists, through the program's own compute_point_record."""
    if not s["points"]:
        return _result("R3", 0, 0.0, "no points recorded: nothing to compare")
    gaze = {int(r["sample_idx"]): (float(r["raw_gaze_x"]), float(r["raw_gaze_y"])) for r in s["gaze"]}
    ends = {e["target_idx"]: json.loads(e["detail_json"]) for e in s["ends"]}   # last one per target
    worst = 0.0
    for i, p in enumerate(s["points"]):
        end = ends.get(p["target_idx"])
        if end is None:
            worst = math.inf
            continue
        mapped = [acc.map_raw_and_corrected(s["homs"].get(h), gaze.get(si), s["model"])
                  if gaze.get(si) is not None else (None, None) for si, h in end["render_samples"]]
        if any(m[0] is None for m in mapped):
            worst = math.inf
            continue
        target = {"name": p["name"], "group": p["group"], "ecc_deg": float(p["ecc_deg"]),
                  "x": float(p["target_x"]), "y": float(p["target_y"])}
        rec = acc.compute_point_record(target, [m[0] for m in mapped], [m[1] for m in mapped],
                                       1.0, 1, 1.0)      # degrees are not compared
        for row in [p] + ([s["legacy"][i]] if i < len(s["legacy"]) else []):
            for key in ("raw", "corr"):
                worst = max(worst, _diff((rec[f"{key}_x"], rec[f"{key}_y"]), _xy(row, key)))
    if len(s["legacy"]) != len(s["points"]):
        worst = math.inf
    return _result("R3", len(s["points"]), worst,
                   f"{len(s['points'])} point(s) vs points.csv and accuracy_data.csv")


def check_folder(folder):
    s = load(folder)
    return [check_r1(s), check_r2(s), check_r3(s)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Recompute an accuracy test from its own folder (R1-R3).")
    ap.add_argument("folder", help="one accuracy_test/<session> folder written by VA_center_opt >= 1.12.0")
    args = ap.parse_args(argv)
    try:
        results = check_folder(args.folder)
    except (OSError, KeyError, ValueError) as e:
        print(f"ERROR: {args.folder} is not a bundle v2 accuracy session: {e}", file=sys.stderr)
        return 2
    for r in results:
        print(f"{r['check']}  {'PASS' if r['ok'] else 'FAIL'}  compared {r['compared']:>6}  "
              f"max |diff| {r['max_diff_px']:.3g} px  ({r['note']})")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
