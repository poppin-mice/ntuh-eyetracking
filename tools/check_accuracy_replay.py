# -*- coding: utf-8 -*-
"""Recompute an accuracy test from its own folder (bundle v2, doc/dev/LOGGING_SPEC.md 8).

    python tools/check_accuracy_replay.py <accuracy session folder>

The checks R1-R3 live in ntuh/sol/accuracy_replay.py (the replayer uses them too). Exit code 0
when all three pass, 1 when one fails, 2 when the folder is not a bundle v2 accuracy session.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntuh.sol.accuracy_replay import (  # noqa: E402,F401  (re-exported for callers of this tool)
    TOL_PX, check_folder, check_r1, check_r2, check_r3, load, recompute_points)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Recompute an accuracy test from its own folder (R1-R3).")
    ap.add_argument("folder", help="one accuracy_test/<session> folder written by VA_center_opt >= 1.12.0")
    args = ap.parse_args(argv)
    try:
        if not (Path(args.folder) / "points.csv").is_file():
            raise OSError("no points.csv")
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
