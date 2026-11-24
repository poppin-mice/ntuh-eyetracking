"""python -m tools.calib_replay <root> [<root> ...] --out <folder>      (see README.md)"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from . import evaluate, legacy
from .models import MODELS

REPO = Path(__file__).resolve().parents[2]
ID = ("run", "source", "user", "subject", "condition", "loaded")
DIFFS = ("d_none_mean_deg", "d_none_median_deg", "d_idw_mean_deg", "d_idw_median_deg")
TRIAL_COLS = ID + ("protocol", "model", "param", "selected") + evaluate.METRICS + DIFFS
POINT_COLS = ID + ("protocol", "model", "param", "point", "group", "ecc_deg", "target_x", "target_y",
                   "gaze_x", "gaze_y", "pred_x", "pred_y", "err_px", "err_deg", "err_deg_legacy")
PROTOCOLS = {
    "P1": "Fitted on the points of the saved profile, tested on every point of the accuracy test "
          "that looked that profile up.",
    "P2": "Leave one point out: fitted on the other points of the same accuracy test.",
    "P3": "A random k points of the accuracy test calibrate, the other points test.",
}


def _write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _three(rows, col):
    """mean of trial means / mean of trial medians / median of trial medians"""
    mean, med = [r[f"mean_{col}"] for r in rows], [r[f"median_{col}"] for r in rows]
    return f"{np.mean(mean):.2f} / {np.mean(med):.2f} / {np.median(med):.2f}"


def _summary(args, runs, empty, trials, rng):
    sel = [r for r in trials if r["selected"]]
    recorded = {(r["source"], r["run"], r["model"]): r for r in sel if r["protocol"] == "recorded"}
    out = [
        "# Calibration replay, legacy mode", "",
        f"- Accuracy tests: {len(runs)}, points: {sum(len(r['raw']) for r in runs)}, distinct targets: "
        f"{len({p['name'] for r in runs for p in r['points']})}, "
        f"subjects: {len({r['subject'] for r in runs})}.",
        f"- P3: k = {args.k}, {args.layouts} random layouts each. Hyper-parameters chosen on the "
        f"per-trial {args.select}. Bootstrap: {args.boot} resamples of subjects. Seed {args.seed}.",
        "- Errors are degrees of visual angle with the real screen geometry. \"legacy\" is the "
        "px_to_deg the program itself reports.",
        "- Three numbers a / b / c: mean of trial means / mean of trial medians / median of trial "
        "medians. Brackets: 95 % bootstrap interval over subjects.", "",
        "## Accuracy tests", "",
        "Subject and condition are read from the folder name. P1 check: the largest distance, over "
        "the points, between the paired profile applied to the recorded raw point and the recorded "
        "corrected point; a large value means the profile is not the one the test used.", "",
        "| run | User Name in the file | subject | condition | points | aborted | our calibration "
        "loaded | P1 check, px | raw -> corrected mean, legacy deg |", "|" + "---|" * 9,
    ]
    for r in runs:
        raw, corr = (recorded[r["source"], r["run"], m]["mean_deg_legacy"] for m in ("none", "corr"))
        out.append(f"| {r['run']} | {r['user']} | {r['subject']} | {r['condition']} | {len(r['raw'])} | "
                   f"{'yes' if r['meta'].get('aborted') else 'no'} | "
                   f"{'yes' if r['meta'].get('calib_model_loaded') else 'no'} | "
                   f"{r.get('p1_check_px', '')} | {raw:.2f} -> {corr:.2f} |")
    differ = [f"{r['run']} (file: {r['user']})" for r in runs
              if r["run"] != f"{r['user']}_{r['meta'].get('timestamp')}"]     # how the program names it
    out += ["", "Folder name and User Name in the file disagree: " + ("; ".join(differ) or "none") + "."]
    if empty:
        out.append("Left out, no points: " + "; ".join(empty) + ".")

    out += ["", "## As recorded, by condition", "",
            "| condition | runs | our calibration loaded | not loaded and corrected = raw on every point "
            "| raw, legacy deg | corrected, legacy deg |", "|---|---|---|---|---|---|"]
    for cond in sorted({r["condition"] for r in runs}):
        rs = [r for r in runs if r["condition"] == cond]
        unloaded = [r for r in rs if not r["meta"].get("calib_model_loaded")]
        same = sum(np.array_equal(r["raw"], r["corr"]) for r in unloaded)
        cells = (_three([recorded[r["source"], r["run"], m] for r in rs], "deg_legacy")
                 for m in ("none", "corr"))
        out.append(f"| {cond or '?'} | {len(rs)} | {len(rs) - len(unloaded)} | {same} of {len(unloaded)} | "
                   + " | ".join(cells) + " |")

    def ci(rows, col):
        vals, subjects = [r[col] for r in rows], [r["subject"] for r in rows]
        lo, hi = evaluate.boot_ci(vals, subjects, rng, args.boot)
        return f"{np.mean(vals):.2f} [{lo:.2f}, {hi:.2f}]"

    protocols = sorted({r["protocol"] for r in sel} - {"recorded"}, key=lambda p: (p[:2], int(p[4:] or 0)))
    for protocol in protocols:
        out += ["", f"## {protocol}", "", PROTOCOLS[protocol[:2]] + " A difference below 0 "
                "means a smaller error than the baseline, in the same accuracy tests.", "",
                "| model | parameter (runs) | runs | subjects | mean of trial means | mean of trial medians "
                "| points < 2 deg, % | points < 5 deg, % | vs none, trial mean | vs none, trial median "
                "| vs idw_shipped, trial mean | vs idw_shipped, trial median |", "|" + "---|" * 12]
        for model in MODELS:
            rows = [r for r in sel if r["protocol"] == protocol and r["model"] == model]
            if not rows:
                continue
            params = Counter(r["param"] for r in rows if r["param"] is not None)
            out.append(f"| {model} | {', '.join(f'{p:g} ({n})' for p, n in sorted(params.items()))} | "
                       f"{len(rows)} | {len({r['subject'] for r in rows})} | {ci(rows, 'mean_deg')} | "
                       f"{ci(rows, 'median_deg')} | {np.mean([r['pct_lt2'] for r in rows]):.0f} | "
                       f"{np.mean([r['pct_lt5'] for r in rows]):.0f} | "
                       + " | ".join(ci(rows, d) for d in DIFFS) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m tools.calib_replay",
        description="Offline calibration replay, legacy mode: fit calibration models on recorded "
                    "accuracy tests again and compare them. Reads only.")
    ap.add_argument("roots", nargs="+", type=Path,
                    help="folder of accuracy-test folders, folder of zips of them, or one zip")
    ap.add_argument("--out", required=True, type=Path,
                    help="folder for per_point.csv, per_trial.csv and summary.md; "
                         "outside the repo and the data")
    ap.add_argument("--profiles", type=Path,
                    help="calibration folder holding <User Name>_sol_2d_offset.json files (protocol P1)")
    ap.add_argument("--conditions", default="", help="only these conditions, e.g. C2,C4 (default: all)")
    ap.add_argument("--k", default="5,9,13", help="P3: numbers of calibration points (default 5,9,13)")
    ap.add_argument("--layouts", type=int, default=20, help="P3: random layouts per test and k (default 20)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples of subjects (default 2000)")
    ap.add_argument("--select", choices=("median", "mean"), default="median",
                    help="per-trial error the hyper-parameters are chosen on (default median)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = args.out.resolve()
    for keep_out in [REPO, *(r.resolve() for r in args.roots)]:
        if out_dir == keep_out or keep_out in out_dir.parents:
            ap.error(f"--out must be outside {keep_out}")

    runs = [r for root in args.roots for r in legacy.load_runs(root)]
    wanted = {c.strip().upper() for c in args.conditions.split(",") if c.strip()}
    runs = [r for r in runs if not wanted or r["condition"] in wanted]
    empty = [r["run"] for r in runs if not len(r["raw"])]
    runs = [r for r in runs if len(r["raw"])]
    if not runs:
        ap.error("no accuracy test with points found")
    profiles = legacy.load_profiles(args.profiles) if args.profiles else {}
    ks = [int(k) for k in args.k.split(",") if k.strip()]
    rng = np.random.default_rng(args.seed)

    trials, predicted = [], []
    for run in runs:
        ident = {"run": run["run"], "source": run["source"], "user": run["user"], "subject": run["subject"],
                 "condition": run["condition"], "loaded": int(bool(run["meta"].get("calib_model_loaded")))}
        scores, points = evaluate.evaluate_run(run, legacy.profile_for(run, profiles), ks, args.layouts, rng)
        for (protocol, model, param), s in scores.items():
            row = {**ident, "protocol": protocol, "model": model, "param": param,
                   **dict(zip(evaluate.METRICS, s))}
            for tag, baseline in (("none", "none"), ("idw", "idw_shipped")):
                b = scores.get((protocol, baseline, None))
                for m in ("mean_deg", "median_deg"):
                    i = evaluate.METRICS.index(m)
                    row[f"d_{tag}_{m}"] = s[i] - b[i] if b is not None else ""
            trials.append(row)
        if ident["loaded"] and ("P1", "idw_shipped", None) in points:
            gap = np.linalg.norm(points["P1", "idw_shipped", None] - run["corr"], axis=1).max()
            run["p1_check_px"] = f"{gap:.1f}"
        predicted.append((run, ident, points))
    evaluate.mark_selected(trials, f"{args.select}_deg")

    selected = {(r["source"], r["run"], r["protocol"], r["model"], r["param"])
                for r in trials if r["selected"]}
    point_rows = []
    for run, ident, points in predicted:
        for (protocol, model, param), pred in points.items():
            if (run["source"], run["run"], protocol, model, param) not in selected:
                continue
            px, exact, legacy_deg = evaluate.point_errors(pred, run["target"], run["meta"])
            for i, p in enumerate(run["points"]):
                point_rows.append({
                    **ident, "protocol": protocol, "model": model, "param": param, "point": p["name"],
                    "group": p["group"], "ecc_deg": p["ecc_deg"],
                    "target_x": run["target"][i, 0], "target_y": run["target"][i, 1],
                    "gaze_x": run["raw"][i, 0], "gaze_y": run["raw"][i, 1],
                    "pred_x": pred[i, 0], "pred_y": pred[i, 1],
                    "err_px": px[i], "err_deg": exact[i], "err_deg_legacy": legacy_deg[i]})

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_trial.csv", trials, TRIAL_COLS)
    _write_csv(out_dir / "per_point.csv", point_rows, POINT_COLS)
    (out_dir / "summary.md").write_text(_summary(args, runs, empty, trials, rng), encoding="utf-8")
    print(f"{len(runs)} accuracy tests -> {out_dir}")
    return trials


if __name__ == "__main__":
    main()
