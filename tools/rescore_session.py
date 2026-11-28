# -*- coding: utf-8 -*-
"""Score a session again from its own files, with the scoring code of this checkout
(doc/session_bundle.md, "Rescoring").

    python tools/rescore_session.py <session folder> [--out <folder>]

For when a software fix changes how a result is computed: the sessions already collected are
re-scored instead of collected again. The session folder is only read. The output goes to

    <out>/<session id>/rescore_<app version>.json      (default <out>: _derived next to the session)

and records which program version did the rescoring, the sha256 of every input it read, the new
result, and whether it is identical to the result the session saved.

    va              trial_events.csv   -> the content of va_result.json
    accuracy_test   accuracy_data.csv  -> accuracy_report.json "summary" and meta "result"

Run with the version that recorded the session, the output is identical to what the session
saved. Exit code 0 identical, 1 different (or nothing to compare with), 2 cannot rescore.
Mapping the gaze samples again (another homography or offset) needs bundle v2.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ntuh.recording import session_bundle as sb  # noqa: E402  (stdlib only)


def rescore_va(folder):
    """(inputs read, new result, the result the session saved or None)."""
    from ntuh.vavf.stimuli import VA_RESULT_FILE, build_va_result
    from ntuh.version import get_version
    with open(folder / "trial_events.csv", newline="", encoding="utf-8") as f:
        trials = [(float(row["cpd"]), row["result"] == "PASS", row.get("trial_type") or "normal")
                  for row in csv.DictReader(f)]
    return ["trial_events.csv"], build_va_result(trials, get_version("VA_center_opt")), _saved(folder / VA_RESULT_FILE)


def rescore_accuracy(folder):
    from ntuh.sol import accuracy_test as acc
    with open(folder / "accuracy_data.csv", newline="", encoding="utf-8") as f:
        records = [{key: value if key in ("name", "group") else int(value) if key == "n_samples" else float(value)
                    for key, value in row.items()} for row in csv.DictReader(f)]
    report = _saved(folder / "accuracy_report.json")
    saved = report and {"summary": report.get("summary"), "result": (report.get("meta") or {}).get("result")}
    return ["accuracy_data.csv"], {"summary": acc.summarize(records), "result": acc.result_block(records)}, saved


RESCORERS = {"va": rescore_va, "accuracy_test": rescore_accuracy}
IGNORED_WHEN_COMPARING = ("app_version",)       # says which version wrote the result, not the result


def _saved(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def rescore(folder, out_root=None):
    """Rescore one session folder. Returns (path written, the record written)."""
    folder = Path(folder).resolve()
    meta = _saved(folder / sb.META_FILE)
    if meta is None:
        raise ValueError(f"no readable {sb.META_FILE}: not a session bundle")
    kind = meta.get("session_kind")
    if kind not in RESCORERS:
        raise ValueError(f"nothing to rescore for a {kind!r} session (supported: {', '.join(RESCORERS)})")
    out_root = Path(out_root).resolve() if out_root else folder.parent / "_derived"
    if out_root == folder or folder in out_root.parents:
        raise ValueError("the output folder must be outside the session folder")

    inputs, result, saved = RESCORERS[kind](folder)
    strip = lambda d: {k: v for k, v in d.items() if k not in IGNORED_WHEN_COMPARING}
    software = sb.software_block("VA_center_opt")
    record = {
        "schema": "ntuh.rescore", "schema_version": 1,
        "session_id": meta.get("session_id"), "session_kind": kind,
        "session_app_version": (meta.get("software") or {}).get("app_version"),
        "rescored_at": sb.now_iso(),
        "rescored_with": software,
        "inputs": {name: {"sha256": sb.sha256_file(folder / name)} for name in inputs},
        # None: the session saved no result to compare with (data from before the result files).
        "identical_to_saved": None if saved is None else strip(result) == strip(saved),
        "result": result,
    }
    out = out_root / folder.name / f"rescore_{software['app_version']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, record


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a session again from its own files.")
    ap.add_argument("session", help="a session folder (VA or accuracy test) with session_meta.json")
    ap.add_argument("--out", help="where the derived results go (default: _derived next to the session)")
    args = ap.parse_args(argv)
    try:
        out, record = rescore(args.session, args.out)
    except (OSError, ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    verdict = {True: "IDENTICAL to the saved result", False: "DIFFERENT from the saved result",
               None: "no saved result to compare with"}[record["identical_to_saved"]]
    print(f"{verdict}: {record['session_kind']} {record['session_id']} -> {out}")
    return 0 if record["identical_to_saved"] else 1


if __name__ == "__main__":
    sys.exit(main())
