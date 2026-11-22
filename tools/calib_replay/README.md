# calib_replay: offline calibration replay

A developer tool. It fits calibration models on recorded accuracy tests again and compares them, so a
better Sol calibration can be chosen on data instead of in the field. It only reads the data. It is not
part of the apps and is left out of the exe builds.

**This version is the legacy mode.** It works on what accuracy tests record today: one averaged point
per target (`accuracy_report.json`), and the saved Sol 2D offset profile. Sample-level sessions
(LOGGING_SPEC section 4) are read once the apps write them; see "Not here yet".

## Run

From the repo root, with the repo venv (numpy and OpenCV are enough):

```
venv\Scripts\python.exe -m tools.calib_replay <root> [<root> ...] --out <folder>
```

- `<root>`: a folder of accuracy-test folders, a folder of zips of them, or one zip. Zip members are
  read in memory; nothing is extracted.
- `--out`: where `per_point.csv`, `per_trial.csv` and `summary.md` go. It must be outside this repo and
  outside the data: results from real data never enter the repo.
- `--profiles <folder>`: one calibration folder with `<User Name>_sol_2d_offset.json` files. Needed for
  protocol P1 only.
- `--conditions C2,C4`: only these conditions. `--k 5,9,13` and `--layouts 20`: protocol P3.
  `--select median|mean`: what the hyper-parameters are chosen on. `--boot 2000`, `--seed 0`.

The 35 tests of 2026-09-18 take under a minute.

## What is read from where

- **Subject and condition come from the folder name** (`s01_c3_...`, `A01_S03_...`, `A02_03_...`), not
  from the User Name stored in the file. Where the two disagree, the folder was renamed by hand to the
  right condition and the file keeps what was typed; `summary.md` lists every such run.
- **The profile is found by the User Name stored in the file**, as the program finds it. A profile
  saved after the accuracy test ended is not used: it cannot be the one that test loaded.

## Models

All work in screen pixels on the pairs (gaze, target).

| name | what it does |
|---|---|
| `none` | no correction |
| `idw_shipped` | the program's own `Sol2DOffsetModel` (IDW power 2, snap within 10 px), imported, not rewritten |
| `idw_shrink(a)` | a times the `idw_shipped` offset |
| `const_mean`, `const_median` | one offset for the whole screen |
| `huber_shrink(w)` | w times a Huber estimate of that one offset (c = 1.345) |
| `gain_offset` | per axis: target = a * gaze + b |
| `affine`, `affine_shrunk(lambda)` | target = A gaze + b; with lambda, A is ridge-shrunk toward no scaling and no rotation. lambda is in units of the mean variance of the calibration gaze points |
| `const_plus_idw_residual(a)` | `const_mean`, plus a times the IDW of what it leaves |

Grids: a and w from 0 to 1 in steps of 0.1; lambda 0.03, 0.1, 0.3, 1, 3, 10 (`models.py`).

## Protocols

- **recorded**: the raw and corrected points as the file has them. This reproduces the program's own
  numbers.
- **P1**: fit on the points of the saved profile, test on every point of the accuracy test that looked
  it up. "Does our calibration help?" Needs `--profiles`.
- **P2**: leave one point out within an accuracy test.
- **P3_k**: a random k points of an accuracy test calibrate, the rest test; averaged over the layouts.
- **P4**: a hyper-parameter is chosen for each subject on the other subjects only (lowest mean
  per-trial error over their runs). It needs at least two subjects.

In legacy mode a model is applied to the averaged raw point, not to each sample. For a loaded profile,
`summary.md` shows how far that is from the recorded corrected point ("P1 check").

## Output

- `per_trial.csv`: one row per accuracy test, protocol, model and hyper-parameter value. `selected` = 1
  marks the value chosen by P4 (always 1 for a model without one). Mean and median error in degrees
  with the real screen geometry (`ntuh/sol/replay_math.py`) and with the legacy `px_to_deg`, the share
  of points under 2 and under 5 degrees, and the difference to `none` and to `idw_shipped` in the same
  test.
- `per_point.csv`: every point of the selected rows, for recorded, P1 and P2.
- `summary.md`: the list of accuracy tests, the recorded numbers by condition, and one table per
  protocol with 95 % intervals from a bootstrap over subjects.

## Tests

`tests/test_calib_replay.py`, synthetic data. Two tests reproduce the audit's numbers on real data and
run only when the dataset folder is given:

```
set NTUH_ACC_0918_ROOT=<the 2026-09-18 accuracy dataset folder>
set NTUH_ACC_LAB_ROOT=<the 2026-08-25 lab-pilot folder of accuracy zips>
venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_calib_replay.py
```

If a number does not reproduce, report the difference. Do not tune anything to match.

## Not here yet

These need the sample-level files of LOGGING_SPEC section 4, which the apps do not write yet:

- the session loader (`load.py`) and pairing an accuracy test with its calibration by `profile_id`,
- recomputing screen gaze from raw gaze and the logged homography (`remap.py`),
- per-point centroid estimators other than the recorded mean (`points.py`),
- the noise floor from repeated targets (P5).
