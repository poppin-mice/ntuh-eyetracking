# CLAUDE.md — ntuh-eyetracking

Guidance for anyone working in this repository — **human contributors and AI coding agents alike**
(e.g. Claude Code used by the NTUH team and by BU). Read this before making changes.

This repo is the **NTUH Eye-Tracking Suite**: three Windows apps (`VA_center_opt`, `calibration`,
`replayer`) for VA/VF screening with Ganzin Sol glasses + webcam. Setup, how to run, and how to build
are in [`README.md`](README.md). This file is the **rules of engagement**.

---

## Development workflow (mandatory)

We use a **feature-branch → pull-request → review → manual-merge** model. There are two long-lived
branches:

- **`main`** — released / stable. Only updated by a release merge. Never commit here directly.
- **`develop`** — integration branch. All work merges here **via reviewed pull requests**, never by a
  direct push.

### Steps for every change

1. **Branch off `develop`:**
   ```
   git switch develop && git pull
   git switch -c feature/<short-topic>      # or fix/<...>, chore/<...>, docs/<...>
   ```
2. **Commit** small, focused changes with clear messages (see [Commits](#commit-conventions)).
3. **Push the branch** and **open a pull request into `develop`** on GitHub
   (`gh pr create --base develop`). Write what changed and why, and how it was verified.
4. **A human reviews the PR on GitHub and merges it by hand** after approval.

### Rules for AI agents (Claude bots) — non-negotiable

- **Never push to `develop` or `main`.** Work only on a `feature/…`, `fix/…`, `chore/…`, or
  `docs/…` branch.
- **Open the PR and stop.** Do **not** merge it — not `gh pr merge`, not the GitHub UI, not a
  fast-forward push. A human reviews and merges.
- **Do not force-push** shared branches or rewrite published history.
- If asked to "merge", interpret it as "open the PR for review" unless a human explicitly confirms
  they are performing/authorising the merge themselves.
- Keep PRs **small and single-purpose** so they are reviewable.

### Recommended best practice (repo settings)

- Enable **branch protection** on `main` and `develop`: require a PR, require ≥1 approving review,
  disallow direct pushes and force-pushes. This makes the rules above enforced, not just documented.
- Prefer **squash-merge** for feature PRs to keep `develop` history linear and readable.
- Delete the feature branch after merge (`gh pr merge --delete-branch`, run **by the human** merger).

## Commit conventions

- **Do NOT add a `Co-Authored-By: Claude …` trailer** to commit messages.
- Present-tense, imperative subject line (≤ ~72 chars); body explains the *why* and how it was
  verified. Reference the app when relevant (e.g. `VA_center_opt: …`).

## Code constraints (repo-specific)

- **Do not edit the vendored `gazefollower/` internals** — it is a vendored upstream library. Wrap or
  extend it from `ntuh/` instead.
- **Sol scene decode runs in an isolated child process on purpose.** The Ganzin SDK's native H.264
  decode can hard-crash; process isolation (`ntuh/sol/sol_child.py` + `preview_client.py`) contains it
  and the parent respawns the child. Do **not** move Sol decoding into the parent, and avoid adding
  per-frame load to the child (it degrades the crash-prone decode). A live-camera tester view was
  tried in VA v1.0.2 and reverted for exactly this reason.
- The three apps stay **import-light at module top level**: heavy imports (pygame/tk/mediapipe/
  gazefollower/SDK) live inside `main()`/functions because `multiprocessing 'spawn'` re-imports the
  entry module in the child.
- Every version-visible change ships with a **version bump** (see below).

## Terminology

- **"Sol glasses"** = Ganzin Sol smart eye-tracking glasses. **"Chronus"** = the paired phone app.
- Use **"gaze estimation"** (not "eye tracking") for the webcam screen-coordinate prediction task.
- **"Simplified / offset calibration"** = the Sol offset-calibration in this repo (1/3/5/9-pt, screen-
  space or camera-space).

## Project data & roles

### Who does what

- **Edan** (GitHub `poppin-mice`): the bridge to NTUH. Triages issues, reviews and merges PRs, and owns
  `ntuh-eyetracking-model` (the model, paper and delivery), which reads this suite's output files.
- **Bouing** (GitHub `bobo2ni`): maintains this repository and does the first analysis and QC of the
  collected data. Apart from onboarding, each task is either **`mode:manual-check`** (checking by hand) or
  **`mode:claude-dev`** (a code change driven with Claude Code from a Claude-dev issue).
- **NTUH colleagues** (收測人員: clinicians / OTs): collect field data with the released exe suite. They
  report problems and requests in the **NTUH tracker sheet**, not on GitHub.

### Codes only: no personal data in this public repository

This repository is **public**. Participants are referred to by **code only** (`s01`, `K03`, `A02` ...) in
PRs, commits, code comments, test fixtures, screenshots and docs here, and in the task issues in the
private QC repo too. Never write a participant's name, birthday, case-record content or a face image here,
and never copy anything from the case records (個案狀況記錄). Test data is synthetic. If you find personal
data in the repo or its history, stop and tell Edan; do not try to rewrite history yourself.

### Task issues (private QC repo)

**Issues are disabled in this public repository.** All task issues (onboarding, `mode:manual-check`,
`mode:claude-dev` and data-QC), the QC scripts and the QC reports live in the **private** repository
`poppin-mice/ntuh-eyetracking-qc`; progress is followed on the GitHub Project board "NTUH Eye-Tracking".
Code, branches and PRs stay here. A PR references its task **cross-repo**: write
`Closes poppin-mice/ntuh-eyetracking-qc#<n>` (a bare `#<n>` would point at this repo). The keyword closes
the task when the PR is merged into `develop`, this repo's default branch.

### Tracker IDs

Every issue and PR carries a **`REQ-xxxx`** ID. That ID is the row in the NTUH tracker sheet, so NTUH's
request and our code change stay linked (use `internal` for work that did not come from the tracker).
The tracker's link is kept out of this public repo. NTUH works from the tracker sheet only; the REQ ID is
what links a sheet row to its task issue.

### Field data (NAS)

- Root: `Z:/ntuh-eyetracking-data` (on the server: `/home/edan/nas/ntuh-eyetracking-data`). It is organised
  by modality first: `VA/`, `VF/`, `Accuracy_Test/`, plus the case records (`個案狀況記錄/`), which
  never enter this repo.
- **Never rename a session folder.** Session folder names are kept byte-for-byte as they are on the NAS
  (including the few that were renamed by hand in the field; the mismatch is recorded, not repaired),
  because the model repo and the QC scripts refer to them by name, and so will each dataset's
  `sessions.csv` manifest.
  Only dataset-level folders may be moved or renamed. A new naming scheme applies to future data only, and
  it comes from a software change, not from renaming files.
- Accuracy-test **condition codes** (field sessions are named `sNN_cK`, e.g. `s01_c3`):
  **C1** practice · **C2** no calibration · **C3** our 5-pt simplified (offset) calibration (簡化校正) ·
  **C4** Sol built-in calibration (Chronus 燈塔 mode) · **C5** both C4 and C3.
- Any change to an output file or column (`VA_output/`, `VF_output/`, `accuracy_test/`) must be listed in
  the PR and in the app's `doc/*.md`. The model repo and the QC scripts parse these files.

### Logging spec

The contract for what each session must record (session metadata, sample-level Sol gaze, homography,
events, calibration profiles) is **`doc/dev/LOGGING_SPEC.md`**. It is a draft (v0.1, 2026-09-27) and is
added by its own PR. Until that PR is merged, ask Edan for the draft. New logging work follows it, and a
format change updates it in the same PR.

### Tests

`python -m pytest` runs the hardware-free tests in `tests/` (settings in `pytest.ini`; the CI workflow
`.github/workflows/tests.yml` runs them on every PR). Add a test when logic can be checked without Sol
glasses, a webcam or a display. Do not write tests that pin a known bug: fixing the bug must not require
editing a test that asserts the wrong behaviour.

## Developer tooling

Optional but recommended for contributors working with AI coding agents. Neither is needed to build
or run the apps.

### RTK — token-optimizing CLI proxy (https://github.com/rtk-ai/rtk)

A single Rust binary that filters/compresses command output before an agent reads it (big token
savings). Install one way, then wire the Claude Code hook:

```
# Windows: download the prebuilt binary from the GitHub Releases page and put it on PATH,
#          or:  cargo install --git https://github.com/rtk-ai/rtk
# macOS/Linux:  brew install rtk   (or the install.sh from the repo)

rtk init -g            # install the Claude Code auto-rewrite hook (then restart Claude Code)
rtk --version          # verify install
rtk init --show        # verify the hook
rtk gain               # token-savings dashboard
```

Config lives at `~/.config/rtk/config.toml`. If `rtk gain` errors, you may have a name-collision with
a different `rtk` on PATH — check `which rtk` / `where rtk`.

### Ponytail — code-minimization plugin (https://github.com/DietrichGebert/ponytail)

A Claude Code plugin that steers agents toward minimal, necessary code (YAGNI / "do we need this?").

```
/plugin marketplace add DietrichGebert/ponytail
/plugin install ponytail@ponytail
```

Usage: `/ponytail [lite|full|ultra|off]` to set intensity, `/ponytail-review` to review a diff for
over-engineering. Default mode via `PONYTAIL_DEFAULT_MODE` or `~/.config/ponytail/config.json`.

## Versioning & release flow

Each app has an **independent** version in `ntuh/version.py` → `APP_VERSIONS`, shown in its window
title. Scheme is `MAJOR.MINOR.PATCH`:

- **PATCH** — bug fix / no user-visible behavior change (a fix that *fills a gap* in an existing
  feature is a PATCH, even if it adds a window/UI).
- **MINOR** — new user-facing feature or option.
- **MAJOR** — breaking change to the workflow or data layout.

### Releasing (scripted local release)

The build needs Windows + the native stack + the vendored Ganzin wheel + produces ~2 GB, so releases
are built **locally** with `release.py` (not in CI).

1. On a `feature/…` (or `release/…`) branch, for each **changed** app:
   - bump its entry in `ntuh/version.py` and add a one-line **changelog** note in that file;
   - add a dated release note `doc/YYYYMMDD_release_note.txt` and list it at the top of
     `MANUAL_DOCS` in `stage_release.py`;
   - update the app's `doc/*.md` if behavior changed.
2. Open a PR into `develop`; review; a human merges.
3. Build + package from `develop`:
   ```
   python packaging/release.py --tag
   ```
   This cleans/builds all three apps, runs `stage_release.py`, writes
   `release/YYYYMMDD_NTUH_EyeTracking_Suite.zip` (date-first, git-ignored), and creates per-app git
   tags `VA_center_opt-vX.Y.Z`, `calibration-vX.Y.Z`, `replayer-vX.Y.Z` for the current versions.
4. **A human** pushes the tags (`git push --tags`) and, for a stable release, opens a PR to merge
   `develop → main`. Hand off the zip from `release/`.

### Release artifact conventions

- Release zips live in the **git-ignored `release/`** folder and are named **date-first**:
  `YYYYMMDD_NTUH_EyeTracking_Suite.zip`. Keep only the **current** build there; remove superseded
  archives.
- Git tags are **per app**: `<app>-vX.Y.Z` (e.g. `VA_center_opt-v1.0.1`).

## Where things live

See [`README.md`](README.md#repository-layout). Quick pointers: app code in `ntuh/` (incl. `ntuh/sol/`,
`ntuh/recording/`); versions in `ntuh/version.py`; all build & release tooling in `packaging/` (`*.spec`,
`build_exe.bat`, `stage_release.py`, `release.py`, `pyinstaller_helpers.py`, `hooks/`); end-user docs in `doc/`.
