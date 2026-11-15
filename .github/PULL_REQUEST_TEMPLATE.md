> **PUBLIC repository.** Participant **codes only** (s01, K03, A02 ...). No names, birthdays, case-record
> contents, face screenshots or raw data files in this PR, its commits or test fixtures.

## What / why

Closes poppin-mice/ntuh-eyetracking-qc#<issue>   ·   Tracker: REQ-____   ·   Mode: `mode:claude-dev` / `mode:manual-check` / Edan

Tasks are tracked in the private repo `poppin-mice/ntuh-eyetracking-qc` and on the "NTUH Eye-Tracking" Project board (Issues are off here); keep the full cross-repo form above, a bare `#N` points at this repo.

<one paragraph: what changed and why>

## Version bump (see CLAUDE.md, "Versioning & release flow")

- [ ] `ntuh/version.py` bumped for each changed app (PATCH fix / MINOR feature / MAJOR workflow or data-layout break) + one-line changelog
- [ ] Release-note line drafted for the next `doc/YYYYMMDD_release_note.txt`
- [ ] No bump: tests / CI / docs / repo hygiene only, no user-visible behaviour change (say why below)

## Data / output impact

- [ ] None
- [ ] Changes files or columns in `VA_output/`, `VF_output/` or `accuracy_test/` -> described below, `doc/<app>.md` updated, QC scripts and model-repo loaders to be told

## How verified

- [ ] `python -m pytest` passes locally (CI `tests` workflow green)
- [ ] Single monitor 1920x1080 @100%
- [ ] Dual monitor, mixed DPI
- [ ] Sol connected (if relevant)
- [ ] Output opened in replayer

<paste console output / screenshots - NO participant faces or names>

## Reviewer

- [ ] Scope matches the issue; PR is single-purpose
- [ ] Codes only - no names or face images anywhere in the diff, commits or description
- [ ] Squash-merge, delete the branch
- [ ] Tracker row updated with the PR link and the new status
