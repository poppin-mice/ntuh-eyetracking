# Third-party notices

This file lists third-party material that is **in this repository** or **bundled into the release
builds**. Its licensing review is open as tracker item **REQ-0045**, and the decisions belong to the
PI. Nothing listed here has been removed or relicensed. This repository has no licence file of its
own yet. Choosing one, and deciding whether the repository stays public, are part of REQ-0045.

---

## 1. GazeFollower (vendored in `gazefollower/`)

| | |
|---|---|
| Upstream | https://github.com/GanchengZhu/GazeFollower |
| Author | Gancheng Zhu (`gazefollower/version.py`: `Copyright (c) 2023-2024 Gancheng Zhu`) |
| Version | **1.0.0** (`gazefollower/version.py`) |
| Upstream snapshot | This repository's history descends from upstream commit `d8c1969` (2025-05-06). That is upstream tag `v1.0.0` (`fa368ba`, 2025-03-25) plus a README/example-only change. |
| Licence | **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** |
| Licence text | [`gazefollower/LICENSE`](gazefollower/LICENSE) |

**Licence file.** `gazefollower/LICENSE` is a byte-for-byte copy of upstream's `LICENSE-CC-BY-NC-SA`
at tag `v1.0.0` (git blob `bfef380`, the same blob as at `d8c1969` and at upstream HEAD). This
repository carried that file until commit `01f184e` (2025-08-12) deleted it. It has been restored
here. Upstream's README said CC BY 4.0 for one week (commit `ec5442e`, 2024-11-14). Upstream's
first licence file (`f5aa84a`, 2024-11-21) is CC BY-NC-SA 4.0, and so is every later one. The
vendored snapshot is from 2025, so **CC BY-NC-SA 4.0 is the licence that applies to it.**

**Conflicting metadata string.** `gazefollower/version.py` still declares
`__license__ = "Creative Commons Attribution 4.0 (CC BY 4.0)"`. The string dates from the upstream
commit that added `version.py` (`5df50f9`, 2024-11-05) and is unchanged at `d8c1969` and at upstream
HEAD (v1.1.0). It is left
as is here, because the vendored `gazefollower/` internals are not edited. The licence file and
upstream's README both say CC BY-NC-SA 4.0. Whether to ask upstream to fix the string is part of
REQ-0045.

**Local modifications (CC BY-NC-SA asks for changes to be indicated).** Relative to upstream
`d8c1969`, this repository changed or added the following files. `git diff d8c1969 -- gazefollower/`
shows the full diff:

- `gazefollower/GazeFollower.py`
- `gazefollower/calibration/CalibrationController.py`
- `gazefollower/calibration/SVRCalibration.py`
- `gazefollower/camera/WebCamCamera.py`
- `gazefollower/misc/DefaultConfig.py`
- `gazefollower/ui/CalibrationUI.py`
- `gazefollower/res/image/妙蛙種子.png` (added, see section 2.3)

The model weights (`gazefollower/res/model_weights/base.mnn`) and the other `gazefollower/res/` files
are unchanged upstream files. PyInstaller bundles `gazefollower/`, including this licence file,
into the `VA_center_opt` and `calibration` builds (`packaging/*.spec`).

**To be confirmed by the PI (REQ-0045):**

- **NonCommercial:** whether the current and any planned use of the suite (research
  screening at NTUH, any later deployment) counts as non-commercial under CC BY-NC-SA 4.0.
- **ShareAlike:** what the ShareAlike term requires for the modified `gazefollower/` and for
  the builds that bundle it.
- **Attribution in the builds:** whether the release zip / manual needs an attribution notice
  beyond the bundled licence file.

---

## 2. Items under review (REQ-0045) - kept in the repository for now

None of these has a recorded source or licence in the repository. They stay in place until the PI
decides. The options are to keep them, replace them, move them to a private location, or make the
repository private.

### 2.1 Study documents

| Path | Size | Added | Note |
|---|---|---|---|
| `doc/data_collection_guide/20260720/20260720_資料收集操作手冊.docx` | 6.9 MB | `0a632aa` (2026-07-20) | Data-collection guide / study protocol for NTUH. It is internal study material, so check whether it should be public. |

### 2.2 Ganzin material

| Path | Size | Added | Note |
|---|---|---|---|
| `doc/data_collection_guide/Chronus_ch.pdf` | 5.8 MB | `6315ba1` (2026-08-10) | Ganzin Chronus app manual (Chinese). The text "Confidential" appears on **39 of its 41 pages**. |
| `doc/data_collection_guide/20260720/figures/chronus_p*.png` (12 files) | - | `0a632aa` (2026-07-20) | Page renders of the Chronus manual. The page checked (`chronus_p05.png`) carries a "Ganzin Confidential" footer. |
| `vendor/ganzin_sol_sdk-2.0.1-py3-none-any.whl` | 64.6 KB | `05d7799` (2026-08-08) | Ganzin Sol SDK 2.0.1. The wheel metadata declares `License :: OSI Approved :: MIT License`, but the wheel contains no licence file. Confirm redistribution terms with Ganzin. |

### 2.3 Images

| Path | Note |
|---|---|
| `calibration_images/bulbasaur.png`, `charizard.png`, `pikachu.png`, `squirtle.png` | The file names indicate Pokémon characters (third-party IP). Source and licence are not recorded. PyInstaller bundles them into the builds. |
| `stimulus_images/Pikachu01.jpg`, `Pikachu02.png`, `Pikachu03.jpg`, `pikachu.png` | Same as above. |
| `gazefollower/res/image/妙蛙種子.png` | Bulbasaur. Added in this repository, not an upstream GazeFollower file. |
| `stimulus_images/ball.jpg`, `ball.webp` | Source and licence are not recorded. |

---

## 3. Python dependencies bundled into the release builds (not yet inventoried)

The release builds bundle the packages from `requirements.txt` through PyInstaller, and each package
keeps its own licence. They have not been inventoried yet. Two are flagged for the REQ-0045 review,
based on the licence metadata of the installed packages:

- **PyQt6 6.11.0** declares `License-Expression: GPL-3.0-only`. `packaging/replayer.spec` bundles
  it with `collect_all('PyQt6')`.
- **pygame 2.6.1** declares `License: LGPL`.
