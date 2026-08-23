"""Single source of truth for the release version of the three built apps.

Each app shows its version in the window title (e.g. "Calibration (v1.0.0)") so a
user can tell exactly which build they are running.

Release policy
--------------
- Versions are MAJOR.MINOR.PATCH.
    PATCH  -> bug fix / no behaviour change users would notice
    MINOR  -> new user-facing feature or option
    MAJOR  -> breaking change to the workflow or data layout
- Whenever a program is modified for a release, bump *that program's* entry below
  before building, and add a one-line note to its changelog so we know which
  version a given build corresponds to.

Changelog
---------
VA_center_opt
    1.0.0  Baseline: versioning introduced.
    1.0.1  Fix: the Sol gaze accuracy test gave the operator no way to tell whether the
           subject was fixating the target. Added an operator-only tester-monitor view
           (schematic of the subject screen with the target, the live gaze dot, and the
           target->gaze offset in px/deg) on the configured Tester Screen, with
           focus-independent SPACE/Q. Nothing is drawn on the subject screen (so the dot
           cannot be chased); no change on a single-monitor setup.
    1.1.0  The "wait for stable valid data" gate can now be overridden: pressing SPACE
           (from the subject or the tester window) starts the trial immediately. The
           dashboard banner shows the hint. VF trials are unaffected.
    1.1.1  Fix: connecting to Sol failed with 'No mapping found for keys: frozenset({...})'
           and leaked an aiohttp session, because the phone had moved to remote API 2.0.0
           while the vendored SDK was 1.2.2. Upgraded the vendored wheel to
           ganzin_sol_sdk-2.0.1 (2.x nests replies under .result and no longer matches them
           by exact field set). Connect errors now list every failed init step, not just the
           first. The vendored wheel must match the Chronus app's remote API version.
    1.2.0  The Accuracy Test's operator window now shows the subject's LIVE full-resolution
           Sol scene camera instead of the gray schematic, with the raw and offset-corrected
           gaze drawn in camera space and live accuracy/precision in px and deg. The target
           is not redrawn - the real one is visible on the subject's screen in the video -
           and a line runs from the gaze marker to it. Only when a separate Tester
           Screen is configured. This is the v1.0.2 view that was reverted, redone without its
           cost: the worker sends a plain contiguous frame copy (~1.0 ms) instead of a
           full-frame JPEG encode (~12.0 ms), so the crash-prone decode process is no longer
           starved. Worker errors are now printed instead of silently dropped.
           Test Screen / Tester Screen moved from the Sol Calib tab's "Display Settings" to
           General -> Screen & Viewing: every flow uses them, not just Sol calib. Same saved
           settings keys, so existing settings files keep working.
    1.2.1  Fix: when the Sol scene (front-camera) stream died mid accuracy-test, nothing said so.
           The worker keeps republishing its LAST homography at 15 Hz and gaze keeps streaming, so
           the tester view showed a frozen picture under a live gaze dot and a green "LIVE /
           press SPACE" banner - a 24 s stall went unnoticed for a whole 21-point run, and every
           point taken meanwhile was mapped through a frozen head pose (precision still looks
           healthy, so saved reports cannot reveal it). The worker now times its newest decoded
           frame and ships that age with each homography; the test shows Homography: STALE with a
           red "SCENE VIDEO STALLED <age>" banner, refuses to start a point while stale, and
           discards a point that goes stale mid-collection. Scene-stream subscribe / first-frame /
           silent-end / error are now reported to the parent console (they were prints inside the
           child, which has no console), as are worker crash and respawn. A recovered crash no
           longer counts against MAX_RESPAWNS, so several across a long test can't abort it.
           Also: Connect now verifies the installed Ganzin wheel is 2.x BEFORE opening a socket,
           and names the version and the interpreter if it is not. A stale 1.x wheel used to
           connect and then die on the first reply it could not parse, with
           "'NoneType' object has no attribute 'camera_param'" - which mentions neither the SDK
           nor the version. The phone's remote API version is logged next to the SDK's on every
           connect, and a reachable-but-not-ready phone (glasses detached, Chronus backgrounded)
           now reports its own FAILED message instead of that AttributeError.
    1.2.2  Fix: the app declared only legacy System-DPI awareness, so on a mixed-DPI
           multi-monitor setup DWM stretched the fullscreen window by the ratio between the
           selected screen's scale and the primary's. Stimulus geometry no longer matched the
           physical pixels EnumDisplaySettings reports (targets pushed off the right/bottom
           edge when scaling up, silently mis-sized when scaling down). Now per-monitor
           V2 aware, with a fallback chain for pre-1703 Windows.
calibration
    1.0.0  Baseline: versioning introduced.
    1.0.1  Multi-screen selection, screen-width (cm) input, image size shown in
           cm, flexible/auto-sized config window, configurable profile output
           folder, remembered settings, a webcam preview button (identify cameras
           without calibrating), 'q'-to-quit on the calibration screens, and
           English-keyboard switch/restore with crash recovery (all matching
           VA_center_opt).
    1.1.0  A finished profile is now also copied to the sibling
           VA_center_opt/calibration_profiles/ folder (release layout), so it can be used
           without the manual copy step; the "Done" dialog shows both paths.
    1.1.1  Fix: the calibration target was clipped at the corner points on every screen
           below 4K. gazefollower places the corner targets W*50/1920 x H*50/1080 px from
           the screen edge, so the default 170x170 image lost 35 px per side at 1920x1080
           and ~29% of its area at 1366x768. The requested size is now capped to twice
           that margin for the selected screen (the image is never shifted inwards - its
           center must stay on the calibration coordinate). The cap for the selected screen
           is shown next to the size box and cannot be exceeded: the up arrow stops at it,
           typing a larger number snaps to it, and a too-large value from saved settings is
           reduced as soon as it is seen. Picking a calibration screen sets the size to that
           screen's cap (restoring the remembered screen at startup does not).
           The two "Image size" boxes became one: gazefollower draws the target aspect-fit
           inside the w x h box, so the rendered size was always min(w, h) and editing the
           larger dimension did nothing - e.g. on a 3200x2000 screen the per-axis cap was
           166 x 186 and everything from 166 up looked identical. Default is now 100 px
           (fits 1920x1080 exactly; the old 170 was clipped on every screen below 4K).
           Settings saved by an earlier version carry over as min(width, height) - the size
           that was really being drawn. Same DPI-awareness fix as VA_center_opt
           1.2.2: on a mixed-DPI setup the calibration window was stretched and the
           right/bottom points fell outside the visible screen.
replayer
    1.0.0  Baseline: versioning introduced.
"""

APP_VERSIONS = {
    "VA_center_opt": "1.2.2",
    "calibration": "1.1.1",
    "replayer": "1.0.0",
}


def get_version(app: str) -> str:
    """Return the 'MAJOR.MINOR.PATCH' string for an app key (see APP_VERSIONS)."""
    return APP_VERSIONS.get(app, "0.0.0")
