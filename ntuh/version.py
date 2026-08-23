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
           Fix: Test Screen / Tester Screen / Sol preview screen were restored from the
           settings file without checking the display still exists. After unplugging,
           renaming or re-resolutioning a monitor the picker showed a screen that was gone
           while the ':'-split index parsers resolved it to whatever monitor now sits at
           that index - so a test could run on a screen the operator never chose. All three
           are now re-resolved against the connected monitors (shared with the calibration
           app's picker), matched on the saved display's INDEX rather than on the whole
           saved label: the label also carries the monitor name and resolution, both of
           which Windows can report differently after a resolution or aspect-ratio change,
           and either used to invalidate the match and move the selection to another
           display. The choice is now just re-labelled, and only a display that is really
           gone falls back to the first screen.
    1.3.0  A "Font" spinner on the settings window's button bar (always reachable, whichever
           tab is open), applied live and remembered between runs. The settings text is small
           on the high-resolution clinic screens, more so now that the app is per-monitor DPI
           aware and Windows no longer stretches the window. The ttk styles and every
           per-widget font= now share five live font objects, so resizing those restyles the
           window with no rebuild; the Tk named fonts move too, because the vista theme
           ignores a style-level -font for Entry/Combobox/Spinbox and about half of those
           widgets here carry no explicit font. 15 hardcoded ("Arial", 9/10/12) sizes are
           gone with it.
           The Sol tab's "Preview Gaze Mapping -> Screen" picker is gone: the gaze preview
           and the Accuracy Test now run on General -> Screen & Viewing -> Test Screen, the
           screen the subject is actually tested on. It was a second, separately remembered
           copy of the same setting that silently defaulted to screen 0, so a preview or an
           accuracy test could run on a different monitor than the test itself. The saved
           'sol_preview_screen' key is dropped and ignored.
           General -> Screen & Viewing: "Test Screen" is renamed "Subject Screen" and
           "Tester Screen" is renamed "Examiner Screen", and the examiner one is now listed
           first. Labels only - the saved keys (sol_offset_user_screen /
           sol_offset_tester_screen) are unchanged, so existing settings files load as they
           did.
           Fix: with only one display connected, operator-only windows still opened - on the
           subject's screen, on top of the stimulus. The Sol 2D-calibration monitoring window
           was never gated at all, and the VA/VF tester dashboard placed itself via a rect
           that clamped an unusable examiner index to monitor 0. resolve_tester_rect now
           returns None whenever there is no separate examiner screen (one display, examiner
           == subject, or a stale index) and every examiner view honours that; the Examiner
           Screen picker is disabled on a single display, without touching the saved value so
           the choice survives until a second display is attached. The dashboard object still
           RUNS in that case and only draws no window: it is the only sampler of webcam
           validity (sol_quality.add_webcam), which the "wait for stable valid data" gate and
           the end-of-test quality metrics both read. Stopping it outright left the gate
           waiting forever, so every trial had to be force-started with SPACE.
    1.4.0  Negative Sample Collection Mode (VA only, OFF by default): inserts N "catch"
           trials at random positions among the normal trials. A catch trial renders the
           grating too fine to resolve, so the subject has no findable target and the gaze in
           that window is a negative ("non-target fixation") sample for the ML pipeline. The
           trial looks and scores like a normal one to the subject but never reaches the
           staircase or the VA score; trial_events.csv gains a trial_type column
           (normal/catch) that the pipeline labels from. The catch frequency is derived, not
           configured - the highest the display renders without aliasing (0.4 cycles/px,
           under the 0.5 Nyquist limit) - and the mode refuses to run when that is not above
           the 20 cpd staircase ceiling (a 1080p screen at 50 cm manages only 13.8 cpd) so it
           cannot silently collect mislabelled samples. Catch trials still owed when the
           staircase ends early are forced in before completion, and one that raises consumes
           its slot rather than looping forever.
           Removed the per-test VA_<user>_opt.csv summary: it duplicated trial_events.csv in a
           less useful form (no timestamps, no trial_type) and was the only reader of the
           in-memory `results` list, which is gone with it.
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
           Also fixed: the saved screen was restored by index only, so after a monitor was
           unplugged, renamed or re-resolutioned the picker showed a display that no longer
           existed while the calibration ran on whatever monitor now sat at that index.
    1.2.0  A "GUI font size" spinner on the settings window, applied live as it changes and
           remembered between runs - the settings text is small on the high-resolution
           clinic screens (more so now that the app is per-monitor DPI aware and Windows no
           longer stretches the window). It retunes Tk's named fonts, so the whole window
           follows at once.
replayer
    1.0.0  Baseline: versioning introduced.
    1.1.0  A "Font" spinner in the menu-bar corner, applied live and remembered between runs
           (QSettings), matching the control the other two apps gained. Qt needed a different
           mechanism than their Tk one: QApplication.setFont only reaches widgets created
           after the call, so each live widget is set explicitly, and the six hardcoded
           `font-size: 11/12px` stylesheet rules had to go first because a stylesheet beats
           the widget font and pinned the list, status bar and hint labels to a fixed size.
           The spinner sits at the bottom-left of the status bar; the status text became a
           label because statusBar().showMessage() hides every widget added with addWidget(),
           which would have made the spinner vanish on the first session load.
           Each video block's title badge ("Screen"/"Webcam"/"Sol") and its "No <stream>"
           placeholder are painted from the widget font instead of a hardcoded QFont, so they
           follow the control too; the badge box is measured from that font rather than a
           fixed 22px. Fixed alongside: the badge width was computed from the painter's
           PREVIOUS font because fontMetrics() was read before setFont().
           The timeline's validity strips, their SOL/CAM labels and the elapsed-time text are
           derived from the font too, and the widget's height with them (was a hardcoded 84px
           holding 10px strips and 7pt labels), so the labels grow without being clipped by
           their strip.
           The config panel moved into a QScrollArea: its five group boxes grow with the font
           and used to push the window's minimum height past the screen - at 16pt and above,
           a maximized window pushed the status bar off the bottom. Worst-case window minimum
           across 7-20pt is now 419px instead of 956px. The panel's setFixedWidth(260) went
           with it - pinned, it could neither shrink to the viewport (the vertical scrollbar
           left 245px, so a horizontal one appeared) nor widen when the splitter was dragged
           out. Width now comes from the splitter, floored so it cannot be dragged narrower
           than the content.
           Gaze-point labels keep their own size - they are drawn on the video canvas in video
           coordinates and scale with it, not with the UI chrome.
"""

APP_VERSIONS = {
    "VA_center_opt": "1.4.0",
    "calibration": "1.2.0",
    "replayer": "1.1.0",
}


def get_version(app: str) -> str:
    """Return the 'MAJOR.MINOR.PATCH' string for an app key (see APP_VERSIONS)."""
    return APP_VERSIONS.get(app, "0.0.0")
