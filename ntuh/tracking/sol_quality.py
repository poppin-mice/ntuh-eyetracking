"""Sol/webcam gaze-data quality accounting (thread-safe), plus the small DashboardState
holder. Self-contained logic extracted verbatim from VA_center_opt.py (no behaviour change).
Used identically by the VA and VF test loops.
"""
import threading
import time
from collections import deque


# ---------- Sol Gaze Quality (missing-data-rate over RECEIVED samples) ----------
def sol_sample_is_valid(sample):
    """Return True/False for a Sol GazeData sample's validity, mirroring the recorder's
    `is_valid` definition (combined.gaze_3d.validity). Returns False when undeterminable
    so that undecodable samples count as 'missing'."""
    try:
        return bool(sample.combined.gaze_3d.validity)
    except Exception:
        try:
            return bool(sample.combined.gaze_2d.validity)
        except Exception:
            return False


def _eye_status(sample, attr):
    """EyeStatus of one eye as an int (0 NO_EYE, 1 BLINK, 2 NORMAL); None when not available."""
    try:
        es = getattr(getattr(sample, attr), 'eye_status', None)
        return None if es is None else int(getattr(es, 'value', es))
    except Exception:
        return None


def sol_sample_usable(sample):
    """True when a Sol sample may count as evidence of where the subject looks: the combined gaze
    is valid (sol_sample_is_valid) AND the glasses see at least one eye. A sample with both eyes
    NO_EYE can still carry a gaze point - frozen at its last place - and is not evidence. This is
    the validity the VA pass rule uses (ntuh/vavf/hold_tracker.py)."""
    if not sol_sample_is_valid(sample):
        return False
    return not (_eye_status(sample, 'left_eye') == 0 and _eye_status(sample, 'right_eye') == 0)


def _sol_eye_valid(sample, attr):
    """Per-eye Sol validity: True if that eye's gaze is valid (SDK confidence threshold), else
    fall back to eye_status == NORMAL. attr is 'left_eye' or 'right_eye'."""
    try:
        eye = getattr(sample, attr)
    except Exception:
        return False
    try:
        return bool(eye.gaze.validity)
    except Exception:
        pass
    try:
        es = getattr(eye, 'eye_status', None)
        return int(getattr(es, 'value', es)) == 2  # EyeStatus.NORMAL
    except Exception:
        return False


class _ValidityCounter:
    """Counts valid/total samples overall and within trials (inter-trial excluded)."""
    __slots__ = ('total', 'valid', 'trial_total', 'trial_valid')

    def __init__(self):
        self.total = 0; self.valid = 0; self.trial_total = 0; self.trial_valid = 0

    def add(self, is_valid, in_trial):
        self.total += 1
        if is_valid:
            self.valid += 1
        if in_trial:
            self.trial_total += 1
            if is_valid:
                self.trial_valid += 1

    def overall_pct(self):
        return (self.valid / self.total * 100.0) if self.total else None

    def trial_pct(self):
        return (self.trial_valid / self.trial_total * 100.0) if self.trial_total else None


class SolQualityTracker:
    """Tracks gaze-data quality during the test, channel by channel, with an overall and a
    trial-only (inter-trial excluded) VALIDITY rate per channel:
      Sol    : combined / left eye / right eye  (over RECEIVED samples; Sol may drop packets)
      Webcam : face placement OK / left eye / right eye  (over detected frames)
    Also keeps a rolling real-time window of Sol-combined validity for the live tester gauge.
    Thread-safe: Sol fed from the test loop, webcam fed from the dashboard thread.
    """

    CHANNELS = ('sol_combined', 'sol_left', 'sol_right', 'wc_face', 'wc_left', 'wc_right')

    def __init__(self, window_sec=3.0):
        self.window_sec = float(window_sec) if window_sec and window_sec > 0 else 3.0
        self._lock = threading.Lock()
        self._win = deque()          # (recv_time_s, sol_combined_valid)
        self._wc_win = deque()       # (recv_time_s, webcam_all_valid) - for trial-start gating
        self._in_trial = False
        self.chan = {k: _ValidityCounter() for k in self.CHANNELS}

    def _trim(self, now):
        cutoff = now - self.window_sec
        for w in (self._win, self._wc_win):
            while w and w[0][0] < cutoff:
                w.popleft()

    def set_in_trial(self, flag):
        with self._lock:
            self._in_trial = bool(flag)

    def in_trial(self):
        with self._lock:
            return self._in_trial

    def add_sample(self, sample, now=None):
        """Record one received Sol sample (combined + per-eye validity)."""
        combined = sol_sample_is_valid(sample)
        left = _sol_eye_valid(sample, 'left_eye')
        right = _sol_eye_valid(sample, 'right_eye')
        if now is None:
            now = time.time()
        with self._lock:
            it = self._in_trial
            self.chan['sol_combined'].add(combined, it)
            self.chan['sol_left'].add(left, it)
            self.chan['sol_right'].add(right, it)
            self._win.append((now, combined))
            self._trim(now)

    def add_webcam(self, face_ok, left_ok, right_ok, now=None):
        """Record one webcam detection frame's quality (from the face-quality overlay)."""
        all_valid = bool(face_ok) and bool(left_ok) and bool(right_ok)
        if now is None:
            now = time.time()
        with self._lock:
            it = self._in_trial
            self.chan['wc_face'].add(bool(face_ok), it)
            self.chan['wc_left'].add(bool(left_ok), it)
            self.chan['wc_right'].add(bool(right_ok), it)
            self._wc_win.append((now, all_valid))
            self._trim(now)

    def window_validity(self):
        """Rolling-window VALIDITY % for trial-start gating: returns (sol_pct, webcam_pct), each
        None if no samples are in the window. sol = combined gaze validity; webcam = face AND both
        eyes valid (the same signals shown on the dashboard)."""
        now = time.time()
        with self._lock:
            self._trim(now)
            sol = (sum(1 for (_, v) in self._win if v) / len(self._win) * 100.0) if self._win else None
            wc = (sum(1 for (_, v) in self._wc_win if v) / len(self._wc_win) * 100.0) if self._wc_win else None
            return sol, wc

    def snapshot(self):
        """Sol-combined MISSING-rate snapshot for the live gauge + result screen (back-compatible).
        realtime is None when no samples arrived in the window (=> 'NO SIGNAL')."""
        now = time.time()
        with self._lock:
            self._trim(now)
            n = len(self._win)
            realtime = None
            if n > 0:
                inv = sum(1 for (_, v) in self._win if not v)
                realtime = (inv / n * 100.0, n)
            c = self.chan['sol_combined']
            ov = c.overall_pct(); tr = c.trial_pct()
            return {
                'realtime': realtime,
                'overall': (100.0 - ov) if ov is not None else None,
                'trial': (100.0 - tr) if tr is not None else None,
                'total': c.total, 'invalid': c.total - c.valid,
                'trial_total': c.trial_total, 'trial_invalid': c.trial_total - c.trial_valid,
                'window_sec': self.window_sec,
            }

    def validity_report(self, trial_only=True):
        """Per-channel VALIDITY % (trial-only by default) as {channel: (pct_or_None, n_samples)}."""
        with self._lock:
            out = {}
            for k in self.CHANNELS:
                c = self.chan[k]
                out[k] = ((c.trial_pct() if trial_only else c.overall_pct()),
                          (c.trial_total if trial_only else c.total))
            return out


def build_quality_summary(quality, cfg):
    """Trial-only validity summary for the trackers enabled in cfg. Returns
    (display_lines, json_dict): display_lines = [(label, pct_or_None), ...];
    json_dict maps channel -> {validity_pct, samples}. Only enabled trackers are included."""
    if quality is None:
        return [], {}
    rep = quality.validity_report(trial_only=True)
    cfg = cfg or {}
    groups = []
    if cfg.get('enable_sol'):
        groups += [('sol_combined', 'Sol combined'), ('sol_left', 'Sol left eye'),
                   ('sol_right', 'Sol right eye')]
    if cfg.get('enable_webcam'):
        groups += [('wc_face', 'Webcam face'), ('wc_left', 'Webcam left eye'),
                   ('wc_right', 'Webcam right eye')]
    lines, data = [], {}
    for key, label in groups:
        pct, n = rep.get(key, (None, 0))
        lines.append((label, pct))
        data[key] = {'validity_pct': pct, 'samples': n}
    return lines, data


def build_summary_lines(quality, cfg):
    """End-of-test tester display: one uniform list of (label, pct, kind) for enabled trackers.
    Every value is a VALIDITY percentage (100% = all data valid, 0% = all missing), so all lines
    read the same way and use kind 'valid' (green when high). Sol shows whole-test and trial-only
    combined validity plus per-eye validity; webcam shows face/eye validity. Enabled trackers only."""
    if quality is None:
        return []
    cfg = cfg or {}
    rep = quality.validity_report(trial_only=True)
    snap = quality.snapshot()
    inv = lambda p: ((100.0 - p) if p is not None else None)   # missing% -> validity%
    lines = []
    if cfg.get('enable_sol'):
        lines.append(("Sol valid (whole)", inv(snap['overall']), 'valid'))
        lines.append(("Sol valid (trials)", inv(snap['trial']), 'valid'))
        lines.append(("Sol left eye", rep['sol_left'][0], 'valid'))
        lines.append(("Sol right eye", rep['sol_right'][0], 'valid'))
    if cfg.get('enable_webcam'):
        lines.append(("Webcam face", rep['wc_face'][0], 'valid'))
        lines.append(("Webcam left eye", rep['wc_left'][0], 'valid'))
        lines.append(("Webcam right eye", rep['wc_right'][0], 'valid'))
    return lines


class DashboardState:
    """Small thread-safe holder for live trial info shown on the tester dashboard."""

    def __init__(self):
        self._lock = threading.Lock()
        self._d = {'trial_number': 0, 'cpd': 0.0, 'side': '', 'phase': 'init'}

    def update(self, **kw):
        with self._lock:
            self._d.update(kw)

    def get(self):
        with self._lock:
            return dict(self._d)
