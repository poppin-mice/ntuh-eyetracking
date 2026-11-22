# -*- coding: utf-8 -*-
"""The VA pass rule: the gaze must stay on the target's side for `pass_dur` seconds, and only
VALID, FRESH gaze samples count as evidence. Pure: no pygame, no SDK types, no files.

Until 1.9.x the test took the last mapped gaze point every frame, whether or not the sample
behind it was valid, and kept using it when no new sample arrived. A frozen point on the
target's side then passed a trial with no eye detected at all (rule version 1).

Rule version 2 (HoldTracker), checked once per rendered frame against the newest gaze sample:

    counts      the sample is valid, on the target's side and at most max_sample_age_s old
    looked away the sample is valid and fresh but NOT on the target's side: the hold starts over
    no gaze     the sample is invalid, too old, or there is none: the timer pauses; once
                blink_grace_s has passed since the last sample that counted, the hold starts over

PASS when the time that counted reaches pass_dur. Both thresholds are measured from the sample,
not from the check: the time between two checks counts only as far as the last sample vouches for
it (max_sample_age_s), and a good sample that comes blink_grace_s or more after the last one begins
a new hold. So a loop that stalls, or checks late, cannot add up time nobody looked at. The two
thresholds are settings; their defaults are the suggestion of the 2026-09-18 audit and are still
to be confirmed (REQ-0027).
"""
from collections import namedtuple

PASS_RULE_VERSION = "2"
DEFAULT_MAX_SAMPLE_AGE_S = 0.2
DEFAULT_BLINK_GRACE_S = 0.3
QUALITY_GAP_S = 0.3             # the "gap" counted in trial_quality.csv (n_gaps_gt_300ms)

# t: when the program received the sample (seconds, the clock `now` is on). valid: usable as
# evidence (see ntuh.tracking.sol_quality.sol_sample_usable, and it could be mapped to the
# screen). in_correct_half: it maps to the target's half of the screen.
GazeSample = namedtuple("GazeSample", "t valid in_correct_half")

TRIAL_QUALITY_COLUMNS = ["trial_number", "n_sol_samples", "valid_fraction_trial",
                         "valid_fraction_final_window", "hold_start_s", "max_sample_age_ms",
                         "n_gaps_gt_300ms", "pass_rule_version"]


class HoldTracker:
    """One trial's hold. Call update(now, newest sample) every frame; read passed, held_s,
    hold_start and in_progress."""

    def __init__(self, pass_dur, max_sample_age_s=DEFAULT_MAX_SAMPLE_AGE_S,
                 blink_grace_s=DEFAULT_BLINK_GRACE_S):
        self.pass_dur = float(pass_dur)
        self.max_sample_age_s = float(max_sample_age_s)
        self.blink_grace_s = float(blink_grace_s)
        self.passed = False
        self.hold_start = None      # `now` at which the hold in progress began
        self.held_s = 0.0           # seconds that have counted toward pass_dur in that hold
        self._counted_at = None     # `now` of the previous update, if the gaze counted then
        self._last_good_t = None    # receive time of the last sample that counted

    @property
    def in_progress(self):
        """A hold has begun and has not started over (it may be paused by a blink)."""
        return self.hold_start is not None

    def _start_over(self):
        self.hold_start = self._counted_at = self._last_good_t = None
        self.held_s = 0.0

    def update(self, now, sample):
        """Take the newest gaze sample (or None) at time `now`. Returns True once passed."""
        if self.passed:
            return True
        fresh = (sample is not None and sample.valid
                 and now - sample.t <= self.max_sample_age_s)
        counts = fresh and sample.in_correct_half
        if fresh and not counts:                            # looked away
            self._start_over()
        elif self.hold_start is not None:
            # A sample that counted vouches for max_sample_age_s. Past that the hold is without
            # gaze, and so it is whenever the newest sample does not count.
            without = now - self._last_good_t
            if (not counts or without > self.max_sample_age_s) and without >= self.blink_grace_s:
                self._start_over()
        if counts:
            if self.hold_start is None:
                self.hold_start = now
            elif self._counted_at is not None:              # it counted at the last check too
                self.held_s += min(now, self._last_good_t + self.max_sample_age_s) - self._counted_at
            self._counted_at, self._last_good_t = now, sample.t
            self.passed = self.held_s + 1e-9 >= self.pass_dur
        else:
            self._counted_at = None                         # no gaze: the timer pauses
        return self.passed


def trial_quality_row(trial_number, samples, t_start, t_end, final_window_s, hold_start=None):
    """One row of trial_quality.csv (TRIAL_QUALITY_COLUMNS).

    samples: (receive time, valid) of EVERY Sol sample received during the trial, in order.
    The final window is the last final_window_s seconds of the trial (the pass duration: the
    stretch a PASS was decided on). max_sample_age_ms is the longest the trial went without a
    new sample; n_gaps_gt_300ms counts the stretches longer than 0.3 s without a valid one.
    Both include the time before the first and after the last sample. hold_start is the `now`
    at which the hold that passed began (None for a FAIL).
    """
    valid_times = [t for t, valid in samples if valid]
    final = [valid for t, valid in samples if t >= t_end - final_window_s]

    def longest_and_count(times, threshold):
        edges = [t_start, *times, t_end]
        gaps = [b - a for a, b in zip(edges, edges[1:])]
        return max(gaps), sum(gap > threshold for gap in gaps)

    fraction = lambda flags: round(sum(flags) / len(flags), 4) if flags else ""
    return {
        "trial_number": trial_number,
        "n_sol_samples": len(samples),
        "valid_fraction_trial": fraction([valid for _, valid in samples]),
        "valid_fraction_final_window": fraction(final),
        "hold_start_s": "" if hold_start is None else round(hold_start - t_start, 3),
        "max_sample_age_ms": round(longest_and_count([t for t, _ in samples], 0)[0] * 1000),
        "n_gaps_gt_300ms": longest_and_count(valid_times, QUALITY_GAP_S)[1],
        "pass_rule_version": PASS_RULE_VERSION,
    }
