# -*- coding: utf-8 -*-
"""Synthetic test data shared by the hardware-free tests. Nothing here is field data.

    from helpers.synthetic import parse_outcomes, make_sol_stream

parse_outcomes   "P2 P4 F6" -> [(2.0, True), (4.0, True), (6.0, False)]
make_sol_stream  Sol samples with their receive times, with stream gaps and invalid spans
"""
from ntuh.sol import sol_ipc

NO_EYE, BLINK, NORMAL = 0, 1, 2     # EyeStatus values as the Sol child sends them (sol_ipc.py)


def parse_outcomes(seq):
    """Staircase outcome sequence -> [(cpd, passed), ...].

    Each token is P (pass) or F (fail) followed by the level in cpd the trial was shown at:
    "P2 P4 F6" -> [(2.0, True), (4.0, True), (6.0, False)]. A malformed token raises ValueError,
    so a typo in a test sequence cannot turn into a silently different sequence.
    """
    out = []
    for tok in seq.split():
        if tok[0] not in "PF":
            raise ValueError(f"bad outcome {tok!r}: want P<cpd> or F<cpd>")
        out.append((float(tok[1:]), tok[0] == "P"))
    return out


def _within(t, spans):
    return any(t0 <= t < t1 for t0, t1 in spans)


def _eye(valid):
    return {
        "confidence": 1.0 if valid else 0.0,
        "eye_status": NORMAL if valid else NO_EYE,
        "gaze": {"validity": valid,
                 "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
                 "direction": {"x": 0.0, "y": 0.0, "z": 1.0}},
        "pupil3d": {"validity": 1.0 if valid else 0.0, "diameter": 4.0 if valid else 0.0},
        "pupil2d": {"x": 0.0, "y": 0.0},
    }


def make_sol_stream(duration_s, rate_hz, gaps=(), invalid=(), xy=(640.0, 360.0)):
    """Synthetic Sol gaze stream: [(recv_time_s, sample), ...] in time order.

    One sample every 1/rate_hz starting at t = 0 (t = duration_s itself is not included).
    `sample` has the shape SolPreviewClient.drain_gaze() hands the parent
    (sample.combined.gaze_2d.x, sample.left_eye.eye_status, ...). The receive time is returned
    beside it because the parent stamps it on arrival; it is not a field of the sample.

    gaps     [(t0, t1), ...]  no sample at all while t0 <= t < t1 (the stream is lost).
    invalid  [(t0, t1), ...]  samples still arrive while t0 <= t < t1 but are invalid: combined
                              validity False, both eyes NO_EYE with confidence 0. They keep
                              carrying x/y, so a test can put invalid gaze on the target.
    xy       scene-camera gaze_2d (x, y) in px: one constant point, or a function of t.
    """
    stream = []
    for i in range(round(duration_s * rate_hz)):
        t = i / rate_hz
        if _within(t, gaps):
            continue
        valid = not _within(t, invalid)
        x, y = xy(t) if callable(xy) else xy
        stream.append((t, sol_ipc.reconstruct({
            "combined": {
                "gaze_2d": {"validity": valid, "x": float(x), "y": float(y)},
                "gaze_3d": {"validity": valid, "x": 0.0, "y": 0.0, "z": 1.0},
            },
            "left_eye": _eye(valid),
            "right_eye": _eye(valid),
            "timestamp": round(t * 1000),   # synthetic device clock (ms since stream start)
        })))
    return stream
