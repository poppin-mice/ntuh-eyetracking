"""Calibration models to compare.

MODELS maps a name to (fit, grid). fit(gaze, target) takes the calibration pairs as (n, 2) arrays in
screen pixels and returns predict(xy, param) -> corrected (m, 2). grid lists the values to try for the
model's one hyper-parameter; it is None when the model has none (param is then ignored).
"""
import contextlib

import numpy as np

from ntuh.sol.offset_calibration_2d import OFFSET_MODE_SCREEN, Sol2DOffsetModel

GRID = tuple(round(0.1 * i, 1) for i in range(11))     # shrink factors 0.0 .. 1.0
LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
HUBER_C = 1.345


def _shipped(gaze, target):
    """The shipped model itself (IDW power 2, snap within 10 px), fitted as the calibration fits it."""
    model = Sol2DOffsetModel(offset_mode=OFFSET_MODE_SCREEN)
    for g, t in zip(gaze.tolist(), target.tolist()):
        model.add_calibration_point(tuple(g), tuple(t))
    with contextlib.redirect_stdout(None):              # train() prints every point
        model.train()
    return lambda xy: np.array([model.predict(tuple(p)) for p in xy.tolist()], dtype=float).reshape(-1, 2)


def _none(gaze, target):
    return lambda xy, _=None: xy


def _idw_shipped(gaze, target):
    f = _shipped(gaze, target)
    return lambda xy, _=None: f(xy)


def _idw_shrink(gaze, target):
    """a times the shipped IDW offset (0 = none, 1 = idw_shipped)."""
    f = _shipped(gaze, target)
    return lambda xy, a: xy + a * (f(xy) - xy)


def _constant(centre):
    """One 2-D offset for the whole screen, times w when a shrink factor is given."""
    def fit(gaze, target):
        offset = centre(target - gaze)
        return lambda xy, w=None: xy + (1.0 if w is None else w) * offset
    return fit


def _huber_location(d):
    """Huber M-estimate of the centre of 2-D points: a point further than HUBER_C robust SDs from the
    centre counts less, in proportion to its distance."""
    mu = np.median(d, axis=0)
    scale = 1.4826 * np.median(np.linalg.norm(d - mu, axis=1))
    if scale == 0:
        return mu
    for _ in range(50):
        r = np.linalg.norm(d - mu, axis=1)
        w = np.minimum(1.0, HUBER_C * scale / np.maximum(r, 1e-12))
        mu = (w[:, None] * d).sum(axis=0) / w.sum()
    return mu


def _gain_offset(gaze, target):
    """Per axis: target = a * gaze + b, least squares."""
    centre, d = gaze.mean(axis=0), target - gaze
    gc = gaze - centre
    ss = (gc ** 2).sum(axis=0)
    gain = np.divide((gc * (d - d.mean(axis=0))).sum(axis=0), ss, out=np.zeros(2), where=ss > 0)
    return lambda xy, _=None: xy + d.mean(axis=0) + gain * (xy - centre)


def _affine(gaze, target):
    """target = A gaze + b. With lam > 0, A is ridge-shrunk toward the identity; lam is in units of
    the mean variance of the calibration gaze points, so 0 is the plain least-squares fit and a
    large value is const_mean."""
    centre, d = gaze.mean(axis=0), target - gaze
    gc, dc = gaze - centre, d - d.mean(axis=0)
    cov = gc.T @ gc

    def predict(xy, lam=None):
        ridge = (lam or 0.0) * np.trace(cov) / 2.0 * np.eye(2)
        m = np.linalg.lstsq(cov + ridge, gc.T @ dc, rcond=None)[0]
        return xy + d.mean(axis=0) + (xy - centre) @ m
    return predict


def _const_plus_idw_residual(gaze, target):
    """const_mean, plus a times the shipped IDW of what const_mean leaves (1 = idw_shipped)."""
    offset = (target - gaze).mean(axis=0)
    f = _shipped(gaze, target - offset)
    return lambda xy, a: xy + offset + a * (f(xy) - xy)


MODELS = {
    "none": (_none, None),
    "idw_shipped": (_idw_shipped, None),
    "idw_shrink": (_idw_shrink, GRID),
    "const_mean": (_constant(lambda d: d.mean(axis=0)), None),
    "const_median": (_constant(lambda d: np.median(d, axis=0)), None),
    "huber_shrink": (_constant(_huber_location), GRID),
    "gain_offset": (_gain_offset, None),
    "affine": (_affine, None),
    "affine_shrunk": (_affine, LAMBDAS),
    "const_plus_idw_residual": (_const_plus_idw_residual, GRID),
}
