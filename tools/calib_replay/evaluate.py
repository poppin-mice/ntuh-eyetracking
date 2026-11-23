"""Protocols, metrics, hyper-parameter choice and the bootstrap."""
import numpy as np

from ntuh.sol.accuracy_test import px_to_deg
from ntuh.sol.replay_math import angular_error_deg

from .models import MODELS

METRICS = ("n_points", "mean_deg", "median_deg", "mean_deg_legacy", "median_deg_legacy",
           "pct_lt2", "pct_lt5")


def point_errors(pred, target, meta):
    """Per point: error in px, angular error with the real screen geometry, legacy px_to_deg."""
    px = np.linalg.norm(pred - target, axis=1)
    exact = angular_error_deg(pred, target, meta["screen_w"], meta["screen_h"],
                              meta["view_dist_cm"], meta["screen_width_cm"])
    legacy = np.array([px_to_deg(e, meta["view_dist_cm"], meta["screen_w"], meta["screen_width_cm"])
                       for e in px])
    return px, exact, legacy


def _splits(run, nodes, ks, n_layouts, rng):
    """{protocol: layouts}. A layout is a list of folds (fit gaze, fit target, points to predict)."""
    g, t = run["raw"], run["target"]
    every = np.arange(len(g))
    out = {}
    if nodes is not None:                               # P1: the saved calibration, as deployed
        out["P1"] = [[(nodes[0], nodes[1], every)]]
    if len(g) >= 2:                                     # P2: leave one point out
        out["P2"] = [[(np.delete(g, i, 0), np.delete(t, i, 0), every[i:i + 1]) for i in every]]
    for k in ks:                                        # P3: k points calibrate, the rest test
        if k < len(g):
            fits = [np.sort(rng.choice(len(g), k, replace=False)) for _ in range(n_layouts)]
            out[f"P3_k{k}"] = [[(g[f], t[f], np.setdiff1d(every, f))] for f in fits]
    return out


def evaluate_run(run, nodes, ks, n_layouts, rng):
    """One accuracy test through every protocol, model and hyper-parameter value.

    Returns (trials, points), both keyed (protocol, model, param). trials: the METRICS, averaged over
    the protocol's layouts. points: the predicted xy of every point, for one-layout protocols."""
    g, t, meta = run["raw"], run["target"], run["meta"]

    def score(pred):
        seen = ~np.isnan(pred[:, 0])
        _, exact, legacy = point_errors(pred[seen], t[seen], meta)
        return [seen.sum(), exact.mean(), np.median(exact), legacy.mean(), np.median(legacy),
                100.0 * np.mean(exact < 2), 100.0 * np.mean(exact < 5)]

    points = {("recorded", "none", None): g, ("recorded", "corr", None): run["corr"]}
    trials = {key: score(pred) for key, pred in points.items()}
    for protocol, layouts in _splits(run, nodes, ks, n_layouts, rng).items():
        for name, (fit, grid) in MODELS.items():
            scores = {p: [] for p in grid or (None,)}
            for layout in layouts:
                preds = {p: np.full(g.shape, np.nan) for p in scores}
                for fit_g, fit_t, test in layout:
                    predict = fit(fit_g, fit_t)
                    for p in scores:
                        preds[p][test] = predict(g[test], p)
                for p in scores:
                    scores[p].append(score(preds[p]))
                    if len(layouts) == 1:
                        points[protocol, name, p] = preds[p]
            for p in scores:
                trials[protocol, name, p] = np.mean(scores[p], axis=0)
    return trials, points


def mark_selected(rows, by):
    """Set row["selected"] (1 / 0) on per-trial rows.

    A model with a hyper-parameter gets, for each subject, the value with the lowest mean `by` over
    the runs of the OTHER subjects, never the subject's own. Other models are always selected."""
    table = {}                                          # (protocol, model) -> param -> subject -> values
    for r in rows:
        if r["param"] is not None:
            (table.setdefault((r["protocol"], r["model"]), {}).setdefault(r["param"], {})
             .setdefault(r["subject"], []).append(r[by]))
    best = {}
    for key, by_param in table.items():
        for subject in {s for d in by_param.values() for s in d}:
            others = {p: [v for s, vs in d.items() if s != subject for v in vs]
                      for p, d in by_param.items()}
            others = {p: np.mean(v) for p, v in others.items() if v}
            if others:
                best[key + (subject,)] = min(others, key=others.get)
    for r in rows:
        r["selected"] = int(r["param"] is None
                            or best.get((r["protocol"], r["model"], r["subject"])) == r["param"])


def boot_ci(values, subjects, rng, reps=2000):
    """95 % interval of the mean of `values` (one per run), resampling whole subjects."""
    names = sorted(set(subjects))
    sums = np.array([sum(v for v, s in zip(values, subjects) if s == n) for n in names])
    counts = np.array([sum(s == n for s in subjects) for n in names])
    pick = rng.integers(0, len(names), (reps, len(names)))
    return np.percentile(sums[pick].sum(axis=1) / counts[pick].sum(axis=1), [2.5, 97.5])
