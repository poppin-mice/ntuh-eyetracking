"""Sol gaze accuracy-test: target layout (concentric rings + screen corners), raw-vs-corrected
gaze mapping, and the report (CSV + JSON + heatmap/by-angle PNGs).

Pure logic (no SDK, no GUI) so it is unit-testable. The interactive pygame flow lives in
settings_window.run_sol_accuracy_test, which calls into here.
"""
import csv
import json
import math
import os

import numpy as np


def px_per_cm_of(screen_w_px, screen_width_cm):
    return screen_w_px / float(screen_width_cm) if screen_width_cm else 0.0


def eccentricity_deg(x_px, y_px, screen_w_px, screen_h_px, view_dist_cm, screen_width_cm):
    """Visual-angle eccentricity (deg) of a screen pixel from screen center."""
    ppc = px_per_cm_of(screen_w_px, screen_width_cm)
    if ppc <= 0 or view_dist_cm <= 0:
        return 0.0
    r_cm = math.hypot(x_px - screen_w_px / 2.0, y_px - screen_h_px / 2.0) / ppc
    return math.degrees(math.atan(r_cm / view_dist_cm))


def px_to_deg(err_px, view_dist_cm, screen_w_px, screen_width_cm):
    """Convert a screen-pixel error magnitude to degrees of visual angle (small-angle exact)."""
    ppc = px_per_cm_of(screen_w_px, screen_width_cm)
    if ppc <= 0 or view_dist_cm <= 0:
        return 0.0
    return math.degrees(math.atan((err_px / ppc) / view_dist_cm))


def compute_accuracy_targets(screen_w, screen_h, view_dist_cm, screen_width_cm,
                             ring_eccentricities_deg=(10.0, 20.0), points_per_ring=8,
                             include_corners=True, corner_inset_frac=0.06):
    """Build the accuracy-test target list.

    center (0 deg) + one concentric ring per requested eccentricity (points_per_ring points evenly
    spaced, i.e. 8 -> the 4 cardinal up/down/left/right + the 4 diagonal directions) + the 4 screen
    corners. Ring points that fall off-screen are skipped (e.g. 20 deg vertical on a 4K screen at
    60 cm). Eccentricity -> pixels via the viewing distance and the screen's px/cm.

    Returns a list of dicts: {name, group, ecc_deg, x, y}. group in {center, <ecc>deg, corner}.
    """
    ppc = px_per_cm_of(screen_w, screen_width_cm)
    cx, cy = screen_w / 2.0, screen_h / 2.0
    targets = [{"name": "center", "group": "center", "ecc_deg": 0.0, "x": cx, "y": cy}]

    n = max(1, int(points_per_ring))
    for ecc in ring_eccentricities_deg:
        r_px = view_dist_cm * math.tan(math.radians(ecc)) * ppc
        for k in range(n):
            theta = k * (360.0 / n)
            x = cx + r_px * math.cos(math.radians(theta))
            y = cy - r_px * math.sin(math.radians(theta))  # screen y grows downward
            if 0 <= x < screen_w and 0 <= y < screen_h:
                targets.append({"name": f"{ecc:.0f}deg@{int(theta)}", "group": f"{ecc:.0f}deg",
                                "ecc_deg": float(ecc), "x": float(x), "y": float(y)})

    if include_corners:
        ix, iy = corner_inset_frac * screen_w, corner_inset_frac * screen_h
        for nm, x, y in (("corner-TL", ix, iy), ("corner-TR", screen_w - ix, iy),
                         ("corner-BR", screen_w - ix, screen_h - iy), ("corner-BL", ix, screen_h - iy)):
            targets.append({"name": nm, "group": "corner",
                            "ecc_deg": eccentricity_deg(x, y, screen_w, screen_h, view_dist_cm, screen_width_cm),
                            "x": float(x), "y": float(y)})
    return targets


def _apply_homography(H, pt):
    """Map a 2D point through a 3x3 homography. Returns (x, y) or None."""
    if H is None:
        return None
    s = np.array([pt[0], pt[1], 1.0], dtype=float)
    d = np.asarray(H, dtype=float) @ s
    if abs(d[2]) < 1e-9:
        return None
    return (float(d[0] / d[2]), float(d[1] / d[2]))


def map_raw_and_corrected(H, gaze_2d, offset_model):
    """Map a raw scene-camera gaze_2d to screen pixels BOTH without and with the offset model.

    Returns (raw_screen, corrected_screen), either may be None if H is unavailable. Handles both
    screen-mode (offset applied AFTER the homography) and legacy camera-mode (before) models, and
    bypasses the on-screen culling so large errors are still measured.
    """
    raw = _apply_homography(H, gaze_2d)
    if raw is None:
        return None, None
    if offset_model is None or not getattr(offset_model, "is_trained", False):
        return raw, raw
    mode = getattr(offset_model, "offset_mode", "")
    try:
        if mode == "screen":
            corr = tuple(offset_model.predict(raw))
        else:  # legacy camera/angular: correct in camera space, then map
            corr_cam = offset_model.predict(gaze_2d)
            corr = _apply_homography(H, corr_cam)
    except Exception:
        corr = raw
    return raw, (corr if corr is not None else raw)


def compute_point_record(target, raw_samples, corr_samples, view_dist_cm, screen_w, screen_width_cm):
    """Build a per-point record from collected screen samples.

    target: {name, group, ecc_deg, x, y}. raw_samples / corr_samples: lists of (x, y) screen px
    (before / after offset). ACCURACY = |mean(samples) - target|. PRECISION = SD of the samples
    about their own mean (RMS distance from the centroid - gaze stability/noise). Both in px + deg.
    """
    def stats(samples):
        a = np.asarray(samples, dtype=float)
        if len(a) == 0:
            return np.array([0.0, 0.0]), 0.0
        mean = a.mean(axis=0)
        prec_px = float(np.sqrt(np.mean(np.sum((a - mean) ** 2, axis=1)))) if len(a) > 1 else 0.0
        return mean, prec_px

    raw_mean, raw_prec = stats(raw_samples)
    cor_mean, cor_prec = stats(corr_samples)
    tx, ty = float(target["x"]), float(target["y"])
    err_raw = float(np.hypot(raw_mean[0] - tx, raw_mean[1] - ty))
    err_cor = float(np.hypot(cor_mean[0] - tx, cor_mean[1] - ty))
    d = lambda px: px_to_deg(px, view_dist_cm, screen_w, screen_width_cm)
    return {
        "name": target["name"], "group": target["group"], "ecc_deg": float(target["ecc_deg"]),
        "target_x": tx, "target_y": ty,
        "raw_x": float(raw_mean[0]), "raw_y": float(raw_mean[1]),
        "corr_x": float(cor_mean[0]), "corr_y": float(cor_mean[1]),
        "err_raw_px": err_raw, "err_corr_px": err_cor,
        "err_raw_deg": d(err_raw), "err_corr_deg": d(err_cor),
        "prec_raw_px": raw_prec, "prec_corr_px": cor_prec,
        "prec_raw_deg": d(raw_prec), "prec_corr_deg": d(cor_prec),
        "n_samples": len(corr_samples),
    }


def summarize(records):
    """Group per-point records by `group`; mean/median/max ACCURACY + mean PRECISION (raw & corr)."""
    groups = {}
    for r in records:
        groups.setdefault(r["group"], []).append(r)
    summary = {}
    for g, rs in groups.items():
        raw = np.array([r["err_raw_deg"] for r in rs], dtype=float)
        cor = np.array([r["err_corr_deg"] for r in rs], dtype=float)
        summary[g] = {
            "n": len(rs),
            "ecc_deg": float(np.mean([r["ecc_deg"] for r in rs])),
            # degrees
            "raw_mean_deg": float(np.mean(raw)), "raw_max_deg": float(np.max(raw)),
            "corr_mean_deg": float(np.mean(cor)), "corr_median_deg": float(np.median(cor)),
            "corr_max_deg": float(np.max(cor)),
            "prec_raw_deg": float(np.mean([r["prec_raw_deg"] for r in rs])),
            "prec_corr_deg": float(np.mean([r["prec_corr_deg"] for r in rs])),
            # pixels
            "raw_mean_px": float(np.mean([r["err_raw_px"] for r in rs])),
            "corr_mean_px": float(np.mean([r["err_corr_px"] for r in rs])),
            "corr_max_px": float(np.max([r["err_corr_px"] for r in rs])),
            "prec_raw_px": float(np.mean([r["prec_raw_px"] for r in rs])),
            "prec_corr_px": float(np.mean([r["prec_corr_px"] for r in rs])),
        }
    allraw = np.array([r["err_raw_deg"] for r in records], dtype=float)
    allcor = np.array([r["err_corr_deg"] for r in records], dtype=float)
    summary["_overall"] = {
        "n": len(records),
        "raw_mean_deg": float(np.mean(allraw)) if len(allraw) else 0.0,
        "corr_mean_deg": float(np.mean(allcor)) if len(allcor) else 0.0,
        "corr_max_deg": float(np.max(allcor)) if len(allcor) else 0.0,
        "prec_corr_deg": float(np.mean([r["prec_corr_deg"] for r in records])) if records else 0.0,
        "raw_mean_px": float(np.mean([r["err_raw_px"] for r in records])) if records else 0.0,
        "corr_mean_px": float(np.mean([r["err_corr_px"] for r in records])) if records else 0.0,
        "corr_max_px": float(np.max([r["err_corr_px"] for r in records])) if records else 0.0,
        "prec_corr_px": float(np.mean([r["prec_corr_px"] for r in records])) if records else 0.0,
    }
    return summary


# What the result dialog shows: the overall point count, mean accuracy before / after the offset
# and mean precision. Saved as accuracy_report.json -> meta.result.
RESULT_KEYS = ("n", "raw_mean_deg", "raw_mean_px", "corr_mean_deg", "corr_mean_px",
               "prec_corr_deg", "prec_corr_px")


def result_block(records):
    overall = summarize(records)["_overall"]
    return {key: overall[key] for key in RESULT_KEYS}


def build_meta(user, timestamp, screen_w, screen_h, view_dist_cm, screen_width_cm, n_points,
               offset_model, aborted, gate, profile_path, profile_calibration_timestamp):
    """The "meta" block of accuracy_report.json.

    The first ten keys are the original ones and keep their names and meaning (the QC scripts and
    the model repo read them). The last five say which condition was run and what happened to our
    calibration profile: `gate` is a ntuh.sol.calib_gate.GateResult, `profile_path` the profile
    file this User Name maps to (loaded only when calib_model_loaded is true) and
    `profile_calibration_timestamp` the time inside that profile, None unless it was applied.
    """
    return {
        "user": user, "timestamp": timestamp, "screen_w": screen_w, "screen_h": screen_h,
        "view_dist_cm": view_dist_cm, "screen_width_cm": screen_width_cm,
        "n_points": n_points, "calib_model_loaded": offset_model is not None,
        "offset_mode": getattr(offset_model, 'offset_mode', None) if offset_model else None,
        "aborted": aborted,
        "condition_code": gate.condition,
        "our_calibration_expected": gate.expected,
        "profile_path": profile_path,
        "profile_calibration_timestamp": profile_calibration_timestamp,
        "gate": gate.to_dict(),
    }


def save_accuracy_report(out_dir, records, meta):
    """Write accuracy_data.csv, accuracy_report.json, accuracy_heatmap.png, accuracy_by_angle.png.

    records: list of dicts with keys name, group, ecc_deg, target_x, target_y, raw_x, raw_y,
    corr_x, corr_y, err_raw_px, err_corr_px, err_raw_deg, err_corr_deg, n_samples. Returns the dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = summarize(records) if records else {}

    cols = ["name", "group", "ecc_deg", "target_x", "target_y", "raw_x", "raw_y", "corr_x", "corr_y",
            "err_raw_px", "err_corr_px", "err_raw_deg", "err_corr_deg",
            "prec_raw_px", "prec_corr_px", "prec_raw_deg", "prec_corr_deg", "n_samples"]
    with open(os.path.join(out_dir, "accuracy_data.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow([r.get(c, "") for c in cols])

    with open(os.path.join(out_dir, "accuracy_report.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "points": records}, f, indent=2)

    try:
        _render_pngs(out_dir, records, summary, meta)
    except Exception as e:
        print(f"[Accuracy] Report PNG rendering failed: {e}")

    return out_dir


def _render_pngs(out_dir, records, summary, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sw, sh = meta.get("screen_w", 1920), meta.get("screen_h", 1080)

    # --- Heatmap: before vs after, gaze error over screen position ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, key, title in ((axes[0], "raw", "Before offset (raw)"),
                           (axes[1], "corr", "After offset (corrected)")):
        tx = [r["target_x"] for r in records]
        ty = [r["target_y"] for r in records]
        err = [r[f"err_{key}_deg"] for r in records]
        gx = [r[f"{key}_x"] for r in records]
        gy = [r[f"{key}_y"] for r in records]
        sc = ax.scatter(tx, ty, c=err, cmap="jet", s=260, vmin=0,
                        vmax=max(1.0, max((r["err_raw_deg"] for r in records), default=1.0)),
                        edgecolors="k", zorder=3)
        for x0, y0, x1, y1 in zip(tx, ty, gx, gy):  # target -> measured gaze arrows
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1, alpha=0.6), zorder=2)
        ax.set_xlim(0, sw); ax.set_ylim(sh, 0); ax.set_aspect("equal")
        ax.set_title(f"{title}\nmean {summary.get('_overall', {}).get(key + '_mean_deg', 0):.2f} deg")
        ax.set_xlabel("screen x (px)"); ax.set_ylabel("screen y (px)")
        fig.colorbar(sc, ax=ax, label="error (deg)")
    fig.suptitle(f"Sol gaze accuracy - {meta.get('user', '')}  ({meta.get('n_points', len(records))} pts, "
                 f"dist={meta.get('view_dist_cm', '?')}cm)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_heatmap.png"), dpi=110)
    plt.close(fig)

    # --- Mean error by eccentricity ring (before vs after) ---
    order = [g for g in ("center", "10deg", "14deg", "20deg", "corner") if g in summary]
    order += [g for g in summary if g not in order and not g.startswith("_")]
    if order:
        idx = np.arange(len(order)); bw = 0.38
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))

        def _bars(ax, raw_key, cor_key, ylabel, title):
            ax.bar(idx - bw / 2, [summary[g][raw_key] for g in order], bw, label="raw", color="#d9534f")
            ax.bar(idx + bw / 2, [summary[g][cor_key] for g in order], bw, label="corrected", color="#5cb85c")
            ax.set_xticks(idx); ax.set_xticklabels(order)
            ax.set_ylabel(ylabel); ax.set_xlabel("eccentricity group")
            ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=0.3)

        # Row 1 = degrees, Row 2 = pixels; col 1 = accuracy (error), col 2 = precision (stability)
        _bars(axes2[0][0], "raw_mean_deg", "corr_mean_deg", "mean accuracy error (deg)", "Accuracy - degrees")
        _bars(axes2[0][1], "prec_raw_deg", "prec_corr_deg", "mean precision SD (deg)", "Precision - degrees")
        _bars(axes2[1][0], "raw_mean_px", "corr_mean_px", "mean accuracy error (px)", "Accuracy - pixels")
        _bars(axes2[1][1], "prec_raw_px", "prec_corr_px", "mean precision SD (px)", "Precision - pixels")
        fig2.suptitle("Accuracy & precision by eccentricity (before vs after offset)")
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "accuracy_by_angle.png"), dpi=110)
        plt.close(fig2)
