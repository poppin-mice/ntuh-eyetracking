"""Math shared by the offline calibration replay (tools/calib_replay) and the replayer.

Pure numpy, no SDK and no GUI.
"""
import numpy as np


def angular_error_deg(gaze_xy, target_xy, screen_w_px, screen_h_px, view_dist_cm, screen_width_cm):
    """Angle in degrees between the eye->gaze ray and the eye->target ray.

    The eye is on the screen normal through the screen centre, view_dist_cm away. Points are screen
    pixels, one (x, y) or an array (..., 2). accuracy_test.px_to_deg reads the length of the error as
    if it started at the screen centre, so it overstates an error that lies away from the centre;
    this does not.
    """
    cm_per_px = float(screen_width_cm) / screen_w_px
    centre = np.array([screen_w_px / 2.0, screen_h_px / 2.0])

    def ray(xy):
        xy = (np.asarray(xy, dtype=float) - centre) * cm_per_px
        return np.concatenate([xy, np.full(xy.shape[:-1] + (1,), float(view_dist_cm))], axis=-1)

    a, b = ray(gaze_xy), ray(target_xy)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))
