"""Import-safe IPC helpers + message constants shared by the isolated Sol child and the parent.

Stdlib-only (types.SimpleNamespace) so importing this pulls in NEITHER the Sol SDK / native av
(parent must stay SDK-free) NOR the GUI stack. The child converts SDK GazeData -> flat picklable
dicts here; the parent reconstructs them into attribute-access objects here.
"""
from types import SimpleNamespace

# child -> parent message types (msg_q)
CONNECTED = "connected"
CONNECT_FAILED = "connect_failed"
HOMOGRAPHY = "homography"
POSE = "pose"
STOPPING = "stopping"
ERROR = "error"

# parent -> child command types (cmd_q)
RESUME_SCENE = "resume_scene"
PAUSE_SCENE = "pause_scene"
STOP = "stop"


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _b(v):
    try:
        return bool(v)
    except Exception:
        return False


def _eye_to_dict(e):
    gz = e.gaze
    o = gz.origin
    d = getattr(gz, "direction", None)
    p3 = getattr(e, "pupil3d", None)
    p2 = getattr(e, "pupil2d", None)
    es = getattr(e, "eye_status", None)
    return {
        "confidence": _f(getattr(e, "confidence", 0.0)),
        # plain int (EyeStatus.value): 0=NO_EYE, 1=BLINK, 2=NORMAL; -1 if absent.
        "eye_status": int(getattr(es, "value", es)) if es is not None else -1,
        "gaze": {
            "validity": _b(getattr(gz, "validity", False)),
            "origin": {"x": _f(o.x), "y": _f(o.y), "z": _f(o.z)},
            "direction": ({"x": _f(d.x), "y": _f(d.y), "z": _f(d.z)}
                          if d is not None else {"x": 0.0, "y": 0.0, "z": 0.0}),
        },
        "pupil3d": ({"validity": _f(getattr(p3, "validity", 0.0)), "diameter": _f(getattr(p3, "diameter", 0.0))}
                    if p3 is not None else {"validity": 0.0, "diameter": 0.0}),
        "pupil2d": ({"x": _f(getattr(p2, "x", 0.0)), "y": _f(getattr(p2, "y", 0.0))}
                    if p2 is not None else {"x": 0.0, "y": 0.0}),
    }


def gaze_to_dict(g):
    """Convert an SDK GazeData sample into a flat, picklable nested dict (only bool/int/float/str).

    Duck-typed so it tolerates SDK field-shape variations. Always emits every key (so parent-side
    hasattr() guards behave like the always-present SDK dataclass attributes). Returns None if the
    sample is falsy or cannot be read (the parent treats None as 'no sample this frame').
    """
    if g is None:
        return None
    try:
        c = g.combined
        g2 = c.gaze_2d
        g3 = c.gaze_3d
        return {
            "combined": {
                "gaze_2d": {"validity": _b(getattr(g2, "validity", False)), "x": _f(g2.x), "y": _f(g2.y)},
                "gaze_3d": {"validity": _b(getattr(g3, "validity", False)),
                            "x": _f(g3.x), "y": _f(g3.y), "z": _f(g3.z)},
            },
            "left_eye": _eye_to_dict(g.left_eye),
            "right_eye": _eye_to_dict(g.right_eye),
            # the SDK's GazeData field is timestamp_ms (device clock, Unix ms); reading
            # "timestamp" gave 0 for every sample
            "timestamp": int(getattr(g, "timestamp_ms", None) or getattr(g, "timestamp", 0) or 0),
        }
    except Exception:
        return None


def reconstruct(d):
    """Rebuild a flat dict into attribute-access objects so existing consumer code
    (latest_gaze.combined.gaze_2d.x, sample.left_eye.gaze.origin.z, ...) works unchanged.
    Nested nodes become SimpleNamespace (always truthy, like the SDK dataclasses); leaves pass
    through. None passes through (so a genuinely-absent sub-tree -> hasattr False, value None)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: reconstruct(v) for k, v in d.items()})
    return d
