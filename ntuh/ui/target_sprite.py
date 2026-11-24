"""The target of the Sol 2D calibration and of the accuracy test (REQ-0075).

Today's ring, a still picture (PNG, JPG) or an animated GIF, drawn with its centre on the target
point. A picture's size is a visual angle, so it is the same on every screen. No Tk here; Pillow
and pygame are imported only when a picture is loaded or drawn, so the logic is testable without.
"""
import math
import os
import time

from ntuh.recording.session_bundle import sha256_file
from ntuh.sol.accuracy_test import px_per_cm_of

RING, IMAGE, GIF = "ring", "image", "gif"
RING_PX = 52                # outer diameter of the ring drawn by draw_ring
DEFAULT_FRAME_MS = 100      # a GIF frame without a duration (what browsers show)
DOT_COLOR, DOT_RADIUS_PX = (255, 0, 0), 3
FILETYPES = [("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All", "*.*")]


def size_px(size_deg, view_dist_cm, screen_w_px, screen_width_cm):
    """Screen px of size_deg of visual angle: the inverse of accuracy_test.px_to_deg."""
    return max(1, round(view_dist_cm * math.tan(math.radians(size_deg))
                        * px_per_cm_of(screen_w_px, screen_width_cm)))


def top_left(center, wh):
    """Where to blit a w x h picture so that its centre is on `center`."""
    return int(center[0]) - wh[0] // 2, int(center[1]) - wh[1] // 2


def frame_at(durations_ms, t_s):
    """Index of the GIF frame shown t_s seconds after the start; the animation loops."""
    t = int(t_s * 1000) % sum(durations_ms)
    for i, d in enumerate(durations_ms):
        if t < d:
            return i
        t -= d


def draw_ring(win, center, col):
    """The accuracy-test target as it was before 1.19.0, unchanged."""
    import pygame
    tx, ty = center
    pygame.draw.circle(win, col, (tx, ty), 26, 3)
    pygame.draw.circle(win, col, (tx, ty), 6)
    pygame.draw.line(win, col, (tx - 36, ty), (tx + 36, ty), 1)
    pygame.draw.line(win, col, (tx, ty - 36), (tx, ty + 36), 1)


class Sprite:
    def __init__(self, kind, path=None, frames=(), durations_ms=(), size=RING_PX, center_dot=False,
                 warning=None, requested=None):
        self.kind, self.path, self.frames, self.durations_ms = kind, path, list(frames), list(durations_ms)
        self.size, self.center_dot = size, center_dot
        self.warning, self.requested = warning, requested    # why a picture fell back to the ring
        self.t0 = time.monotonic()

    def draw(self, win, center, col=(255, 255, 0)):
        """Draw the target centred on `center`. `col` is the ring's colour; a picture ignores it."""
        if self.kind == RING:
            return draw_ring(win, center, col)
        import pygame
        frame = self.frames[frame_at(self.durations_ms, time.monotonic() - self.t0)]
        win.blit(frame, top_left(center, frame.get_size()))
        if self.center_dot:
            pygame.draw.circle(win, DOT_COLOR, center, DOT_RADIUS_PX)

    def record(self, size_deg):
        """The `target` block of session_meta.json."""
        ring = self.kind == RING
        return {"kind": self.kind,
                "file": None if ring else os.path.basename(self.path),
                "sha256": None if ring else sha256_file(self.path),
                "size_deg": None if ring else size_deg,
                "size_px": self.size,
                "n_frames": len(self.frames) or None,
                "center_dot": False if ring else bool(self.center_dot),
                "fallback": self.warning,
                "requested_file": self.requested and os.path.basename(self.requested)}

    def note(self, bundle, size_deg):
        """Record the target in the session; a picture that could not be used is a warning event."""
        bundle.update(target=self.record(size_deg))
        if self.warning:
            bundle.event(self.warning, file=self.requested, fallback=RING)


def load(path, size, center_dot=False):
    """The target for `path` ("" = the ring), longest edge `size` px, every GIF frame decoded now.

    A file that is missing or cannot be decoded gives the ring with .warning set.
    """
    path = (path or "").strip()
    if not path:
        return Sprite(RING)
    if not os.path.isfile(path):
        print(f"[Target] not found, using the ring: {path}")
        return Sprite(RING, warning="target_image_missing", requested=path)
    try:
        import pygame
        from PIL import Image, ImageSequence
        frames, durations = [], []
        with Image.open(path) as im:
            scale = size / max(im.size)
            wh = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            for fr in ImageSequence.Iterator(im):
                rgba = fr.convert("RGBA").resize(wh, Image.LANCZOS)
                frames.append(pygame.image.frombytes(rgba.tobytes(), wh, "RGBA"))
                durations.append(fr.info.get("duration") or DEFAULT_FRAME_MS)
    except Exception as e:
        print(f"[Target] cannot read {path}, using the ring: {e}")
        return Sprite(RING, warning="target_image_unreadable", requested=path)
    return Sprite(GIF if len(frames) > 1 else IMAGE, path, frames, durations, size, center_dot)
