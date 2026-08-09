"""Parent-side supervisor for the isolated Sol scene worker (ntuh.sol.sol_child.run_child).

Owns the child Process + the three multiprocessing queues, the spawn/respawn/graceful-stop
lifecycle, and message draining. Stays SDK-free (the whole point of isolation): imports only
multiprocessing, time, ntuh.sol.sol_ipc, and the run_child target (whose SDK imports are deferred
inside the child).
"""
import multiprocessing as mp

from ntuh.sol import sol_ipc
from ntuh.sol.sol_child import run_child


class SolPreviewClient:
    BACKOFF_S = 1.0          # wait this long after a crash before respawning
    MAX_RESPAWNS = 5         # give up after this many consecutive respawns

    def __init__(self, params):
        self.params = dict(params)
        self.proc = None
        self.gaze_q = None
        self.msg_q = None
        self.cmd_q = None
        self.frame_q = None
        self.intentional_stop = False
        self._crashed = False
        self._respawn_at = 0.0
        self.respawn_count = 0
        self.last_exitcode = None

    # ---- lifecycle ----
    def _spawn(self):
        self.gaze_q = mp.Queue(maxsize=120)
        self.msg_q = mp.Queue(maxsize=400)
        self.cmd_q = mp.Queue(maxsize=50)
        # Scene frames for an operator view, only when asked for. Depth 2 so a slow parent makes
        # the child drop frames instead of building a backlog of stale ones.
        self.frame_q = mp.Queue(maxsize=2) if self.params.get("stream_frames") else None
        self.proc = mp.Process(target=run_child,
                               args=(self.params, self.gaze_q, self.msg_q, self.cmd_q, self.frame_q),
                               daemon=True, name="sol_scene_worker")
        self.proc.start()

    def start(self):
        self.intentional_stop = False
        self._crashed = False
        self.respawn_count = 0
        self._spawn()

    def _close_queues(self):
        for q in (self.gaze_q, self.msg_q, self.cmd_q, self.frame_q):
            if q is None:
                continue
            try:
                q.close()
                q.cancel_join_thread()  # don't let the feeder thread block parent exit
            except Exception:
                pass
        self.gaze_q = self.msg_q = self.cmd_q = self.frame_q = None

    def stop(self, timeout=4.0):
        """Graceful stop: signal STOP (-> AsyncClient __aexit__ closes TCP), join, terminate only
        as a last resort, then close the queues."""
        self.intentional_stop = True
        self._send(sol_ipc.STOP)
        if self.proc is not None:
            try:
                self.proc.join(timeout)
            except Exception:
                pass
            if self.proc.is_alive():
                try:
                    self.proc.terminate()
                except Exception:
                    pass
                try:
                    self.proc.join(2.0)
                except Exception:
                    pass
        self._close_queues()
        self.proc = None

    # ---- IPC ----
    def drain_gaze(self, cap=100):
        """Return up to `cap` reconstructed gaze objects (newest last)."""
        out = []
        if self.gaze_q is None:
            return out
        for _ in range(cap):
            try:
                d = self.gaze_q.get_nowait()
            except Exception:
                break
            if d is not None:
                out.append(sol_ipc.reconstruct(d))
        return out

    def drain_msgs(self, cap=400):
        """Return up to `cap` raw control/status message dicts."""
        out = []
        if self.msg_q is None:
            return out
        for _ in range(cap):
            try:
                out.append(self.msg_q.get_nowait())
            except Exception:
                break
        return out

    def drain_frames(self):
        """Newest scene frame as ((h, w, 3) uint8 array, step), or None if none arrived.

        `step` is the child's downscale factor: multiply camera-space overlay coordinates by
        1/step to land on this image. Older frames are dropped so the operator sees live video
        rather than a backlog. numpy-only, to keep this module SDK-free.
        """
        if self.frame_q is None:
            return None
        latest = None
        while True:
            try:
                latest = self.frame_q.get_nowait()
            except Exception:
                break
        if latest is None:
            return None
        try:
            import numpy as np
            shape, step, buf = latest
            return np.frombuffer(buf, dtype=np.uint8).reshape(shape), step
        except Exception:
            return None

    def _send(self, cmd):
        try:
            if self.cmd_q is not None:
                self.cmd_q.put_nowait({"cmd": cmd})
        except Exception:
            pass

    def resume(self):
        self._send(sol_ipc.RESUME_SCENE)

    def pause(self):
        self._send(sol_ipc.PAUSE_SCENE)

    # ---- supervisor ----
    def poll_supervisor(self, now, last_seed_H=None):
        """Call every frame. Returns one of:
        'crashed'   - just detected the child died unexpectedly (caller: mark homography CACHED),
        'respawned' - a fresh child was started (caller: nothing; CONNECTED handler re-sends resume),
        'failed'    - gave up after MAX_RESPAWNS (caller: show error, exit preview),
        None        - nothing to do.
        last_seed_H: the parent's current homography (np.ndarray|None) to seed the new child."""
        if self.intentional_stop or self.proc is None:
            return None
        if self.proc.is_alive():
            return None
        # Child is dead and we did not ask it to stop -> crash.
        if not self._crashed:
            self._crashed = True
            try:
                self.last_exitcode = self.proc.exitcode
            except Exception:
                self.last_exitcode = None
            self._respawn_at = now + self.BACKOFF_S
            return "crashed"
        if self.respawn_count >= self.MAX_RESPAWNS:
            return "failed"
        if now >= self._respawn_at:
            self.respawn_count += 1
            self._close_queues()
            if last_seed_H is not None:
                try:
                    self.params["seed_homography"] = last_seed_H.tolist()
                except Exception:
                    pass
            self._spawn()
            self._crashed = False
            return "respawned"
        return None
