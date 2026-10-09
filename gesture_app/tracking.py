import numpy as np
from .domain import HandFeatures, MotionHistory


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class TrackingModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.histories = {}
        self._last = {}  # id -> (time, center)
        self._next = 0

    @staticmethod
    def _features(det):
        xy = det.landmarks[:, :2]

        def d(a, b):
            return float(np.linalg.norm(xy[a] - xy[b]))

        ext = np.empty(5, dtype=np.float32)
        ext[0] = np.clip((d(4, 17) / max(d(3, 17), 1e-6) - 1.0) / 0.25, 0, 1)
        for i, (tip, pip) in enumerate(((8, 6), (12, 10), (16, 14), (20, 18)), 1):
            ext[i] = np.clip((d(tip, 0) / max(d(pip, 0), 1e-6) - 0.95) / 0.25, 0, 1)
        c = xy[[0, 5, 9, 13, 17]].mean(axis=0)
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        pad = 0.1 * max(x1 - x0, y1 - y0)
        bbox = (float(max(0, x0 - pad)), float(max(0, y0 - pad)), float(min(1, x1 + pad)), float(min(1, y1 + pad)))
        return HandFeatures(det.score, ext, (float(c[0]), float(c[1])), bbox, max(d(0, 9), 1e-3))

    def update(self, dets, now):
        for tid in [t for t, (ts, _) in self._last.items() if now - ts > self.cfg.track_timeout_s]:
            del self._last[tid]
            self.histories.pop(tid, None)
        free, out = set(self._last), []
        for f in sorted((self._features(d) for d in dets), key=lambda f: -f.score):
            best = min(free, key=lambda t: _dist(self._last[t][1], f.center), default=None)
            if best is not None and _dist(self._last[best][1], f.center) <= self.cfg.track_max_jump:
                tid = best
                free.discard(best)
            else:
                tid, self._next = self._next, self._next + 1
            f.track_id = tid
            self._last[tid] = (now, f.center)
            h = self.histories.setdefault(tid, MotionHistory())
            if h.samples and now - h.samples[-1][0] > self.cfg.max_gap_s:
                h.clear()  # interrupted tracking is not valid movement evidence
            h.add(now, f.center[0], f.center[1], tuple(bool(v > 0.5) for v in f.ext[1:]), self.cfg.history_s)
            out.append(f)
        return out

    def clear_motion(self):
        for h in self.histories.values():
            h.clear()   