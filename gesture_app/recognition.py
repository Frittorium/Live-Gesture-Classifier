from collections import deque
import numpy as np
from .domain import Gesture, GestureCandidate, Kind


def pattern_conf(f, want):
    """want: 4 bools (index..pinky) = must be extended."""
    return float(min((e if w else 1 - e) for e, w in zip(f.ext[1:], want)) * f.score)


class StaticPoseRecognizer:
    PATTERNS = {
        Gesture.OPEN_HAND: (True, True, True, True),
        Gesture.CLOSE_HAND: (False, False, False, False),
        Gesture.PEACE_SIGN: (True, True, False, False),
    }

    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(self, feats, hist, now):
        out = []
        for f in feats:
            if hist[f.track_id].speed(now, 0.3) > self.cfg.static_max_speed:
                continue  # moving hands are handled by the temporal recognizer
            for g, want in self.PATTERNS.items():
                c = pattern_conf(f, want)
                if c > 0:
                    out.append(GestureCandidate(g, c, Kind.STATIC, (f.track_id,)))
        return out


class TemporalMovementRecognizer:
    def __init__(self, cfg):
        self.cfg = cfg
        T, F = True, False
        self.specs = [
            ((T, T, T, T), cfg.swipe_min_dx, Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT),
            ((T, F, F, F), cfg.wave_min_dx, Gesture.WAVE_LEFT_INDEX, Gesture.WAVE_RIGHT_INDEX),
            ((T, T, F, F), cfg.wave_min_dx, Gesture.WAVE_LEFT_INDEX_MIDDLE, Gesture.WAVE_RIGHT_INDEX_MIDDLE),
        ]

    def evaluate(self, feats, hist, now):
        out = []
        for f in feats:
            w = hist[f.track_id].window(now, self.cfg.motion_window_s)
            if len(w) < 4:
                continue
            xs = np.array([s[1] for s in w])
            ys = np.array([s[2] for s in w])
            dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
            if abs(dx) < 2 * abs(dy):
                continue
            if (np.sign(dx) * np.diff(xs) >= -0.004).mean() < 0.8:
                continue  # not a consistent sweep
            for want, min_dx, left, right in self.specs:
                if abs(dx) < min_dx:
                    continue
                if np.mean([s[3] == want for s in w]) < 0.8:
                    continue
                conf = pattern_conf(f, want) * (0.6 + 0.4 * min(1.0, abs(dx) / (2 * min_dx)))
                out.append(GestureCandidate(left if dx < 0 else right, conf, Kind.DYNAMIC, (f.track_id,)))
        return out


class TwoHandRecognizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._hist = deque()

    def evaluate(self, feats, hist, now):
        if len(feats) < 2:
            self._hist.clear()
            return []
        a, b = sorted(feats, key=lambda f: -f.score)[:2]
        dist = _hand_dist(a, b)
        self._hist.append((now, dist))
        while self._hist[0][0] < now - self.cfg.clap_window_s:
            self._hist.popleft()
        if dist >= self.cfg.join_dist:
            return []
        clap = max(d for _, d in self._hist) >= self.cfg.clap_open_dist
        prayer = all(float(np.mean(f.ext[1:])) > 0.6 for f in (a, b))
        if not (clap or prayer):
            return []
        conf = (0.75 + 0.25 * (1 - dist / self.cfg.join_dist)) * min(a.score, b.score)
        # prayer and clap intentionally share one external identifier
        return [GestureCandidate(Gesture.JOIN_HANDS, conf, Kind.TWO_HAND, (a.track_id, b.track_id),
                                 {"pattern": "clap" if clap else "prayer"})]


def _hand_dist(a, b):
    scale = (a.scale + b.scale) / 2
    return float(np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) / scale)


class GestureRecognitionModule:
    def __init__(self, cfg):
        self.strategies = [StaticPoseRecognizer(cfg), TemporalMovementRecognizer(cfg), TwoHandRecognizer(cfg)]

    def evaluate(self, feats, hist, now):
        out = []
        for s in self.strategies:
            out.extend(s.evaluate(feats, hist, now))
        return out