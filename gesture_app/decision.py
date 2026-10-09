from .domain import GestureCandidate, Kind, RecognitionResult

PRIORITY = {Kind.TWO_HAND: 3, Kind.DYNAMIC: 2, Kind.STATIC: 1}


class DecisionManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self._pending = None      # (gesture, since) for static hold validation
        self._active = None       # GestureCandidate currently held
        self._last_seen = 0.0
        self._last_emit = float("-inf")

    def process(self, candidates, now):
        valid = [c for c in candidates
                 if 0.0 <= c.confidence <= 1.0 and c.confidence >= self.cfg.confidence_threshold]
        # deterministic conflict policy: priority, then confidence, then gesture name
        best = max(valid, key=lambda c: (PRIORITY[c.kind], c.confidence, c.gesture.value), default=None)
        if best is None:
            self._pending = None
            return self._current(now), None
        if not self._confirmed(best, now):
            return self._current(now), None

        self._last_seen = now
        is_new = (self._active is None or best.gesture != self._active.gesture
                  or (best.kind == Kind.DYNAMIC and now - self._last_emit >= self.cfg.cooldown_s))
        self._active = best
        event = None
        if is_new:
            self._last_emit = now
            event = RecognitionResult(best.gesture, best.confidence, now, best.kind, best.track_ids, True)
        return self._current(now), event

    def _confirmed(self, c, now):
        if c.kind != Kind.STATIC:
            self._pending = None
            return True
        if self._pending is None or self._pending[0] != c.gesture:
            self._pending = (c.gesture, now)
        return now - self._pending[1] >= self.cfg.static_hold_s

    def _current(self, now):
        a = self._active
        if a is None:
            return None
        if now - self._last_seen > self.cfg.release_s:
            self._active = None  # release -> re-arm
            return None
        return RecognitionResult(a.gesture, a.confidence, now, a.kind, a.track_ids, False)