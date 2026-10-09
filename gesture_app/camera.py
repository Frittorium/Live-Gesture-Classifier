import logging
import threading
import time
from collections import deque
import cv2
from PySide6.QtCore import QThread, Signal
from .domain import ErrorRecord, FrameData

log = logging.getLogger("camera")


class LatestFrameBuffer:
    """Bounded buffer, latest-frame-wins."""
    def __init__(self, size=2):
        self._q = deque(maxlen=size)  # oldest dropped automatically
        self._cv = threading.Condition()
        self._closed = False

    def put(self, frame):
        with self._cv:
            self._q.append(frame)
            self._cv.notify()

    def get_latest(self, timeout=0.2):
        with self._cv:
            if not self._q and not self._closed:
                self._cv.wait(timeout)
            if not self._q:
                return None
            frame = self._q.pop()
            self._q.clear()  # discard obsolete frames
            return frame

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class CameraWorker(QThread):
    error = Signal(object)

    def __init__(self, cfg, buffer):
        super().__init__()
        self.cfg, self.buffer = cfg, buffer
        self._stamps = deque(maxlen=30)

    @property
    def fps(self):
        s = list(self._stamps)
        return (len(s) - 1) / (s[-1] - s[0]) if len(s) > 1 and s[-1] > s[0] else 0.0

    def _open(self):
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
            return cap
        cap.release()
        return None

    def run(self):
        cap = self._open()
        if cap is None:
            self.error.emit(ErrorRecord("camera", "camera_unavailable", "Camera unavailable", False))
            return
        fails = retries = seq = 0
        try:
            while not self.isInterruptionRequested():
                if cap is None:
                    if retries >= self.cfg.camera_retries:
                        self.error.emit(ErrorRecord("camera", "capture_failed", "Camera lost; retries exhausted", False))
                        return
                    retries += 1
                    self.msleep(500)
                    cap, fails = self._open(), 0
                    continue
                ok, img = cap.read()
                if not ok or img is None:
                    fails += 1
                    if fails >= self.cfg.max_read_failures:
                        cap.release()
                        cap = None
                        self.error.emit(ErrorRecord("camera", "capture_interrupted", "Capture interrupted; retrying", True))
                    else:
                        self.msleep(20)
                    continue
                fails = retries = 0
                if self.cfg.mirror:
                    img = cv2.flip(img, 1)
                now = time.monotonic()
                self._stamps.append(now)
                seq += 1
                self.buffer.put(FrameData(img, now, seq))
        finally:
            if cap is not None:
                cap.release()