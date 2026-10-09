import logging
import time
from collections import deque
from PySide6.QtCore import QThread, Signal
from .decision import DecisionManager
from .domain import ErrorRecord, HandView, Kind, ModelLoadError, ProcessedFrame
from .recognition import GestureRecognitionModule
from .tracking import TrackingModule
from .vision import VisionModule

log = logging.getLogger("processing")


class ProcessingWorker(QThread):
    frame_ready = Signal(object)
    error = Signal(object)

    def __init__(self, cfg, buffer, camera_fps):
        super().__init__()
        self.cfg, self.buffer, self.camera_fps = cfg, buffer, camera_fps

    def run(self):
        try:
            vision = VisionModule(self.cfg)
            tracker = TrackingModule(self.cfg)
            recog = GestureRecognitionModule(self.cfg)
            decision = DecisionManager(self.cfg)
        except ModelLoadError as e:
            self.error.emit(ErrorRecord("vision", "model_load", str(e), False))
            return
        except Exception as e:
            self.error.emit(ErrorRecord("vision", "model_load", f"Init failed: {type(e).__name__}", False))
            return

        stamps, fails = deque(maxlen=30), 0
        try:
            while not self.isInterruptionRequested():
                frame = self.buffer.get_latest(0.1)
                if frame is None:
                    continue
                try:
                    dets = vision.detect(frame)
                    feats = tracker.update(dets, frame.timestamp)
                    cands = recog.evaluate(feats, tracker.histories, frame.timestamp)
                    current, event = decision.process(cands, frame.timestamp)
                    fails = 0
                except Exception as e:
                    fails += 1
                    log.error("Inference failure (%s), count=%d", type(e).__name__, fails)
                    if fails >= self.cfg.max_inference_failures:
                        self.error.emit(ErrorRecord("processing", "inference", "Repeated inference failures", False))
                        return
                    continue

                if event is not None and event.kind == Kind.DYNAMIC:
                    tracker.clear_motion()  # same motion must not retrigger
                ids = set(current.track_ids) if current else set()
                hands = [HandView(f.bbox, f.track_id in ids) for f in feats]
                stamps.append(time.monotonic())
                s = list(stamps)
                pfps = (len(s) - 1) / (s[-1] - s[0]) if len(s) > 1 and s[-1] > s[0] else 0.0
                self.frame_ready.emit(ProcessedFrame(frame.image, hands, current, event, self.camera_fps(), pfps))
        finally:
            vision.close()