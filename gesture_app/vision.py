import logging
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from .domain import HandDetection, ModelLoadError

log = logging.getLogger("vision")


class VisionModule:
    def __init__(self, cfg):
        path = cfg.resolved_model_path()
        if not path.is_file():
            raise ModelLoadError(f"Hand landmark model not found: {path}")
        try:
            opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._lm = mp_vision.HandLandmarker.create_from_options(opts)
        except Exception as e:
            raise ModelLoadError(f"Model failed to load: {type(e).__name__}") from e
        self._last_ms = -1

    def detect(self, frame):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
        ts = int(frame.timestamp * 1000)
        ts = max(ts, self._last_ms + 1)
        self._last_ms = ts
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        out = []
        for lms, handed in zip(res.hand_landmarks, res.handedness):
            arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            score = float(handed[0].score)
            if arr.shape != (21, 3) or not np.isfinite(arr).all() or not 0.0 <= score <= 1.0:
                continue  # invalid output rejected
            out.append(HandDetection(arr, score))
        return out

    def close(self):
        self._lm.close()