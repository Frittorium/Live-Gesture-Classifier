from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class Gesture(str, Enum):
    SWIPE_LEFT = "Swipe Left"
    SWIPE_RIGHT = "Swipe Right"
    PEACE_SIGN = "Peace Sign"
    CLOSE_HAND = "Close Hand"
    OPEN_HAND = "Open Hand"
    WAVE_LEFT_INDEX = "Wave Left with Index Finger Raised"
    WAVE_RIGHT_INDEX = "Wave Right with Index Finger Raised"
    WAVE_LEFT_INDEX_MIDDLE = "Wave Left with Index and Middle Fingers Raised"
    WAVE_RIGHT_INDEX_MIDDLE = "Wave Right with Index and Middle Fingers Raised"
    JOIN_HANDS = "Join Both Hands in a Clap/Prayer Gesture"


class Kind(Enum):
    STATIC = 1
    DYNAMIC = 2
    TWO_HAND = 3


class AppState(Enum):
    INITIALIZING = "Initializing"
    READY = "Ready"
    RUNNING = "Running"
    ERROR = "Error"
    STOPPING = "Stopping"
    STOPPED = "Stopped"


class ModelLoadError(Exception):
    pass


@dataclass
class ErrorRecord:
    component: str
    category: str
    message: str
    recoverable: bool = False


@dataclass
class FrameData:
    image: np.ndarray
    timestamp: float
    seq: int


@dataclass
class HandDetection:
    landmarks: np.ndarray  # (21, 3) normalized
    score: float


@dataclass
class HandFeatures:
    score: float
    ext: np.ndarray  # extension scores [0,1]: thumb, index, middle, ring, pinky
    center: tuple
    bbox: tuple      # normalized x0, y0, x1, y1
    scale: float
    track_id: int = -1


@dataclass
class MotionHistory:
    samples: deque = field(default_factory=deque)  # (t, x, y, finger_mask)

    def add(self, t, x, y, mask, max_age):
        self.samples.append((t, x, y, mask))
        while self.samples and self.samples[0][0] < t - max_age:
            self.samples.popleft()

    def window(self, now, win):
        return [s for s in self.samples if s[0] >= now - win]

    def speed(self, now, win):
        w = self.window(now, win)
        if len(w) < 2 or w[-1][0] - w[0][0] <= 1e-6:
            return 0.0
        return float(np.hypot(w[-1][1] - w[0][1], w[-1][2] - w[0][2]) / (w[-1][0] - w[0][0]))

    def clear(self):
        self.samples.clear()


@dataclass
class GestureCandidate:
    gesture: Gesture
    confidence: float
    kind: Kind
    track_ids: tuple
    meta: dict = field(default_factory=dict)


@dataclass
class RecognitionResult:
    gesture: Gesture
    confidence: float
    timestamp: float
    kind: Kind
    track_ids: tuple
    is_event: bool


@dataclass
class HandView:
    bbox: tuple
    recognized: bool


@dataclass
class ProcessedFrame:
    image: np.ndarray
    hands: list
    current: Optional[RecognitionResult]
    event: Optional[RecognitionResult]
    camera_fps: float
    processing_fps: float