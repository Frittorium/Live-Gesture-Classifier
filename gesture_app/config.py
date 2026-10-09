import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path

log = logging.getLogger("config")
APP_DIR = Path.home() / ".gesture_app"
CONFIG_PATH = APP_DIR / "config.json"


@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    mirror: bool = True
    model_path: str = "models/hand_landmarker.task"
    confidence_threshold: float = 0.7
    buffer_size: int = 2
    max_read_failures: int = 30
    camera_retries: int = 3
    max_inference_failures: int = 20
    # tracking
    track_max_jump: float = 0.25
    track_timeout_s: float = 0.3
    max_gap_s: float = 0.25
    history_s: float = 1.0
    # recognition
    static_max_speed: float = 0.2
    motion_window_s: float = 0.6
    swipe_min_dx: float = 0.30
    wave_min_dx: float = 0.15
    join_dist: float = 1.1
    clap_open_dist: float = 2.0
    clap_window_s: float = 0.5
    # decision
    static_hold_s: float = 0.15
    release_s: float = 0.3
    cooldown_s: float = 0.8

    def resolved_model_path(self) -> Path:
        p = Path(self.model_path)
        return p if p.is_absolute() else Path(__file__).resolve().parent / p


RANGES = {
    "camera_index": (0, 32), "frame_width": (160, 3840), "frame_height": (120, 2160),
    "confidence_threshold": (0.0, 1.0), "buffer_size": (1, 10),
    "max_read_failures": (1, 1000), "camera_retries": (0, 20), "max_inference_failures": (1, 1000),
    "track_max_jump": (0.01, 1.0), "track_timeout_s": (0.05, 5.0), "max_gap_s": (0.05, 5.0),
    "history_s": (0.2, 5.0), "static_max_speed": (0.0, 5.0), "motion_window_s": (0.2, 3.0),
    "swipe_min_dx": (0.05, 1.0), "wave_min_dx": (0.05, 1.0), "join_dist": (0.2, 5.0),
    "clap_open_dist": (0.5, 10.0), "clap_window_s": (0.1, 3.0), "static_hold_s": (0.0, 3.0),
    "release_s": (0.0, 3.0), "cooldown_s": (0.0, 10.0),
}


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    cfg, defaults = AppConfig(), AppConfig()
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("Config unreadable (%s); using defaults", type(e).__name__)
            raw = {}
        for f in fields(AppConfig):
            if f.name not in raw:
                continue
            v, d = raw[f.name], getattr(defaults, f.name)
            ok = isinstance(v, bool) if isinstance(d, bool) else (
                isinstance(v, (int, float)) and not isinstance(v, bool) if isinstance(d, float)
                else isinstance(v, type(d)) and not isinstance(v, bool) if isinstance(d, int)
                else isinstance(v, type(d)))
            if ok:
                setattr(cfg, f.name, v)
            else:
                log.warning("Invalid type for %s; using default", f.name)
    for name, (lo, hi) in RANGES.items():
        if not lo <= getattr(cfg, name) <= hi:
            log.warning("%s out of range; using default", name)
            setattr(cfg, name, getattr(defaults, name))
    return cfg