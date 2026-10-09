import logging
from PySide6.QtCore import QObject, Signal
from .camera import CameraWorker, LatestFrameBuffer
from .domain import AppState
from .processing import ProcessingWorker

log = logging.getLogger("controller")


class ApplicationController(QObject):
    frame_ready = Signal(object)
    state_changed = Signal(str)
    error_raised = Signal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.state = AppState.INITIALIZING
        self.camera = self.processing = self._buffer = None

    def _set_state(self, s):
        self.state = s
        log.info("State -> %s", s.value)
        self.state_changed.emit(s.value)

    def initialize(self):
        self._set_state(AppState.INITIALIZING)
        if not self.cfg.resolved_model_path().is_file():
            self.error_raised.emit(f"Model file missing: {self.cfg.resolved_model_path()}")
            self._set_state(AppState.ERROR)
            return
        self._set_state(AppState.READY)

    def start(self):
        if self.state not in (AppState.READY, AppState.STOPPED):
            return
        self._buffer = LatestFrameBuffer(self.cfg.buffer_size)
        cam = self.camera = CameraWorker(self.cfg, self._buffer)
        self.processing = ProcessingWorker(self.cfg, self._buffer, lambda: cam.fps)
        for w in (self.camera, self.processing):
            w.error.connect(self._on_error)
        self.processing.frame_ready.connect(self.frame_ready)
        self.processing.start()
        self.camera.start()
        self._set_state(AppState.RUNNING)

    def stop(self):
        if self.state == AppState.RUNNING:
            self._shutdown()

    def _shutdown(self):
        self._set_state(AppState.STOPPING)
        workers = [w for w in (self.camera, self.processing) if w is not None]
        for w in workers:
            w.requestInterruption()
        if self._buffer:
            self._buffer.close()
        for w in workers:
            if not w.wait(3000):
                log.error("Worker %s did not stop in time", type(w).__name__)
        self.camera = self.processing = self._buffer = None
        self._set_state(AppState.STOPPED)

    def _on_error(self, rec):
        log.error("%s/%s: %s (recoverable=%s)", rec.component, rec.category, rec.message, rec.recoverable)
        self.error_raised.emit(rec.message)
        if not rec.recoverable and self.state == AppState.RUNNING:
            self._shutdown()
            self._set_state(AppState.ERROR)