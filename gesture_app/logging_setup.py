import logging
import queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from .config import APP_DIR


def setup_logging():
    """Non-blocking logging. Never log frames or landmarks."""
    logging.raiseExceptions = False
    log_dir = APP_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    q = queue.Queue(-1)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(QueueHandler(q))
    listener = QueueListener(q, handler)
    listener.start()
    return listener