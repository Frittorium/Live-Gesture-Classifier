import sys
from PySide6.QtWidgets import QApplication
from .config import load_config
from .controller import ApplicationController
from .logging_setup import setup_logging
from .ui import MainWindow


def main():
    listener = setup_logging()
    cfg = load_config()
    app = QApplication(sys.argv)
    ctrl = ApplicationController(cfg)
    win = MainWindow(ctrl)
    win.show()
    ctrl.initialize()
    code = app.exec()
    listener.stop()
    sys.exit(code)


if __name__ == "__main__":
    main()