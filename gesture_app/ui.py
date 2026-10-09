from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QVBoxLayout, QWidget)


class MainWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.ctrl = controller
        self.setWindowTitle("Offline Gesture Recognition")

        self.preview = QLabel("Camera stopped")
        self.preview.setMinimumSize(640, 480)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#111;color:#aaa;")

        self.vals = {}
        grid = QGridLayout()
        for i, name in enumerate(["Last Gesture", "Current Gesture", "Confidence", "Camera FPS", "Processing FPS"]):
            grid.addWidget(QLabel(name + ":"), i, 0)
            v = QLabel("-")
            v.setStyleSheet("font-weight:bold;")
            grid.addWidget(v, i, 1)
            self.vals[name] = v

        self.start_btn, self.stop_btn = QPushButton("Start"), QPushButton("Stop")
        self.status = QLabel("")
        self.start_btn.clicked.connect(self.ctrl.start)
        self.stop_btn.clicked.connect(self.ctrl.stop)
        btns = QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)

        side = QVBoxLayout()
        side.addLayout(grid)
        side.addLayout(btns)
        side.addWidget(self.status)
        side.addStretch()
        root = QHBoxLayout()
        root.addWidget(self.preview, 1)
        root.addLayout(side)
        w = QWidget()
        w.setLayout(root)
        self.setCentralWidget(w)

        self.ctrl.frame_ready.connect(self.on_frame)
        self.ctrl.state_changed.connect(self.on_state)
        self.ctrl.error_raised.connect(lambda m: self.status.setText(f"Error: {m}"))
        self.on_state(self.ctrl.state.value)

    def on_state(self, s):
        self.start_btn.setEnabled(s in ("Ready", "Stopped"))
        self.stop_btn.setEnabled(s == "Running")
        if s in ("Ready", "Running", "Stopped"):
            self.status.setText(s)
        if s == "Stopped":
            self.preview.clear()
            self.preview.setText("Camera stopped")

    def on_frame(self, pf):
        h, w = pf.image.shape[:2]
        img = QImage(pf.image.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        p = QPainter(img)
        p.setFont(QFont("Sans", 12, QFont.Bold))
        for hand in pf.hands:
            color = QColor("#2ecc71") if hand.recognized else QColor("#8c8c8c")
            p.setPen(QPen(color, 3))
            x0, y0, x1, y1 = hand.bbox
            rect = QRect(int(x0 * w), int(y0 * h), int((x1 - x0) * w), int((y1 - y0) * h))
            p.drawRect(rect)
            if hand.recognized and pf.current:
                p.drawText(rect.x(), max(16, rect.y() - 6), pf.current.gesture.value)
        p.end()
        self.preview.setPixmap(QPixmap.fromImage(img).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if pf.event:
            self.vals["Last Gesture"].setText(pf.event.gesture.value)
        self.vals["Current Gesture"].setText(pf.current.gesture.value if pf.current else "-")
        self.vals["Confidence"].setText(f"{pf.current.confidence:.2f}" if pf.current else "-")
        self.vals["Camera FPS"].setText(f"{pf.camera_fps:.1f}")
        self.vals["Processing FPS"].setText(f"{pf.processing_fps:.1f}")

    def closeEvent(self, e):
        self.ctrl.stop()
        super().closeEvent(e)