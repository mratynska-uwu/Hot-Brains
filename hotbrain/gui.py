"""PySide6 operator interface and stimulus timing state machine."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImageReader, QKeySequence, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox, QTabWidget, QVBoxLayout,
    QWidget, QTableWidget, QTableWidgetItem, QToolTip,
)

from .acquisition import DeviceManager
from .features import CHANNELS, DEFAULT_EPOCH, EpochSpec, FEATURE_NAMES, FS, START_MS, extract
from .quality import ChannelQuality, assess_signal_quality
from .training import OnlineTrainer, TrialPlan, image_files


class Stimulus(QWidget):
    onset = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(520, 380)
        self.state = "idle"
        self.pixmap = QPixmap()
        self._emit_on_paint = False
        self.setAutoFillBackground(False)

    def fixation(self):
        self.state = "fixation"
        self._emit_on_paint = False
        self.update()

    def image(self, path: Path):
        self.pixmap = QPixmap(str(path))
        if self.pixmap.isNull():
            raise ValueError(f"Cannot display image: {path.name}")
        self.state = "image"
        self._emit_on_paint = True
        self.update()

    def idle(self):
        self.state = "idle"
        self._emit_on_paint = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101827"))
        if self.state == "image":
            size = self.pixmap.size()
            size.scale(self.rect().adjusted(24, 24, -24, -24).size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = self.rect()
            target.setSize(size)
            target.moveCenter(self.rect().center())
            painter.drawPixmap(target, self.pixmap)
        else:
            painter.setPen(QColor("#eaf1ff"))
            font = QFont("Arial", 52 if self.state == "fixation" else 22)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "+" if self.state == "fixation" else "Ready for training")
        painter.end()
        if self._emit_on_paint:
            self._emit_on_paint = False
            self.onset.emit(time.perf_counter())


class PresentationPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        heading = QLabel("Stimulus presentation")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        self.stimulus = Stimulus()
        layout.addWidget(self.stimulus, 1)
        self.note = QLabel()
        self.note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.note)
        self.set_timing(.6, 1.0, DEFAULT_EPOCH)

    def set_timing(self, fixation_min: float, fixation_max: float, spec: EpochSpec):
        self.note.setText(f"Fixation: {fixation_min:g}–{fixation_max:g} s   ·   Image: {spec.image_duration_ms / 1000:g} s   ·   EEG epoch: {START_MS} to +{spec.end_ms} ms")


class TrainingPanel(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    save_requested = Signal()
    discard_requested = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        heading = QLabel("Scene valence training")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        form = QFormLayout()
        self.participant = QLineEdit("debug")
        self.participant.setPlaceholderText("Participant ID")
        form.addRow("Participant ID", self.participant)
        self.serial = QLineEdit("UN-2019.03.01")
        self.serial.setPlaceholderText("Blank = first available Unicorn")
        form.addRow("Unicorn serial", self.serial)
        self.count = QSpinBox()
        self.count.setRange(2, 100000)
        self.count.setSingleStep(2)
        self.count.setValue(160)
        form.addRow("Presentations", self.count)
        self.persistent = QCheckBox("Continue until manually stopped")
        self.persistent.toggled.connect(self.count.setDisabled)
        form.addRow("Mode", self.persistent)
        self.image_seconds = QDoubleSpinBox()
        self.image_seconds.setRange(.7, 60.0)
        self.image_seconds.setDecimals(2)
        self.image_seconds.setSingleStep(.1)
        self.image_seconds.setSuffix(" s")
        self.image_seconds.setValue(1.0)
        form.addRow("Image duration", self.image_seconds)
        self.fixation_min = QDoubleSpinBox()
        self.fixation_min.setRange(.2, 60.0)
        self.fixation_min.setDecimals(2)
        self.fixation_min.setSingleStep(.1)
        self.fixation_min.setSuffix(" s")
        self.fixation_min.setValue(.6)
        form.addRow("Fixation minimum", self.fixation_min)
        self.fixation_max = QDoubleSpinBox()
        self.fixation_max.setRange(.2, 60.0)
        self.fixation_max.setDecimals(2)
        self.fixation_max.setSingleStep(.1)
        self.fixation_max.setSuffix(" s")
        self.fixation_max.setValue(1.0)
        form.addRow("Fixation maximum", self.fixation_max)
        self.stop_key = QComboBox()
        self.stop_key.addItems(["Escape", "Space", "Enter", *[f"F{i}" for i in range(1, 13)],
                                *"ABCDEFGHIJKLMNOPQRSTUVWXYZ", *"0123456789"])
        form.addRow("Stop key", self.stop_key)
        self.oscar = QCheckBox("Enable g.Pype OSCAR artifact removal")
        self.oscar.setChecked(True)
        form.addRow("EEG processing", self.oscar)
        self.warmup = QCheckBox("Allow OSCAR to settle before trials")
        self.warmup.setChecked(True)
        form.addRow("Before trials", self.warmup)
        self.warmup_seconds = QSpinBox()
        self.warmup_seconds.setRange(2, 120)
        self.warmup_seconds.setValue(12)
        self.warmup_seconds.setSuffix(" s")
        form.addRow("Settling duration", self.warmup_seconds)
        layout.addLayout(form)
        self.detail = QLabel("Finite runs are exactly balanced and shuffled; continuous runs draw each label independently at 50%. OSCAR adapts online; g.Pype does not expose a saved calibration state.")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        buttons = QHBoxLayout()
        self.start = QPushButton("Start training")
        self.stop = QPushButton("Stop training")
        self.save = QPushButton("Save fitted model")
        self.discard = QPushButton("Discard run")
        self.stop.setEnabled(False)
        self.save.setEnabled(False)
        self.discard.setEnabled(False)
        for button in (self.start, self.stop, self.save, self.discard):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.stop_hint = QLabel("Press Escape in any app window to stop training.")
        self.stop_key.currentTextChanged.connect(
            lambda key: self.stop_hint.setText(f"Press {key} in any app window to stop training."))
        layout.addWidget(self.stop_hint)
        self.start.clicked.connect(self.start_requested)
        self.stop.clicked.connect(self.stop_requested)
        self.save.clicked.connect(self.save_requested)
        self.discard.clicked.connect(self.discard_requested)
        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        layout.addWidget(self.status)
        self.device_status = QLabel("Checking for Unicorn…")
        self.device_status.setWordWrap(True)
        layout.addWidget(self.device_status)
        self.count_status = QLabel("Trials: 0 · usable EEG: 0 · held-out predictions: 0")
        layout.addWidget(self.count_status)
        self.accuracy = QLabel("Last 20 held-out predictions: —")
        self.accuracy.setObjectName("metric")
        layout.addWidget(self.accuracy)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#101827")
        self.plot.setLabel("left", "Correct", units="%")
        self.plot.setLabel("bottom", "Presented trial")
        self.plot.setYRange(0, 100)
        self.plot.showGrid(x=True, y=True, alpha=.2)
        self.curve = self.plot.plot(pen=pg.mkPen("#6dd7bf", width=3), symbol="o", symbolSize=6)
        layout.addWidget(self.plot, 1)
        self.weights = QTableWidget(len(FEATURE_NAMES), 2)
        self.weights.setHorizontalHeaderLabels(("Feature", "Standardized weight"))
        self.weights.horizontalHeader().setStretchLastSection(True)
        self.weights.setColumnWidth(0, 370)
        self.weights.setMaximumHeight(195)
        for row, name in enumerate(FEATURE_NAMES):
            self.weights.setItem(row, 0, QTableWidgetItem(name))
            self.weights.setItem(row, 1, QTableWidgetItem("—"))
        layout.addWidget(self.weights)
        self.x: list[int] = []
        self.y: list[float] = []

    def reset_plot(self):
        self.x.clear(); self.y.clear(); self.curve.setData([], [])

    def add_accuracy(self, trial: int, value: float):
        self.x.append(trial); self.y.append(value)
        self.curve.setData(self.x, self.y)

    def update_weights(self, trainer: OnlineTrainer):
        coefficients = trainer.model.steps[1][1].coef_[0] if trainer.model is not None else None
        for row in range(len(FEATURE_NAMES)):
            self.weights.item(row, 1).setText(f"{coefficients[row]:+.3f}" if coefficients is not None else "—")


class ElectrodeMap(QWidget):
    """Top-view brain diagram with eight live channel-status lights."""

    POSITIONS = ((200, 69), (138, 139), (200, 141), (262, 139),
                 (200, 202), (125, 235), (200, 255), (275, 235))

    def __init__(self):
        super().__init__()
        self.ratings = [ChannelQuality(None, "Waiting for EEG") for _ in CHANNELS]
        self.setMinimumHeight(260)
        self.setMaximumHeight(310)
        self.setMouseTracking(True)

    def set_quality(self, ratings: list[ChannelQuality] | None):
        self.ratings = ratings if ratings is not None else [
            ChannelQuality(None, "Waiting for EEG") for _ in CHANNELS]
        self.update()

    def _transform(self):
        scale = min(self.width() / 400, self.height() / 300)
        return scale, (self.width() - 400 * scale) / 2, (self.height() - 300 * scale) / 2

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        scale, dx, dy = self._transform()
        painter.translate(dx, dy)
        painter.scale(scale, scale)

        # Scalp outline and nose establish orientation; the brain sits below
        # the eight channel markers, which remain the most prominent elements.
        painter.setPen(QPen(QColor("#8292aa"), 3))
        painter.setBrush(QColor("#142638"))
        painter.drawEllipse(QRectF(76, 23, 248, 258))
        nose = QPainterPath(QPointF(183, 28))
        nose.lineTo(200, 7)
        nose.lineTo(217, 28)
        painter.setBrush(QColor("#142638"))
        painter.drawPath(nose)

        brain = QPainterPath(QPointF(200, 58))
        brain.cubicTo(161, 35, 127, 56, 121, 84)
        brain.cubicTo(91, 83, 85, 111, 100, 136)
        brain.cubicTo(78, 157, 91, 188, 111, 199)
        brain.cubicTo(110, 231, 142, 253, 171, 238)
        brain.cubicTo(185, 257, 201, 247, 200, 237)
        brain.cubicTo(199, 247, 215, 257, 229, 238)
        brain.cubicTo(258, 253, 290, 231, 289, 199)
        brain.cubicTo(309, 188, 322, 157, 300, 136)
        brain.cubicTo(315, 111, 309, 83, 279, 84)
        brain.cubicTo(273, 56, 239, 35, 200, 58)
        brain.closeSubpath()
        fill = QLinearGradient(115, 65, 285, 248)
        fill.setColorAt(0, QColor("#a75d71"))
        fill.setColorAt(1, QColor("#744b79"))
        painter.setBrush(fill)
        painter.setPen(QPen(QColor("#e29ba1"), 2))
        painter.drawPath(brain)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#633e62"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for points in (((200, 60), (194, 95), (205, 127), (198, 160), (201, 235)),
                       ((126, 103), (159, 90), (171, 117), (150, 135)),
                       ((114, 170), (148, 152), (163, 178), (145, 203)),
                       ((274, 103), (241, 90), (229, 117), (250, 135)),
                       ((286, 170), (252, 152), (237, 178), (255, 203))):
            fold = QPainterPath(QPointF(*points[0]))
            for x, y in points[1:]:
                fold.lineTo(x, y)
            painter.drawPath(fold)

        painter.setFont(QFont("Arial", 11, QFont.Weight.DemiBold))
        for (x, y), name, rating in zip(self.POSITIONS, CHANNELS, self.ratings, strict=True):
            color = QColor({True: "#6dd7bf", False: "#ff727a", None: "#a3acbd"}[rating.good])
            halo = QColor(color)
            halo.setAlpha(105)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(QPointF(x, y), 17, 17)
            painter.setPen(QPen(QColor("#0b1a2a"), 3))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x, y), 10, 10)
            painter.setPen(QColor("#f5f7ff"))
            painter.drawText(QRectF(x - 30, y + 12, 60, 20), Qt.AlignmentFlag.AlignCenter, name)
        painter.end()

    def mouseMoveEvent(self, event):
        scale, dx, dy = self._transform()
        x, y = (event.position().x() - dx) / scale, (event.position().y() - dy) / scale
        for (cx, cy), name, rating in zip(self.POSITIONS, CHANNELS, self.ratings, strict=True):
            if (x - cx) ** 2 + (y - cy) ** 2 <= 18 ** 2:
                QToolTip.showText(event.globalPosition().toPoint(), f"{name}: {rating.detail}", self)
                return
        QToolTip.hideText()

    def leaveEvent(self, event):
        QToolTip.hideText()


class FeaturePanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        heading = QLabel("EEG features over time")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        self.note = QLabel("Waiting for a Unicorn signal. Values are extracted from rolling EEG windows; no synthetic EEG is shown.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        quality_title = QLabel("Electrode signal checks")
        quality_title.setObjectName("heading")
        layout.addWidget(quality_title)
        self.quality_map = ElectrodeMap()
        layout.addWidget(self.quality_map)
        quality_note = QLabel("Top = forehead. Green = recent amplitude looks plausible; red = flat or unusually large; grey = waiting. Hover over an electrode for measurements. This checks the streamed signal (after OSCAR if enabled), not electrode impedance or guaranteed scalp contact. Allow 2 s after connecting.")
        quality_note.setWordWrap(True)
        layout.addWidget(quality_note)
        self.set_quality(None)
        self.selector = QComboBox()
        self.selector.addItems(FEATURE_NAMES)
        layout.addWidget(self.selector)
        self.current = QLabel("Current value: —")
        layout.addWidget(self.current)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#101827")
        self.plot.setLabel("bottom", "Seconds since plot start")
        self.plot.showGrid(x=True, y=True, alpha=.2)
        self.curve = self.plot.plot(pen=pg.mkPen("#8bb9ff", width=2))
        layout.addWidget(self.plot, 1)
        self.samples: list[tuple[float, np.ndarray]] = []
        self.selector.currentIndexChanged.connect(self.redraw)

    def set_quality(self, ratings: list[ChannelQuality] | None):
        self.quality_map.set_quality(ratings)

    def add(self, stamp: float, values: np.ndarray):
        self.samples.append((stamp, values))
        self.samples = [item for item in self.samples if item[0] >= stamp - 60]
        self.redraw()

    def redraw(self):
        if not self.samples:
            self.curve.setData([], []); return
        index = self.selector.currentIndex()
        start = self.samples[0][0]
        self.curve.setData([t - start for t, _ in self.samples], [float(v[index]) for _, v in self.samples])
        self.current.setText(f"Current value: {self.samples[-1][1][index]:.3f}")


class DetachedWindow(QMainWindow):
    def __init__(self, title: str, panel: QWidget, on_close):
        super().__init__()
        self.setWindowTitle(title)
        self.setCentralWidget(panel)
        self.resize(900, 650)
        self._on_close = on_close

    def closeEvent(self, event):
        panel = self.takeCentralWidget()
        self._on_close(panel)
        event.accept()


@dataclass
class PendingEpoch:
    number: int
    label: str
    path: Path
    onset: float
    deadline: float


class MainWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.setWindowTitle("Hot Brain · scene training")
        self.resize(1120, 780)
        self.training = TrainingPanel()
        self.device = DeviceManager(self.training.serial.text().strip() or None)
        self.presentation = PresentationPanel()
        self.epoch_spec = DEFAULT_EPOCH
        for control in (self.training.image_seconds, self.training.fixation_min, self.training.fixation_max):
            control.valueChanged.connect(self.update_timing_note)
        self.features = FeaturePanel()
        self.panels = [self.training, self.presentation, self.features]
        self.titles = ["Training progress", "Image presentation", "EEG features"]
        self.detached: dict[int, DetachedWindow] = {}
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        for index, panel in enumerate(self.panels):
            self._insert_tab(index, panel)
        self.training.start_requested.connect(self.start_run)
        self.training.stop_requested.connect(self.stop_run)
        self.stop_shortcut = QShortcut(QKeySequence(self.training.stop_key.currentText()), self)
        self.stop_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.stop_shortcut.setEnabled(False)
        self.stop_shortcut.activated.connect(self.stop_run)
        self.training.stop_key.currentTextChanged.connect(
            lambda key: self.stop_shortcut.setKey(QKeySequence(key)))
        self.training.save_requested.connect(self.save_model)
        self.training.discard_requested.connect(self.discard_run)
        self.presentation.stimulus.onset.connect(self.on_image_onset)
        self.device.status_changed.connect(self.training.device_status.setText)
        self.device.connected_changed.connect(self.on_connection)
        self.plan: TrialPlan | None = None
        self.trainer = OnlineTrainer()
        self.pending: list[PendingEpoch] = []
        self.next_trial: tuple[str, Path] | None = None
        self.active = False
        self.finishing = False
        self.shown = 0
        self.usable = 0
        self.skipped = 0
        self.last_feature_time = 0.0
        self.last_quality_time = 0.0
        self.run_metadata: dict = {}
        self.trial_log: list[dict] = []
        self.epochs: list[np.ndarray] = []
        self.epoch_trial_numbers: list[int] = []
        self.run_start_clock = 0.0
        self.clock = QTimer(self)
        self.clock.setInterval(100)
        self.clock.timeout.connect(self.poll)
        self.clock.start()
        self.setStyleSheet("QMainWindow,QWidget{background:#182335;color:#e8effc;font-size:13px} QLabel#heading{font-size:24px;font-weight:700;margin:4px 0 10px} QLabel#metric{font-size:18px;font-weight:600;color:#6dd7bf} QLabel#status{font-size:16px;font-weight:600} QPushButton{background:#29466c;border-radius:7px;padding:8px 12px} QPushButton:hover{background:#376293} QPushButton:disabled{background:#354155;color:#8d9aaf} QSpinBox,QDoubleSpinBox,QComboBox{background:#25364e;padding:5px;border:1px solid #627795} QTabBar::tab{padding:9px 17px}")

    def update_timing_note(self):
        spec = EpochSpec(round(self.training.image_seconds.value() * 1000))
        self.presentation.set_timing(self.training.fixation_min.value(), self.training.fixation_max.value(), spec)

    def _insert_tab(self, index: int, panel: QWidget):
        wrapper = QWidget()
        box = QVBoxLayout(wrapper)
        row = QHBoxLayout()
        row.addStretch()
        detach = QPushButton("Open in window ↗")
        detach.clicked.connect(lambda _=False, i=index: self.detach(i))
        row.addWidget(detach)
        box.addLayout(row)
        box.addWidget(panel, 1)
        self.tabs.insertTab(index, wrapper, self.titles[index])

    def detach(self, index: int):
        if index in self.detached:
            self.detached[index].raise_(); return
        wrapper = self.panels[index].parentWidget()
        tab_index = self.tabs.indexOf(wrapper)
        if tab_index < 0:
            return
        self.tabs.removeTab(tab_index)
        self.panels[index].setParent(None)
        window = DetachedWindow(self.titles[index], self.panels[index], lambda panel, i=index: self.reattach(i, panel))
        self.detached[index] = window
        window.show()

    def reattach(self, index: int, panel: QWidget):
        self.detached.pop(index, None)
        self._insert_tab(min(index, self.tabs.count()), panel)
        if index == 1 and self.active:
            self.tabs.setCurrentWidget(panel.parentWidget())

    def on_connection(self, connected: bool):
        self.features.note.setText("Live Unicorn EEG · 250 Hz · OSCAR " + ("on" if self.device.oscar else "off") if connected else "Waiting for Unicorn; continuing image loop without EEG.")
        if not connected:
            self.features.set_quality(None)

    def start_run(self):
        fixation_min = self.training.fixation_min.value()
        fixation_max = self.training.fixation_max.value()
        if fixation_min > fixation_max:
            QMessageBox.warning(self, "Fixation timing", "Fixation minimum must be at most the maximum.")
            return
        self.epoch_spec = EpochSpec(round(self.training.image_seconds.value() * 1000))
        self.device.set_retention(self.training.image_seconds.value() + 5)
        images = image_files(self.root)
        for paths in images.values():
            for path in paths:
                if not QImageReader(str(path)).canRead():
                    QMessageBox.warning(self, "Image error", f"Cannot read {path}")
                    return
        total = None if self.training.persistent.isChecked() else self.training.count.value()
        if total is not None and total % 2:
            QMessageBox.warning(self, "Balanced trials", "Choose an even presentation count.")
            return
        try:
            self.plan = TrialPlan(images, total)
        except ValueError as exc:
            QMessageBox.warning(self, "Training images", str(exc)); return
        self.trainer = OnlineTrainer()
        self.pending.clear()
        self.shown = self.usable = self.skipped = 0
        self.training.reset_plot()
        self.training.update_weights(self.trainer)
        self.training.accuracy.setText("Last 20 held-out predictions: —")
        self.training.start.setEnabled(False)
        self.training.stop.setEnabled(True)
        self.stop_shortcut.setEnabled(True)
        self.training.save.setEnabled(False)
        self.training.discard.setEnabled(False)
        for widget in (self.training.participant, self.training.serial, self.training.count, self.training.persistent, self.training.stop_key, self.training.oscar,
                       self.training.warmup, self.training.warmup_seconds, self.training.image_seconds, self.training.fixation_min, self.training.fixation_max):
            widget.setEnabled(False)
        self.active = True
        self.finishing = False
        self.trial_log = []
        self.epochs = []
        self.epoch_trial_numbers = []
        self.run_start_clock = time.perf_counter()
        # A hidden tab does not paint, so give the participant a visible window.
        if 1 not in self.detached:
            self.detach(1)
        self.detached[1].raise_()
        self.run_metadata = {"started_at": datetime.now().astimezone().isoformat(),
                             "participant_id": self.training.participant.text().strip() or "unnamed",
                             "unicorn_serial": self.training.serial.text().strip() or None,
                             "mode": "persistent" if total is None else "balanced_finite",
                             "requested_trials": total, "oscar_enabled": self.training.oscar.isChecked(),
                             "image_duration_s": self.epoch_spec.image_duration_ms / 1000,
                             "fixation_min_s": fixation_min, "fixation_max_s": fixation_max,
                             "epoch_start_ms": START_MS, "epoch_end_ms": self.epoch_spec.end_ms,
                             "epoch_sample_count": self.epoch_spec.n_samples,
                             "oscar_settling_requested_seconds": self.training.warmup_seconds.value() if self.training.warmup.isChecked() else 0,
                             "oscar_settling_applied_seconds": 0,
                             "dataset": {k: [p.name for p in v] for k, v in images.items()},
                             "timing_note": "Image onset uses Qt paint completion; EEG times anchor to first host frame arrival minus device/OSCAR delay then advance at 250 Hz. Validate with hardware trigger."}
        self.device.configure(self.training.oscar.isChecked(), self.training.serial.text().strip() or None)
        if self.training.oscar.isChecked() and self.training.warmup.isChecked() and self.device.connected:
            seconds = self.training.warmup_seconds.value()
            self.run_metadata["oscar_settling_applied_seconds"] = seconds
            self.training.status.setText(f"OSCAR settling: {seconds} s before first trial")
            self.presentation.stimulus.fixation()
            QTimer.singleShot(seconds * 1000, self.begin_fixation)
        else:
            if self.training.oscar.isChecked() and self.training.warmup.isChecked() and not self.device.connected:
                self.training.status.setText("No EEG yet; starting image loop. OSCAR will adapt when a device connects.")
            self.begin_fixation()

    def begin_fixation(self):
        if not self.active:
            return
        self.next_trial = self.plan.next() if self.plan else None
        if self.next_trial is None:
            self.stop_run(); return
        self.presentation.stimulus.fixation()
        self.training.status.setText(f"Fixation before trial {self.shown + 1}")
        QTimer.singleShot(random.randint(round(self.training.fixation_min.value() * 1000),
                                         round(self.training.fixation_max.value() * 1000)), self.show_trial)

    def show_trial(self):
        if not self.active or self.next_trial is None:
            return
        label, path = self.next_trial
        self.presentation.stimulus.image(path)
        # The paint callback is the closest software-side onset observation.

    def on_image_onset(self, stamp: float):
        if not self.active or self.next_trial is None:
            return
        label, path = self.next_trial
        self.shown += 1
        self.pending.append(PendingEpoch(self.shown, label, path, stamp,
                                         stamp + self.epoch_spec.end_ms / 1000 + 2.0))
        self.trial_log.append({"number": self.shown, "label": label, "image": path.name,
                               "onset_s_from_run_start": round(stamp - self.run_start_clock, 6),
                               "eeg": "pending"})
        self.training.status.setText(f"Image {self.shown}: {path.name}")
        self._update_counts()
        self.next_trial = None
        QTimer.singleShot(self.epoch_spec.image_duration_ms, self.end_image)

    def end_image(self):
        if not self.active:
            return
        self.presentation.stimulus.fixation()
        self.begin_fixation()

    def poll(self):
        now = time.perf_counter()
        if now - self.last_quality_time >= .5:
            self.last_quality_time = now
            if self.device.connected and now - self.device.ring.latest < 1.0:
                _, quality_samples = self.device.ring.recent(3)
                self.features.set_quality(assess_signal_quality(quality_samples))
            else:
                self.features.set_quality(None)
        if self.device.connected and now - self.last_feature_time >= .25:
            spec = self.epoch_spec if self.active or self.finishing else DEFAULT_EPOCH
            times, samples = self.device.ring.recent(max(2, spec.n_samples / FS + 1))
            if len(samples) >= spec.n_samples and times[-1] > self.last_feature_time:
                try:
                    values = extract(samples[-spec.n_samples:], spec)
                    self.features.add(times[-1], values)
                    self.last_feature_time = now
                except ValueError:
                    pass
        for item in self.pending[:]:
            if self.device.connected:
                epoch = self.device.ring.epoch(item.onset, self.epoch_spec)
            else:
                epoch = None
            if epoch is not None:
                try:
                    vector = extract(epoch, self.epoch_spec)
                    # Guard gross artifacts/flatlines without pretending OSCAR has fixed them.
                    if np.ptp(epoch, axis=0).max() > 500 or np.ptp(epoch, axis=0).min() < .05:
                        raise ValueError("artifact or flat channel")
                    prediction = self.trainer.add(vector, item.label)
                    self.usable += 1
                    self.epochs.append(epoch.astype(np.float32))
                    self.epoch_trial_numbers.append(item.number)
                    self.trial_log[item.number - 1]["eeg"] = "usable"
                    self.training.update_weights(self.trainer)
                    if prediction is not None:
                        self.trial_log[item.number - 1].update({"prediction": prediction.label,
                            "p_happy": prediction.probability_happy, "correct": prediction.correct})
                        self.training.add_accuracy(item.number, self.trainer.rolling_accuracy)
                        self.training.accuracy.setText(f"Last {min(20, len(self.trainer.predictions))} held-out predictions: {self.trainer.rolling_accuracy:.1f}%")
                except ValueError:
                    self.skipped += 1
                    self.trial_log[item.number - 1]["eeg"] = "rejected_artifact"
                self.pending.remove(item)
                self._update_counts()
            elif now >= item.deadline:
                self.skipped += 1
                self.trial_log[item.number - 1]["eeg"] = "missing_or_incomplete"
                self.pending.remove(item)
                self._update_counts()
            else:
                # Keep test-before-train predictions in presentation order.
                break
        if self.finishing and not self.pending:
            self.finish_run()

    def _update_counts(self):
        self.training.count_status.setText(f"Trials: {self.shown} · usable EEG: {self.usable} · held-out predictions: {len(self.trainer.predictions)} · skipped: {self.skipped}")

    def stop_run(self):
        if not self.active:
            return
        self.active = False
        self.finishing = True
        self.presentation.stimulus.idle()
        self.training.stop.setEnabled(False)
        self.stop_shortcut.setEnabled(False)
        self.training.status.setText("Finishing pending EEG epochs…")
        if not self.pending:
            self.finish_run()

    def finish_run(self):
        self.finishing = False
        self.training.status.setText(f"Stopped after {self.shown} presentations; {self.usable} usable EEG trials.")
        self.training.save.setEnabled(self.trainer.model is not None)
        self.training.discard.setEnabled(True)
        self.run_metadata.update({"stopped_at": datetime.now().astimezone().isoformat(),
                                  "presented_trials": self.shown, "skipped_eeg_trials": self.skipped,
                                  "trial_log": self.trial_log})
        if self.trainer.model is None:
            self.training.status.setText(self.training.status.text() + " A model needs at least 10 usable EEG trials with both labels.")

    def save_model(self):
        directory = self.root / "saved_models"
        directory.mkdir(exist_ok=True)
        participant = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.run_metadata["participant_id"])
        stem = directory / (f"scene_valence_{participant}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        epochs_path = stem.with_suffix(".epochs.npz")
        self.run_metadata["epochs_file"] = epochs_path.name
        try:
            weights, metadata = self.trainer.save(stem, self.run_metadata)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot save", str(exc)); return
        np.savez_compressed(epochs_path, eeg_uV=np.stack(self.epochs),
                            label_happy=np.asarray(self.trainer.y, dtype=np.uint8),
                            trial_number=np.asarray(self.epoch_trial_numbers, dtype=np.int32))
        self.training.status.setText(f"Saved {weights.name}, {metadata.name}, and {epochs_path.name}")
        self.training.save.setEnabled(False)
        self.training.start.setEnabled(True)
        self.training.discard.setEnabled(False)
        self._enable_settings()

    def discard_run(self):
        self.trainer = OnlineTrainer()
        self.epochs = []
        self.epoch_trial_numbers = []
        self.training.status.setText("Run discarded. Ready to start again.")
        self.training.start.setEnabled(True)
        self.training.save.setEnabled(False)
        self.training.discard.setEnabled(False)
        self._enable_settings()

    def _enable_settings(self):
        for widget in (self.training.participant, self.training.serial, self.training.count, self.training.persistent, self.training.stop_key, self.training.oscar,
                       self.training.warmup, self.training.warmup_seconds, self.training.image_seconds, self.training.fixation_min, self.training.fixation_max):
            widget.setEnabled(True)
        self.training.count.setEnabled(not self.training.persistent.isChecked())

    def closeEvent(self, event):
        self.active = False
        self.clock.stop()
        self.device.close()
        for window in list(self.detached.values()):
            window.close()
        event.accept()


def launch(root: Path):
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(antialias=True)
    window = MainWindow(root)
    window.show()
    return app.exec()
