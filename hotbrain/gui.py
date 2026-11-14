"""PySide6 operator interface and stimulus timing state machine."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImageReader, QKeySequence, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
    QWidget, QTableWidget, QTableWidgetItem, QToolTip,
)

from . import classifiers
from .acquisition import DeviceManager
from .config import CONFIG, AppConfig
from .dataset import Dataset, StimulusImage, load_dataset
from .features import EpochSpec, FeatureExtractor, START_MS
from .quality import ChannelQuality, SignalQuality
from .training import ROLLING, MultiTrainer, TrialPlan

CURVE_COLORS = ("#6dd7bf", "#8bb9ff", "#ffb86b", "#ff727a", "#c79bff", "#f4e06d", "#7fe08a", "#ff9ed8")


class Stimulus(QWidget):
    onset = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 260)
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
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.set_timing(2.0, 3.0, EpochSpec(image_duration_ms=4500))

    def set_timing(self, fixation_min: float, fixation_max: float, spec: EpochSpec):
        self.note.setText(f"Fixation: {fixation_min:g}–{fixation_max:g} s   ·   Image: {spec.image_duration_ms / 1000:g} s   ·   EEG epoch: {START_MS} to +{spec.end_ms} ms")


class TrainingPanel(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    save_requested = Signal()
    discard_requested = Signal()

    def __init__(self, config: AppConfig, min_image_s: float):
        super().__init__()
        self.n_groups = len(config.datasets)
        self.class1_names = {target.name: target.classes[1] for target in config.targets}
        layout = QVBoxLayout(self)
        heading = QLabel("Scene emotion training")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        self.settings_widget = QWidget()
        settings_layout = QVBoxLayout(self.settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.participant = QLineEdit("debug")
        self.participant.setPlaceholderText("Participant ID")
        form.addRow("Participant ID", self.participant)
        self.serial = QLineEdit("UN-2019.03.01")
        self.serial.setPlaceholderText("Blank = first available Unicorn")
        form.addRow("Unicorn serial", self.serial)
        self.count = QSpinBox()
        self.count.setRange(self.n_groups, 100000)
        self.count.setSingleStep(self.n_groups)
        self.count.setValue(max(self.n_groups, 160 - 160 % self.n_groups))
        form.addRow("Presentations", self.count)
        self.persistent = QCheckBox("Continue until manually stopped")
        self.persistent.toggled.connect(self.count.setDisabled)
        form.addRow("Mode", self.persistent)
        self.image_seconds = QDoubleSpinBox()
        self.image_seconds.setRange(min_image_s, 60.0)
        self.image_seconds.setDecimals(2)
        self.image_seconds.setSingleStep(.1)
        self.image_seconds.setSuffix(" s")
        self.image_seconds.setValue(max(4.5, min_image_s))
        form.addRow("Image duration", self.image_seconds)
        self.fixation_min = QDoubleSpinBox()
        self.fixation_min.setRange(.2, 60.0)
        self.fixation_min.setDecimals(2)
        self.fixation_min.setSingleStep(.1)
        self.fixation_min.setSuffix(" s")
        self.fixation_min.setValue(2.0)
        form.addRow("Fixation minimum", self.fixation_min)
        self.fixation_max = QDoubleSpinBox()
        self.fixation_max.setRange(.2, 60.0)
        self.fixation_max.setDecimals(2)
        self.fixation_max.setSingleStep(.1)
        self.fixation_max.setSuffix(" s")
        self.fixation_max.setValue(3.0)
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
        model_grid = QWidget()
        model_layout = QGridLayout(model_grid)
        model_layout.setContentsMargins(0, 0, 0, 0)
        self.train_boxes: dict[str, QCheckBox] = {}
        for index, (key, spec) in enumerate(classifiers.CLASSIFIERS.items()):
            box = QCheckBox(spec.label)
            box.setChecked(key in config.classifiers)
            self.train_boxes[key] = box
            model_layout.addWidget(box, index // 2, index % 2)
        form.addRow("Train models", model_grid)
        model_note = QLabel("Selected models use the same EEG trials. SVM probabilities use cross-validated sigmoid calibration; random forest probabilities are native leaf averages. Each model predicts before refitting.")
        model_note.setWordWrap(True)
        form.addRow("", model_note)
        settings_layout.addLayout(form)
        self.dataset_status = QLabel()
        self.dataset_status.setWordWrap(True)
        settings_layout.addWidget(self.dataset_status)
        self.config_note = QLabel()
        self.config_note.setWordWrap(True)
        self.config_note.setObjectName("warning")
        settings_layout.addWidget(self.config_note)
        self.detail = QLabel(f"Finite runs are exactly balanced across the {self.n_groups} dataset folders and shuffled; "
                             "continuous runs draw a folder uniformly at random. Each target/classifier pair predicts a trial "
                             "before that trial is added to its fit. OSCAR adapts online; g.Pype does not expose a saved calibration state.")
        self.detail.setWordWrap(True)
        settings_layout.addWidget(self.detail)
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setMaximumHeight(420)
        self.settings_scroll.setWidget(self.settings_widget)
        layout.addWidget(self.settings_scroll)
        buttons = QHBoxLayout()
        self.start = QPushButton("Start training")
        self.stop = QPushButton("Stop training")
        self.save = QPushButton("Save fitted models")
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
        self.count_status = QLabel("Trials: 0 · usable EEG: 0")
        layout.addWidget(self.count_status)
        self.accuracy = QTableWidget(0, 6)
        self.accuracy.setHorizontalHeaderLabels(("Target (+)", "Model", f"Last {ROLLING} correct", "N", "P(+)", "Save"))
        self.accuracy.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((100, 130, 125, 45, 75, 50)):
            self.accuracy.setColumnWidth(column, width)
        self.accuracy.setMaximumHeight(240)
        layout.addWidget(self.accuracy)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#101827")
        self.plot.setLabel("left", "Correct (rolling)", units="%")
        self.plot.setLabel("bottom", "Presented trial")
        self.plot.setYRange(0, 100)
        self.plot.showGrid(x=True, y=True, alpha=.2)
        self.plot.addLegend(offset=(-10, 10))
        layout.addWidget(self.plot, 1)
        self.weights = QTableWidget(0, 1)
        self.weights.horizontalHeader().setStretchLastSection(True)
        self.weights.setMaximumHeight(195)
        layout.addWidget(self.weights)
        self.model_keys: list[tuple[str, str]] = []
        self.save_boxes: dict[tuple[str, str], QCheckBox] = {}
        self.save_selection_ready = False
        self.curves: dict[tuple[str, str], tuple[object, list[int], list[float]]] = {}

    def selected_classifiers(self) -> list[str]:
        return [key for key, box in self.train_boxes.items() if box.isChecked()]

    def selected_save_models(self) -> set[tuple[str, str]]:
        return {key for key, box in self.save_boxes.items() if box.isEnabled() and box.isChecked()}

    def update_save_button(self):
        self.save.setEnabled(self.save_selection_ready and bool(self.selected_save_models()))

    def prepare_save_selection(self, trainer: MultiTrainer):
        for key, box in self.save_boxes.items():
            fitted = trainer.models[key].model is not None
            box.setChecked(fitted)
            box.setEnabled(fitted)
        self.save_selection_ready = True
        self.update_save_button()

    def reset_save_selection(self):
        self.save_selection_ready = False
        for box in self.save_boxes.values():
            box.setEnabled(False)
        self.update_save_button()

    def configure_models(self, model_keys: list[tuple[str, str]], feature_names: tuple[str, ...]):
        """Reset the accuracy table, curves and weight table for a new run."""
        self.model_keys = list(model_keys)
        self.reset_save_selection()
        self.save_boxes = {}
        self.plot.clear()
        if self.plot.plotItem.legend is not None:
            self.plot.plotItem.legend.clear()
        self.curves = {}
        self.accuracy.setRowCount(len(model_keys))
        for row, (target, clf) in enumerate(model_keys):
            label = classifiers.CLASSIFIERS[clf].label
            for col, text in enumerate((f"{target} ({self.class1_names[target]})", label, "—", "0", "—")):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.accuracy.setItem(row, col, item)
            save_box = QCheckBox()
            save_box.setToolTip(f"Save fitted {target} / {label} model")
            save_box.setEnabled(False)
            save_box.toggled.connect(self.update_save_button)
            self.accuracy.setCellWidget(row, 5, save_box)
            self.save_boxes[(target, clf)] = save_box
            pen = pg.mkPen(CURVE_COLORS[row % len(CURVE_COLORS)], width=3)
            curve = self.plot.plot(pen=pen, symbol="o", symbolSize=5, name=f"{target} · {label}")
            self.curves[(target, clf)] = (curve, [], [])
        self.weights.clear()
        self.weights.setRowCount(len(feature_names))
        self.weights.setColumnCount(1 + len(model_keys))
        self.weights.setHorizontalHeaderLabels(("Feature", *[f"{t}/{c} weight" for t, c in model_keys]))
        self.weights.setColumnWidth(0, 300)
        for row, name in enumerate(feature_names):
            self.weights.setItem(row, 0, QTableWidgetItem(name))
            for col in range(len(model_keys)):
                self.weights.setItem(row, col + 1, QTableWidgetItem("—"))

    def add_accuracy(self, key: tuple[str, str], trial: int, value: float):
        curve, xs, ys = self.curves[key]
        xs.append(trial)
        ys.append(value)
        curve.setData(xs, ys)

    def update_models(self, trainer: MultiTrainer):
        for row, key in enumerate(self.model_keys):
            model = trainer.models[key]
            accuracy = model.rolling_accuracy
            self.accuracy.item(row, 2).setText(
                "—" if accuracy is None else f"{accuracy:.1f}% (of {min(ROLLING, len(model.predictions))})")
            self.accuracy.item(row, 3).setText(str(len(model.predictions)))
            self.accuracy.item(row, 4).setText(
                "—" if not model.predictions else f"{model.predictions[-1].probability:.3f}")
            weights = classifiers.coefficients(model.model)
            for feature in range(self.weights.rowCount()):
                if model.model is None:
                    text = "—"
                else:
                    text = "n/a (non-linear)" if weights is None else f"{weights[feature]:+.3f}"
                self.weights.item(feature, row + 1).setText(text)


# Approximate 10-20 positions on the 400x300 top-view drawing (nose up).
ELECTRODE_XY = {
    "FPZ": (200, 40), "FP1": (165, 42), "FP2": (235, 42),
    "AFZ": (200, 55), "AF3": (170, 57), "AF4": (230, 57),
    "F7": (118, 78), "F3": (155, 72), "F1": (178, 70), "FZ": (200, 69),
    "F2": (222, 70), "F4": (245, 72), "F8": (282, 78),
    "FC5": (118, 106), "FC3": (146, 104), "FC1": (172, 104), "FCZ": (200, 104),
    "FC2": (228, 104), "FC4": (254, 104), "FC6": (282, 106),
    "T7": (96, 139), "C5": (116, 139), "C3": (138, 139), "C1": (170, 140), "CZ": (200, 141),
    "C2": (230, 140), "C4": (262, 139), "C6": (284, 139), "T8": (304, 139),
    "CP5": (116, 172), "CP3": (145, 172), "CP1": (172, 172), "CPZ": (200, 172),
    "CP2": (228, 172), "CP4": (255, 172), "CP6": (284, 172),
    "P7": (118, 210), "P3": (150, 204), "P1": (176, 203), "PZ": (200, 202),
    "P2": (224, 203), "P4": (250, 204), "P8": (282, 210),
    "PO7": (125, 235), "PO3": (168, 232), "POZ": (200, 230), "PO4": (232, 232), "PO8": (275, 235),
    "O1": (165, 258), "OZ": (200, 255), "O2": (235, 258),
}


def electrode_positions(channels: tuple[str, ...]) -> list[tuple[int, int]]:
    """Known 10-20 names get their scalp position; unknown names line up at the bottom."""
    unknown = [c for c in channels if c.upper() not in ELECTRODE_XY]
    out = []
    for name in channels:
        if name.upper() in ELECTRODE_XY:
            out.append(ELECTRODE_XY[name.upper()])
        else:
            i = unknown.index(name)
            out.append((round(200 + (i - (len(unknown) - 1) / 2) * 50), 290))
    return out


class ElectrodeMap(QWidget):
    """Top-view brain diagram with one live status light per configured electrode."""

    def __init__(self, channels: tuple[str, ...]):
        super().__init__()
        self.channels = channels
        self.POSITIONS = electrode_positions(channels)
        self.ratings = [ChannelQuality(None, "Waiting for EEG") for _ in channels]
        self.setMinimumHeight(260)
        self.setMaximumHeight(310)
        self.setMouseTracking(True)

    def set_quality(self, ratings: list[ChannelQuality] | None):
        self.ratings = ratings if ratings is not None else [
            ChannelQuality(None, "Waiting for EEG") for _ in self.channels]
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
        # the channel markers, which remain the most prominent elements.
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
        for (x, y), name, rating in zip(self.POSITIONS, self.channels, self.ratings, strict=True):
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
        for (cx, cy), name, rating in zip(self.POSITIONS, self.channels, self.ratings, strict=True):
            if (x - cx) ** 2 + (y - cy) ** 2 <= 18 ** 2:
                QToolTip.showText(event.globalPosition().toPoint(), f"{name}: {rating.detail}", self)
                return
        QToolTip.hideText()

    def leaveEvent(self, event):
        QToolTip.hideText()


class FeaturePanel(QWidget):
    def __init__(self, feature_names: tuple[str, ...], channels: tuple[str, ...]):
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
        self.quality_map = ElectrodeMap(channels)
        layout.addWidget(self.quality_map)
        quality_note = QLabel("Top = forehead. Green = recent amplitude looks plausible; red = flat or unusually large; grey = waiting. Hover over an electrode for measurements. This checks the streamed signal (after OSCAR if enabled), not electrode impedance or guaranteed scalp contact. Allow 2 s after connecting.")
        quality_note.setWordWrap(True)
        layout.addWidget(quality_note)
        self.set_quality(None)
        self.selector = QComboBox()
        self.selector.addItems(feature_names)
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


class PanelWindow(QMainWindow):
    def __init__(self, title: str, panel: QWidget, on_close=None):
        super().__init__()
        self.setWindowTitle(title)
        self.setCentralWidget(panel)
        self._on_close = on_close

    def closeEvent(self, event):
        if self._on_close is not None:
            self._on_close()
        event.accept()


@dataclass
class PendingEpoch:
    number: int
    image: StimulusImage
    onset: float
    deadline: float


class MainWindow(QMainWindow):
    def __init__(self, root: Path, config: dict | None = None):
        super().__init__()
        self.root = root
        self.config = AppConfig.from_dict(CONFIG if config is None else config, root)
        self.extractor = FeatureExtractor(self.config.eeg)
        self.quality = SignalQuality(self.config.eeg)
        self.idle_spec = self.extractor.epoch_spec(max(1000, self.extractor.min_image_duration_ms))
        self.setWindowTitle("Hot Brain · scene training")
        self.resize(720, 900)
        self.training = TrainingPanel(self.config, self.extractor.min_image_duration_ms / 1000)
        self.device = DeviceManager(self.config.eeg, self.training.serial.text().strip() or None)
        self.presentation = PresentationPanel()
        self.epoch_spec = self.idle_spec
        for control in (self.training.image_seconds, self.training.fixation_min, self.training.fixation_max):
            control.valueChanged.connect(self.update_timing_note)
        self.update_timing_note()
        self.features = FeaturePanel(self.extractor.names, self.extractor.channels)
        self.setCentralWidget(self.training)
        self.presentation_window = PanelWindow("Hot Brain · image presentation", self.presentation,
                                                self.stop_run)
        self.features_window = PanelWindow("Hot Brain · EEG features", self.features)
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
        warnings = self.config.eeg.warnings()
        self.training.config_note.setText("\n".join(f"Config warning: {w}" for w in warnings))
        self.training.config_note.setVisible(bool(warnings))
        self.dataset: Dataset | None = self.load_dataset(show_errors=False)
        self.plan: TrialPlan | None = None
        self.trainer = MultiTrainer([], [], len(self.extractor.names))
        self.pending: list[PendingEpoch] = []
        self.next_trial: StimulusImage | None = None
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
        self.epoch_labels: list[dict[str, int]] = []
        self.epoch_values: list[dict[str, float]] = []
        self.epoch_groups: list[str] = []
        self.run_start_clock = 0.0
        self.clock = QTimer(self)
        self.clock.setInterval(100)
        self.clock.timeout.connect(self.poll)
        self.clock.start()
        self.setStyleSheet("QMainWindow,QWidget{background:#182335;color:#e8effc;font-size:13px} QLabel#heading{font-size:24px;font-weight:700;margin:4px 0 10px} QLabel#metric{font-size:18px;font-weight:600;color:#6dd7bf} QLabel#warning{color:#ffb86b} QLabel#status{font-size:16px;font-weight:600} QPushButton{background:#29466c;border-radius:7px;padding:8px 12px} QPushButton:hover{background:#376293} QPushButton:disabled{background:#354155;color:#8d9aaf} QSpinBox,QDoubleSpinBox,QComboBox{background:#25364e;padding:5px;border:1px solid #627795} QTabBar::tab{padding:9px 17px}")

    @property
    def _settings_widgets(self) -> list[QWidget]:
        t = self.training
        return [t.participant, t.serial, t.count, t.persistent, t.stop_key, t.oscar, t.warmup, t.warmup_seconds,
                t.image_seconds, t.fixation_min, t.fixation_max, *t.train_boxes.values()]

    def load_dataset(self, show_errors: bool = True) -> Dataset | None:
        try:
            dataset = load_dataset(self.config)
        except (OSError, ValueError) as exc:
            self.training.dataset_status.setText(f"Dataset error: {exc}")
            if show_errors:
                QMessageBox.warning(self, "Dataset", str(exc))
            return None
        self.training.dataset_status.setText(f"Dataset ({self.config.data_dir.name}/): {dataset.summary()}")
        return dataset

    def update_timing_note(self):
        spec = self.extractor.epoch_spec(round(self.training.image_seconds.value() * 1000))
        self.presentation.set_timing(self.training.fixation_min.value(), self.training.fixation_max.value(), spec)

    def show_windows(self, available: QRect | None = None):
        """Open all three movable windows in adjacent slices of one screen."""
        if available is None:
            screen = QApplication.primaryScreen()
            available = screen.availableGeometry() if screen is not None else QRect(0, 0, 1600, 900)
        area = available.adjusted(8, 8, -8, -8)
        gap = 10
        width = area.width() - 2 * gap
        training_width = round(width * .40)
        presentation_width = round(width * .32)
        feature_width = width - training_width - presentation_width
        self.setGeometry(area.x(), area.y(), training_width, area.height())
        self.presentation_window.setGeometry(area.x() + training_width + gap, area.y(),
                                             presentation_width, area.height())
        self.features_window.setGeometry(area.x() + training_width + gap + presentation_width + gap,
                                         area.y(), feature_width, area.height())
        self.show()
        self.presentation_window.show()
        self.features_window.show()

    def on_connection(self, connected: bool):
        eeg = self.config.eeg
        self.features.note.setText(
            f"Live Unicorn EEG · {eeg.fs} Hz · {', '.join(eeg.electrodes)} · OSCAR " + ("on" if self.device.oscar else "off")
            if connected else "Waiting for Unicorn; continuing image loop without EEG.")
        if not connected:
            self.features.set_quality(None)

    def start_run(self):
        fixation_min = self.training.fixation_min.value()
        fixation_max = self.training.fixation_max.value()
        if fixation_min > fixation_max:
            QMessageBox.warning(self, "Fixation timing", "Fixation minimum must be at most the maximum.")
            return
        classifier_keys = self.training.selected_classifiers()
        if not classifier_keys:
            QMessageBox.warning(self, "Classifiers", "Select at least one classifier.")
            return
        try:
            self.epoch_spec = self.extractor.epoch_spec(round(self.training.image_seconds.value() * 1000))
        except ValueError as exc:
            QMessageBox.warning(self, "Image duration", str(exc)); return
        self.dataset = self.load_dataset()
        if self.dataset is None:
            return
        for items in self.dataset.groups.values():
            for item in items:
                if not QImageReader(str(item.path)).canRead():
                    QMessageBox.warning(self, "Image error", f"Cannot read {item.path}")
                    return
        total = None if self.training.persistent.isChecked() else self.training.count.value()
        try:
            self.plan = TrialPlan(self.dataset.groups, total)
        except ValueError as exc:
            QMessageBox.warning(self, "Training images", str(exc)); return
        self.device.set_retention(self.training.image_seconds.value() + 5)
        target_names = [t.name for t in self.config.targets]
        self.trainer = MultiTrainer(target_names, classifier_keys, len(self.extractor.names))
        self.training.configure_models(list(self.trainer.models), self.extractor.names)
        self.pending.clear()
        self.shown = self.usable = self.skipped = 0
        self.training.start.setEnabled(False)
        self.training.stop.setEnabled(True)
        self.training.settings_scroll.setVisible(False)
        self.stop_shortcut.setEnabled(True)
        self.training.save.setEnabled(False)
        self.training.discard.setEnabled(False)
        for widget in self._settings_widgets:
            widget.setEnabled(False)
        self.active = True
        self.finishing = False
        self.trial_log = []
        self.epochs = []
        self.epoch_trial_numbers = []
        self.epoch_labels = []
        self.epoch_values = []
        self.epoch_groups = []
        self.run_start_clock = time.perf_counter()
        self._update_counts()
        # A visible presentation window is required for paint-timed onsets.
        if not self.presentation_window.isVisible():
            self.presentation_window.show()
        eeg = self.config.eeg
        self.run_metadata = {"started_at": datetime.now().astimezone().isoformat(),
                             "participant_id": self.training.participant.text().strip() or "unnamed",
                             "unicorn_serial": self.training.serial.text().strip() or None,
                             "mode": "persistent" if total is None else "balanced_finite",
                             "requested_trials": total, "oscar_enabled": self.training.oscar.isChecked(),
                             "image_duration_s": self.epoch_spec.image_duration_ms / 1000,
                             "fixation_min_s": fixation_min, "fixation_max_s": fixation_max,
                             "epoch_start_ms": START_MS, "epoch_end_ms": self.epoch_spec.end_ms,
                             "epoch_sample_count": self.epoch_spec.n_samples,
                             "fs": eeg.fs, "channels": eeg.electrodes, "device_channels": eeg.device_channels,
                             "feature_names": self.extractor.names, "classifiers": classifier_keys,
                             "config": self.config.raw,
                             "oscar_settling_requested_seconds": self.training.warmup_seconds.value() if self.training.warmup.isChecked() else 0,
                             "oscar_settling_applied_seconds": 0,
                             "dataset": self.dataset.manifest(),
                             "timing_note": f"Image onset uses Qt paint completion; EEG times anchor to first host frame arrival minus device/OSCAR delay then advance at {eeg.fs} Hz. Validate with hardware trigger."}
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
        self.presentation.stimulus.image(self.next_trial.path)
        # The paint callback is the closest software-side onset observation.

    def on_image_onset(self, stamp: float):
        if not self.active or self.next_trial is None:
            return
        item = self.next_trial
        self.shown += 1
        self.pending.append(PendingEpoch(self.shown, item, stamp, stamp + self.epoch_spec.end_ms / 1000 + 2.0))
        self.trial_log.append({"number": self.shown, "group": item.group, "image": item.path.name,
                               "labels": {t: self.dataset.class_name(t, v) for t, v in item.labels.items()},
                               "target_values": dict(item.values),
                               "onset_s_from_run_start": round(stamp - self.run_start_clock, 6),
                               "eeg": "pending"})
        self.training.status.setText(f"Image {self.shown}: {item.group}/{item.path.name}")
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
                self.features.set_quality(self.quality.assess(quality_samples))
            else:
                self.features.set_quality(None)
        if self.device.connected and now - self.last_feature_time >= .25:
            spec = self.epoch_spec if self.active or self.finishing else self.idle_spec
            times, samples = self.device.ring.recent(max(2, spec.n_samples / spec.fs + 1))
            if len(samples) >= spec.n_samples and times[-1] > self.last_feature_time:
                try:
                    values = self.extractor.extract(samples[-spec.n_samples:], spec)
                    self.features.add(times[-1], values)
                    self.last_feature_time = now
                except ValueError:
                    pass
        for item in self.pending[:]:
            epoch = self.device.ring.epoch(item.onset, self.epoch_spec) if self.device.connected else None
            if epoch is not None:
                log = self.trial_log[item.number - 1]
                try:
                    vector = self.extractor.extract(epoch, self.epoch_spec)
                    # Guard gross artifacts/flatlines without pretending OSCAR has fixed them.
                    if np.ptp(epoch, axis=0).max() > 500 or np.ptp(epoch, axis=0).min() < .05:
                        raise ValueError("artifact or flat channel")
                except ValueError:
                    self.skipped += 1
                    log["eeg"] = "rejected_artifact"
                else:
                    predictions = self.trainer.add(vector, item.image.labels)
                    self.usable += 1
                    self.epochs.append(epoch.astype(np.float32))
                    self.epoch_trial_numbers.append(item.number)
                    self.epoch_labels.append(dict(item.image.labels))
                    self.epoch_values.append(dict(item.image.values))
                    self.epoch_groups.append(item.image.group)
                    log["eeg"] = "usable"
                    if predictions:
                        log["predictions"] = {
                            f"{t}/{c}": {"predicted": self.dataset.class_name(t, p.label),
                                         "p_class1": p.probability, "correct": p.correct}
                            for (t, c), p in predictions.items()}
                        for key in predictions:
                            self.training.add_accuracy(key, item.number, self.trainer.models[key].rolling_accuracy)
                    self.training.update_models(self.trainer)
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
        self.training.count_status.setText(f"Trials: {self.shown} · usable EEG: {self.usable} · skipped: {self.skipped}")

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
        self.training.prepare_save_selection(self.trainer)
        self.training.discard.setEnabled(True)
        self.run_metadata.update({"stopped_at": datetime.now().astimezone().isoformat(),
                                  "presented_trials": self.shown, "skipped_eeg_trials": self.skipped,
                                  "trial_log": self.trial_log})
        if not self.trainer.any_fitted:
            self.training.status.setText(self.training.status.text() + " A model needs at least 10 usable EEG trials with both labels.")

    def save_model(self):
        selected = self.training.selected_save_models()
        if not selected:
            QMessageBox.warning(self, "Cannot save", "Select at least one fitted model in the Save column.")
            return
        directory = self.root / "saved_models"
        directory.mkdir(exist_ok=True)
        participant = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.run_metadata["participant_id"])
        stem = directory / (f"scene_{participant}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        epochs_path = stem.with_suffix(".epochs.npz")
        self.run_metadata["epochs_file"] = epochs_path.name
        try:
            written = self.trainer.save(stem, self.run_metadata, selected)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot save", str(exc)); return
        labels = {f"label_{t}": np.asarray([row[t] for row in self.epoch_labels], dtype=np.uint8)
                  for t in self.trainer.targets}
        labels.update({f"value_{t}": np.asarray([row[t] for row in self.epoch_values], dtype=np.float32)
                       for t in self.trainer.targets})
        np.savez_compressed(epochs_path, eeg_uV=np.stack(self.epochs),
                            trial_number=np.asarray(self.epoch_trial_numbers, dtype=np.int32),
                            group=np.asarray(self.epoch_groups, dtype=str),
                            channels=np.asarray(self.config.eeg.electrodes, dtype=str), **labels)
        self.training.status.setText(f"Saved {len(written) + 1} files to saved_models/{stem.name}.*")
        self.training.reset_save_selection()
        self.training.start.setEnabled(True)
        self.training.settings_scroll.setVisible(True)
        self.training.discard.setEnabled(False)
        self._enable_settings()

    def discard_run(self):
        self.trainer = MultiTrainer([], [], len(self.extractor.names))
        self.epochs = []
        self.epoch_trial_numbers = []
        self.epoch_labels = []
        self.epoch_values = []
        self.epoch_groups = []
        self.training.status.setText("Run discarded. Ready to start again.")
        self.training.start.setEnabled(True)
        self.training.settings_scroll.setVisible(True)
        self.training.reset_save_selection()
        self.training.discard.setEnabled(False)
        self._enable_settings()

    def _enable_settings(self):
        for widget in self._settings_widgets:
            widget.setEnabled(True)
        self.training.count.setEnabled(not self.training.persistent.isChecked())

    def closeEvent(self, event):
        self.active = False
        self.clock.stop()
        self.device.close()
        self.presentation_window.close()
        self.features_window.close()
        event.accept()


def launch(root: Path, config: dict | None = None):
    """Start the GUI. ``config`` defaults to ``hotbrain.config.CONFIG``."""
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(antialias=True)
    window = MainWindow(root, config)
    window.show_windows()
    return app.exec()
