"""PySide6 operator interface and stimulus timing state machine."""
from __future__ import annotations

import json
import random
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImageReader, QKeySequence, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
    QWidget, QTableWidget, QTableWidgetItem, QTabWidget, QToolTip,
)

from . import classifiers
from .acquisition import DeviceManager
from .config import CONFIG, AppConfig
from .dataset import Dataset, StimulusImage, load_dataset
from .features import EpochSpec, FeatureExtractor, START_MS
from .quality import ChannelQuality, SignalQuality
from .training import ROLLING, MultiTrainer, TrialPlan, balanced_unique_capacity
from .affect_plane import AffectPlane
from .sessions import (SESSION_TEST, SESSION_TRAINING, SESSION_VALIDATION, OncePlan, Smoother,
                       ValidationScores, affect_point, load_models, split_dataset)

CURVE_COLORS = ("#6dd7bf", "#8bb9ff", "#ffb86b", "#ff727a", "#c79bff", "#f4e06d", "#7fe08a", "#ff9ed8")
EVAL_PREQUENTIAL = "prequential"
EVAL_SPLIT = "split"


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
        row = QHBoxLayout()
        self.stimulus = Stimulus()
        row.addWidget(self.stimulus, 3)
        # Valence-arousal plane: shown in validation and test sessions only.
        self.plane = AffectPlane()
        self.plane.setVisible(False)
        row.addWidget(self.plane, 2)
        layout.addLayout(row, 1)
        self.note = QLabel()
        self.note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.set_timing(2.0, 3.0, EpochSpec(image_duration_ms=3000))

    def set_timing(self, fixation_min: float, fixation_max: float, spec: EpochSpec):
        self.note.setText(f"Fixation: {fixation_min:g}–{fixation_max:g} s   ·   Image: {spec.image_duration_ms / 1000:g} s   ·   EEG epoch: {START_MS} to +{spec.end_ms} ms")


class TrainingPanel(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    save_requested = Signal()
    discard_requested = Signal()

    def __init__(self, config: AppConfig, min_image_s: float):
        super().__init__()
        self.n_groups = config.n_trial_groups
        self.class1_names = {target.name: target.classes[1] for target in config.targets}
        layout = QVBoxLayout(self)
        self.heading = QLabel("Scene emotion training")
        self.heading.setObjectName("heading")
        layout.addWidget(self.heading)
        self.settings_widget = QWidget()
        settings_layout = QVBoxLayout(self.settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.session = QComboBox()
        for key, label, tip in ((SESSION_TRAINING, "Training", "models learn trial by trial"),
                                (SESSION_VALIDATION, "Validation", "held-out images, each once; frozen models"),
                                (SESSION_TEST, "Test", "test images; match the target valence/arousal")):
            self.session.addItem(label, key)
            self.session.setItemData(self.session.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
        form.addRow("Session", self.session)
        models_row = QHBoxLayout()
        self.load_models = QPushButton("Load saved models…")
        self.models_status = QLabel("No trained models yet.")
        self.models_status.setWordWrap(True)
        models_row.addWidget(self.load_models)
        models_row.addWidget(self.models_status, 1)
        form.addRow("Models", models_row)
        self.feedback_classifier = QComboBox()
        self.feedback_classifier.setToolTip("Classifier whose valence/arousal estimate drives the live plane "
                                            "in validation and test sessions.")
        form.addRow("Live display uses", self.feedback_classifier)
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
        self.evaluation = QComboBox()
        self.evaluation.addItem("Prequential · no repeated images", EVAL_PREQUENTIAL)
        self.evaluation.addItem(f"Hold out {round(config.validation_fraction * 100)}% for validation", EVAL_SPLIT)
        form.addRow("Evaluation", self.evaluation)
        self.evaluation_status = QLabel()
        self.evaluation_status.setWordWrap(True)
        form.addRow("", self.evaluation_status)
        self.image_seconds = QDoubleSpinBox()
        self.image_seconds.setRange(min_image_s, 60.0)
        self.image_seconds.setDecimals(2)
        self.image_seconds.setSingleStep(.1)
        self.image_seconds.setSuffix(" s")
        self.image_seconds.setValue(max(3.0, min_image_s))
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
        electrode_grid = QWidget()
        electrode_layout = QGridLayout(electrode_grid)
        electrode_layout.setContentsMargins(0, 0, 0, 0)
        self.electrode_boxes: list[QCheckBox] = []
        for index, (channel, name) in enumerate(zip(config.eeg.device_channels, config.eeg.electrodes, strict=True)):
            box = QCheckBox(f"{channel + 1} · {name}")
            box.setChecked(True)
            box.setToolTip(f"Use Unicorn electrode {channel + 1} ({name}) for model features")
            electrode_layout.addWidget(box, index // 4, index % 4)
            self.electrode_boxes.append(box)
        form.addRow("Model electrodes", electrode_grid)
        self.electrode_status = QLabel("All eight electrodes enter model features; diagnostics show all eight.")
        self.electrode_status.setWordWrap(True)
        form.addRow("", self.electrode_status)
        self.warmup = QCheckBox("Wait before trials (not OSCAR calibration)")
        self.warmup.setChecked(True)
        form.addRow("Before trials", self.warmup)
        self.warmup_seconds = QSpinBox()
        self.warmup_seconds.setRange(2, 120)
        self.warmup_seconds.setValue(12)
        self.warmup_seconds.setSuffix(" s")
        form.addRow("Waiting duration", self.warmup_seconds)
        model_grid = QWidget()
        model_layout = QGridLayout(model_grid)
        model_layout.setContentsMargins(0, 0, 0, 0)
        self.train_boxes: dict[tuple[str, str], QCheckBox] = {}
        for target_index, target in enumerate(config.targets):
            row = target_index * 3
            model_layout.addWidget(QLabel(f"{target.name.title()} ({target.classes[1]})"), row, 0, 1, 2)
            for index, (key, spec) in enumerate(classifiers.CLASSIFIERS.items()):
                box = QCheckBox(spec.label)
                box.setChecked(target.name == "valence" and key in config.classifiers)
                self.train_boxes[(target.name, key)] = box
                model_layout.addWidget(box, row + 1 + index // 2, index % 2)
                box.toggled.connect(self.update_model_limit)
        form.addRow("Train target / model", model_grid)
        self.update_model_limit()
        model_note = QLabel("Choose up to three models for each target. All selected pairs use the same EEG trials. Each predicts before refitting.")
        model_note.setWordWrap(True)
        form.addRow("", model_note)
        settings_layout.addLayout(form)
        self.dataset_status = QLabel()
        self.dataset_status.setWordWrap(True)
        settings_layout.addWidget(self.dataset_status)
        self.refresh_images = QPushButton("Refresh image counts")
        settings_layout.addWidget(self.refresh_images)
        self.config_note = QLabel()
        self.config_note.setWordWrap(True)
        self.config_note.setObjectName("warning")
        settings_layout.addWidget(self.config_note)
        balance = f"the {config.balance_by} classes" if config.balance_by else f"the {self.n_groups} dataset folders"
        self.detail = QLabel(f"Finite runs are exactly balanced across {balance} and shuffled; "
                             "continuous runs draw a group uniformly at random. Each target/classifier pair predicts a trial "
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
        self.start = QPushButton("Start session")
        self.stop = QPushButton("Stop session")
        self.save = QPushButton("Save results")
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
        self.stop_hint.setWordWrap(True)
        layout.addWidget(self.stop_hint)
        self.start.clicked.connect(self.start_requested)
        self.stop.clicked.connect(self.stop_requested)
        self.save.clicked.connect(self.save_requested)
        self.discard.clicked.connect(self.discard_requested)
        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
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

    def heading_session(self, session: str):
        self.heading.setText({SESSION_TRAINING: "Scene emotion training",
                              SESSION_VALIDATION: "Validation · frozen models on held-out images",
                              SESSION_TEST: "Test · match the target valence/arousal"}[session])

    def show_validation(self, scores: ValidationScores):
        """Validation accuracy per frozen model, with balanced accuracy and the majority-class baseline."""
        keys = sorted(scores.models)
        self.save_boxes = {}
        self.accuracy.setRowCount(len(keys))
        self.accuracy.setHorizontalHeaderLabels(("Target (+)", "Model", "Validation acc · balanced", "N",
                                                 "Majority baseline", ""))
        for row, (target, clf) in enumerate(keys):
            accuracy, balanced, n = scores.accuracy((target, clf))
            baseline = scores.majority_baseline((target, clf))
            cells = (f"{target} ({self.class1_names.get(target, '+')})",
                     classifiers.CLASSIFIERS[clf].label if clf in classifiers.CLASSIFIERS else clf,
                     "—" if accuracy is None else f"{accuracy:.1f}% · {balanced:.1f}%",
                     str(n), "—" if baseline is None else f"{baseline:.1f}%", "")
            for col, text in enumerate(cells):
                self.accuracy.removeCellWidget(row, col)
                self.accuracy.setItem(row, col, QTableWidgetItem(text))

    def selected_model_pairs(self) -> list[tuple[str, str]]:
        return [pair for pair, box in self.train_boxes.items() if box.isChecked()]

    def selected_electrode_positions(self) -> tuple[int, ...]:
        return tuple(index for index, box in enumerate(self.electrode_boxes) if box.isChecked())

    def update_model_limit(self):
        if self.session.currentData() != SESSION_TRAINING:
            return
        for target in self.class1_names:
            checked = sum(box.isChecked() for (name, _), box in self.train_boxes.items() if name == target)
            for (name, _), box in self.train_boxes.items():
                if name == target:
                    box.setEnabled(box.isChecked() or checked < 3)

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
        self.accuracy.setHorizontalHeaderLabels(("Target (+)", "Model", f"Last {ROLLING} correct", "N", "P(+)", "Save"))
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
    MODE_BAND = "One band · all electrodes"
    MODE_ELECTRODE = "One electrode · all bands"
    MODE_FD = "Fractal dimension · all electrodes"
    MODE_SINGLE = "Single feature"

    def __init__(self, extractor: FeatureExtractor):
        super().__init__()
        self.extractor = extractor
        feature_names = extractor.names
        channels = extractor.channels
        self.channels = channels
        self.fs = extractor.eeg.fs
        layout = QVBoxLayout(self)
        heading = QLabel("EEG diagnostics and features")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        self.note = QLabel("Waiting for real Unicorn EEG. Diagnostics run before training.")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.oscar_toggle = QCheckBox("Use OSCAR artifact removal (reconnects EEG)")
        self.oscar_toggle.setChecked(True)
        layout.addWidget(self.oscar_toggle)
        quality_title = QLabel("Electrode signal checks")
        quality_title.setObjectName("heading")
        layout.addWidget(quality_title)
        self.quality_map = ElectrodeMap(channels)
        layout.addWidget(self.quality_map)
        self.quality_labels: list[QLabel] = []
        quality_grid = QGridLayout()
        for index, name in enumerate(channels):
            label = QLabel(f"{name}: Waiting")
            quality_grid.addWidget(label, index // 2, index % 2)
            self.quality_labels.append(label)
        layout.addLayout(quality_grid)
        quality_note = QLabel("Green/Good = provisional amplitude check; red/Bad = flat, unusually large, or invalid. This is not an impedance or scalp-contact measurement. Ratings use a filtered 2 s window of the displayed stream.")
        quality_note.setWordWrap(True)
        layout.addWidget(quality_note)
        self.set_quality(None)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        diagnostic = QWidget()
        diagnostic_layout = QVBoxLayout(diagnostic)
        self.raw_note = QLabel("Four seconds of real EEG. Each trace is median-centered for display; vertical spacing is 200 µV. Invalid scans appear as gaps. Paused during sessions.")
        self.raw_note.setWordWrap(True)
        diagnostic_layout.addWidget(self.raw_note)
        self.raw_plot = pg.PlotWidget()
        self.raw_plot.setBackground("#101827")
        self.raw_plot.setLabel("bottom", "Seconds before now")
        self.raw_plot.setLabel("left", "EEG channels (µV)")
        self.raw_plot.showGrid(x=True, y=False, alpha=.2)
        self.raw_plot.setXRange(-4, 0, padding=0)
        self.raw_plot.setYRange(-100, (len(channels) - 1) * 200 + 100, padding=0)
        self.raw_plot.getAxis("left").setTicks([[(float((len(channels) - 1 - i) * 200), name)
                                                for i, name in enumerate(channels)]])
        self.raw_curves = [self.raw_plot.plot(pen=pg.mkPen(CURVE_COLORS[i % len(CURVE_COLORS)], width=1.5))
                           for i in range(len(channels))]
        self.raw_sample_count = 0
        diagnostic_layout.addWidget(self.raw_plot, 1)
        self.tabs.addTab(diagnostic, "Live EEG")
        feature_page = QWidget()
        feature_layout = QVBoxLayout(feature_page)
        feature_note = QLabel("Diagnostic features cover all eight electrodes. Model features use the electrodes selected in the training window. Values update during sessions and freeze afterward.")
        feature_note.setWordWrap(True)
        feature_layout.addWidget(feature_note)
        controls = QHBoxLayout()
        self.mode = QComboBox()
        modes = []
        if extractor.indices("band_power"):
            modes += [self.MODE_BAND, self.MODE_ELECTRODE]
        if extractor.indices("fractal_dimension"):
            modes.append(self.MODE_FD)
        self.mode.addItems([*modes, self.MODE_SINGLE])
        self.band = QComboBox()
        self.band.addItems([name for name, _ in extractor.eeg.bands_hz])
        self.electrode = QComboBox()
        self.electrode.addItems(channels)
        self.selector = QComboBox()
        self.selector.addItems(feature_names)
        for label, widget in (("Show", self.mode), ("Band", self.band), ("Electrode", self.electrode),
                              ("Feature", self.selector)):
            caption = QLabel(label)
            widget.setProperty("caption", caption)
            controls.addWidget(caption)
            controls.addWidget(widget)
        controls.addStretch()
        feature_layout.addLayout(controls)
        self.current = QLabel("Current values: —")
        self.current.setWordWrap(True)
        feature_layout.addWidget(self.current)
        # One small plot per electrode/band (shared, linked time axis; own y scale each).
        self.grid = pg.GraphicsLayoutWidget()
        self.grid.setBackground("#101827")
        self.grid.setMinimumHeight(460)
        feature_layout.addWidget(self.grid, 1)
        self.tabs.addTab(feature_page, "Features")
        self.plots: list = []
        self.curves: list = []
        self.shown: list[tuple[int, str]] = []
        self.samples: list[tuple[float, np.ndarray]] = []
        for widget in (self.mode, self.band, self.electrode, self.selector):
            widget.currentIndexChanged.connect(self.select)
        self.select()

    def selected_features(self) -> list[tuple[int, str]]:
        """(feature index, curve label) pairs for the current display mode."""
        mode = self.mode.currentText()
        info = self.extractor.info
        if mode == self.MODE_BAND:
            return [(i, info[i].location) for i in self.extractor.indices("band_power", band=self.band.currentText())]
        if mode == self.MODE_ELECTRODE:
            return [(i, info[i].band) for i in self.extractor.indices("band_power", location=self.electrode.currentText())]
        if mode == self.MODE_FD:
            return [(i, info[i].location) for i in self.extractor.indices("fractal_dimension")]
        return [(self.selector.currentIndex(), self.selector.currentText())]

    def select(self):
        mode = self.mode.currentText()
        for widget, visible in ((self.band, mode == self.MODE_BAND), (self.electrode, mode == self.MODE_ELECTRODE),
                                (self.selector, mode == self.MODE_SINGLE)):
            widget.setVisible(visible)
            widget.property("caption").setVisible(visible)
        self.shown = self.selected_features()
        family = self.extractor.info[self.shown[0][0]].family if self.shown else ""
        units = {"band_power": "log10 power (µV²)", "region_band_power": "log10 power (µV²)",
                 "fractal_dimension": "Higuchi FD", "erp": "µV"}.get(family, "")
        self.grid.clear()
        self.plots, self.curves = [], []
        n = len(self.shown)
        columns = 2 if n > 3 else 1
        rows = -(-n // columns)
        for k, (_, label) in enumerate(self.shown):
            row, col = divmod(k, columns)
            plot = self.grid.addPlot(row=row, col=col)
            plot.setTitle(label, color="#e8effc", size="10pt")
            plot.showGrid(x=True, y=True, alpha=.2)
            if col == 0:
                plot.setLabel("left", units)
            if row == rows - 1 or k + columns >= n:
                plot.setLabel("bottom", "s since plot start")
            if self.plots:
                plot.setXLink(self.plots[0])
            self.plots.append(plot)
            self.curves.append(plot.plot(pen=pg.mkPen(CURVE_COLORS[k % len(CURVE_COLORS)], width=2)))
        self.redraw()

    def set_quality(self, ratings: list[ChannelQuality] | None):
        self.quality_map.set_quality(ratings)
        for name, label, rating in zip(self.channels, self.quality_labels,
                                       self.quality_map.ratings, strict=True):
            state = {True: "Good", False: "Bad", None: "Waiting"}[rating.good]
            color = {True: "#6dd7bf", False: "#ff727a", None: "#a3acbd"}[rating.good]
            label.setText(f"{name}: {state}")
            label.setStyleSheet(f"color: {color}; font-weight: 600")
            label.setToolTip(rating.detail)

    def set_raw(self, samples: np.ndarray):
        """Show real streamed samples without changing the data used for training."""
        data = np.asarray(samples, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != len(self.channels):
            raise ValueError("Unexpected EEG shape for diagnostic plot")
        self.raw_sample_count = len(data)
        x = (np.arange(len(data)) - max(0, len(data) - 1)) / self.fs
        for index, curve in enumerate(self.raw_curves):
            values = data[:, index]
            finite = np.isfinite(values)
            center = float(np.median(values[finite])) if finite.any() else 0.0
            curve.setData(x, values - center + (len(self.channels) - 1 - index) * 200)

    def clear_raw(self):
        self.raw_sample_count = 0
        for curve in self.raw_curves:
            curve.setData([], [])

    def add(self, stamp: float, values: np.ndarray):
        self.samples.append((stamp, values))
        self.samples = [item for item in self.samples if item[0] >= stamp - 60]
        self.redraw()

    def clear(self):
        self.samples.clear()
        self.set_quality(None)
        self.clear_raw()
        self.redraw()

    def redraw(self):
        if not self.samples:
            for plot, curve, (_, label) in zip(self.plots, self.curves, self.shown):
                curve.setData([], [])
                plot.setTitle(label, color="#e8effc", size="10pt")
            self.current.setText("Current values: —")
            return
        start = self.samples[0][0]
        times = [t - start for t, _ in self.samples]
        latest = self.samples[-1][1]
        for plot, curve, (index, label) in zip(self.plots, self.curves, self.shown):
            curve.setData(times, [float(v[index]) for _, v in self.samples])
            plot.setTitle(f"{label}   <b>{latest[index]:.3f}</b>", color="#e8effc", size="10pt")
        self.current.setText("Current values: " + " · ".join(f"{label} {latest[i]:.3f}" for i, label in self.shown))


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
    marker_code: int
    paint_to_marker_ms: float


def train_epoch(extractor: FeatureExtractor, spec: EpochSpec, trainer: MultiTrainer | None,
                epoch: np.ndarray, labels: dict[str, int]):
    """Process one trial without calling Qt objects from the worker thread."""
    started = time.perf_counter()
    try:
        vector = extractor.extract(epoch, spec)
        if np.ptp(epoch, axis=0).max() > 500 or np.ptp(epoch, axis=0).min() < .05:
            raise ValueError("artifact or flat channel")
    except ValueError:
        return None, None, (time.perf_counter() - started) * 1000
    predictions = trainer.add(vector, labels) if trainer is not None else None
    return vector, predictions, (time.perf_counter() - started) * 1000


class MainWindow(QMainWindow):
    def __init__(self, root: Path, config: dict | None = None):
        super().__init__()
        self.root = root
        self.config = AppConfig.from_dict(CONFIG if config is None else config, root)
        self.display_extractor = FeatureExtractor(self.config.eeg)
        self.extractor = self.display_extractor
        self.model_channel_indices = tuple(range(len(self.config.eeg.electrodes)))
        self.quality = SignalQuality(self.config.eeg)
        self.setWindowTitle("Hot Brain · scene training")
        self.resize(720, 900)
        self.training = TrainingPanel(self.config, self.extractor.min_image_duration_ms / 1000)
        self.device = None
        self.presentation = PresentationPanel()
        self.epoch_spec = self.extractor.epoch_spec(round(self.training.image_seconds.value() * 1000))
        for control in (self.training.image_seconds, self.training.fixation_min, self.training.fixation_max):
            control.valueChanged.connect(self.update_timing_note)
        self.update_timing_note()
        self.features = FeaturePanel(self.display_extractor)
        self.features.oscar_toggle.setChecked(self.training.oscar.isChecked())
        self.features.oscar_toggle.toggled.connect(self.training.oscar.setChecked)
        self.training.oscar.toggled.connect(self.features.oscar_toggle.setChecked)
        self.setCentralWidget(self.training)
        self.presentation_window = PanelWindow("Hot Brain · image presentation", self.presentation,
                                                self.stop_run)
        self.features_window = PanelWindow("Hot Brain · EEG diagnostics and features", self.features)
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
        self._make_device()
        self.training.serial.editingFinished.connect(self.configure_device_from_controls)
        self.training.oscar.toggled.connect(self.configure_device_from_controls)
        warnings = self.config.eeg.warnings()
        self.training.config_note.setText("\n".join(f"Config warning: {w}" for w in warnings))
        self.training.config_note.setVisible(bool(warnings))
        self.split_train: dict[str, list[StimulusImage]] = {}
        self.split_validation: dict[str, list[StimulusImage]] = {}
        self.dataset: Dataset | None = self.load_dataset(show_errors=False)
        self.test_dataset: Dataset | None = None
        self.eval_models: dict = {}
        self.eval_info: dict = {}
        self.validation: ValidationScores | None = None
        self.smoother = Smoother(self.config.feedback_smoothing)
        self.decode_log: list[dict] = []
        self.current_target: tuple[float, float] | None = None
        self.current_trial: int | None = None
        self.running_session = SESSION_TRAINING
        self.last_decode_time = 0.0
        self.plan: TrialPlan | OncePlan | None = None
        self.trainer = MultiTrainer([], [], len(self.extractor.names))
        self.pending: list[PendingEpoch] = []
        # One worker preserves test-before-train order and keeps model fits off Qt's event loop.
        self.training_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eeg-training")
        self.training_queue: deque[tuple[PendingEpoch, np.ndarray, float]] = deque()
        self.training_future: Future | None = None
        self.training_item: tuple[PendingEpoch, np.ndarray, float] | None = None
        self.next_trial: StimulusImage | None = None
        self.active = False
        self.finishing = False
        self.ready_for_new_run = True
        self.evaluation_valid = False
        self.split_available = False
        self.stop_reason: str | None = None
        self.shown = 0
        self.usable = 0
        self.skipped = 0
        self.last_feature_time = 0.0
        self.last_quality_time = 0.0
        self.last_raw_time = 0.0
        self.last_raw_samples = 0
        self.run_metadata: dict = {}
        self.trial_log: list[dict] = []
        self.epochs: list[np.ndarray] = []
        self.epoch_trial_numbers: list[int] = []
        self.epoch_labels: list[dict[str, int]] = []
        self.epoch_values: list[dict[str, float]] = []
        self.epoch_groups: list[str] = []
        self.run_start_clock = 0.0
        self.training.session.currentIndexChanged.connect(self._on_session_changed)
        self.training.persistent.toggled.connect(self._on_session_changed)
        self.training.persistent.toggled.connect(self._refresh_evaluation_options)
        self.training.count.valueChanged.connect(self._refresh_evaluation_options)
        self.training.evaluation.currentIndexChanged.connect(self._refresh_evaluation_options)
        for box in self.training.electrode_boxes:
            box.toggled.connect(self._on_electrodes_changed)
        self.training.load_models.clicked.connect(self.load_saved_models)
        self.training.refresh_images.clicked.connect(self.refresh_dataset)
        self._refresh_evaluation_options()
        self._on_session_changed()
        self.clock = QTimer(self)
        self.clock.setInterval(100)
        self.clock.timeout.connect(self.poll)
        self.clock.start()
        self.warmup_timer = self._precise_timer(self.begin_fixation)
        self.fixation_timer = self._precise_timer(self.show_trial)
        self.image_timer = self._precise_timer(self.end_image)
        self.training.start.setEnabled(False)
        self.training.status.setText("Waiting for live EEG samples before a session can start.")
        self.setStyleSheet("QMainWindow,QWidget{background:#182335;color:#e8effc;font-size:13px} QLabel#heading{font-size:24px;font-weight:700;margin:4px 0 10px} QLabel#metric{font-size:18px;font-weight:600;color:#6dd7bf} QLabel#warning{color:#ffb86b} QLabel#status{font-size:16px;font-weight:600} QPushButton{background:#29466c;border-radius:7px;padding:8px 12px} QPushButton:hover{background:#376293} QPushButton:disabled{background:#354155;color:#8d9aaf} QSpinBox,QDoubleSpinBox,QComboBox{background:#25364e;padding:5px;border:1px solid #627795} QTabBar::tab{padding:9px 17px}")

    def _precise_timer(self, callback):
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.timeout.connect(callback)
        return timer

    @property
    def _settings_widgets(self) -> list[QWidget]:
        t = self.training
        return [t.participant, t.serial, t.count, t.persistent, t.stop_key, t.oscar, t.warmup, t.warmup_seconds,
                t.image_seconds, t.fixation_min, t.fixation_max, t.evaluation,
                t.session, t.load_models, t.refresh_images, t.feedback_classifier,
                *t.electrode_boxes, *t.train_boxes.values()]

    # ------------------------------------------------------------------ EEG source
    def _make_device(self):
        """Create the Unicorn EEG source."""
        self.device = DeviceManager(self.config.eeg, self.training.serial.text().strip() or None)
        self.device.status_changed.connect(self.training.device_status.setText)
        self.device.connected_changed.connect(self.on_connection)
        self.training.device_status.setText("Checking for Unicorn…")

    @property
    def _source_name(self) -> str:
        return f"Unicorn {self.device.serial or '(first available)'}"

    def load_dataset(self, show_errors: bool = True) -> Dataset | None:
        try:
            dataset = load_dataset(self.config)
        except (OSError, ValueError) as exc:
            self.training.dataset_status.setText(f"Dataset error: {exc}")
            if show_errors:
                QMessageBox.warning(self, "Dataset", str(exc))
            return None
        self.split_train, self.split_validation = split_dataset(
            dataset, self.config.validation_fraction, self.config.split_seed)
        n_train = sum(len(v) for v in self.split_train.values())
        n_val = sum(len(v) for v in self.split_validation.values())
        self.training.dataset_status.setText(
            f"Dataset ({self.config.data_dir.name}/): {dataset.summary()} · optional split: "
            f"{n_train} training / {n_val} validation images (seed {self.config.split_seed})")
        return dataset

    def refresh_dataset(self):
        if self.active or self.finishing:
            return
        self.dataset = self.load_dataset()
        self._refresh_evaluation_options()

    def _image_files(self, groups: dict[str, list[StimulusImage]]) -> set[str]:
        return {str(image.path.relative_to(self.config.data_dir))
                for images in groups.values() for image in images}

    def _validation_eligible(self) -> bool:
        if not self.split_available or self.training.evaluation.currentData() != EVAL_SPLIT or not self.eval_models:
            return False
        if self.eval_info.get("evaluation_mode") != EVAL_SPLIT:
            return False
        trained = set(self.eval_info.get("training_image_files", ()))
        return bool(trained) and not (trained & self._image_files(self.split_validation))

    def _refresh_validation_availability(self):
        allowed = self._validation_eligible()
        self.training.session.model().item(1).setEnabled(allowed)
        if not allowed and self.session_type == SESSION_VALIDATION:
            self.training.session.setCurrentIndex(0)

    def _refresh_evaluation_options(self):
        dataset = self.dataset
        if dataset is None:
            capacity = 0
            self.split_available = False
        else:
            capacity = balanced_unique_capacity(dataset.groups)
            self.split_available = (self.config.validation_fraction > 0 and
                                    all(self.split_train.get(group) and self.split_validation.get(group)
                                        for group in dataset.groups))
        count = self.training.count.value()
        group_count = len(dataset.groups) if dataset is not None else 0
        prequential_ok = (not self.training.persistent.isChecked() and group_count > 0
                          and count <= capacity and count % group_count == 0)
        self.training.evaluation.model().item(0).setEnabled(prequential_ok)
        self.training.evaluation.model().item(1).setEnabled(self.split_available)
        if not prequential_ok and self.training.evaluation.currentData() == EVAL_PREQUENTIAL and self.split_available:
            self.training.evaluation.setCurrentIndex(1)
            return
        mode = self.training.evaluation.currentData()
        self.evaluation_valid = prequential_ok if mode == EVAL_PREQUENTIAL else self.split_available
        if mode == EVAL_SPLIT and not self.training.persistent.isChecked() and group_count:
            self.evaluation_valid = self.evaluation_valid and count % group_count == 0
        if dataset is None:
            note = "No usable image dataset."
        elif mode == EVAL_PREQUENTIAL:
            note = (f"{count} unique presentations, balanced across {group_count} groups. "
                    f"Maximum without repeats: {capacity}. Live rolling test-before-train accuracy is the evaluation."
                    if prequential_ok else
                    f"Prequential unavailable: choose a finite multiple of {group_count} no greater than {capacity}.")
        else:
            train_capacity = balanced_unique_capacity(self.split_train)
            note = (f"Held-out validation images never enter training. The training subset supports "
                    f"{train_capacity} balanced presentations without repeats; longer runs repeat training images. "
                    "Run Validation afterward for the held-out score."
                    if self.split_available else
                    "Split unavailable: each group needs at least one training and one validation image.")
        if not self.evaluation_valid:
            note += " No valid evaluation mode for these settings."
        self.training.evaluation_status.setText(note)
        self._refresh_validation_availability()
        self.update_start_availability()

    def _on_electrodes_changed(self):
        if self.active or self.finishing:
            return
        positions = self.training.selected_electrode_positions()
        if not positions:
            self.training.electrode_status.setText("Select at least one electrode for the model.")
            self.update_start_availability()
            return
        if positions == self.model_channel_indices:
            names = ", ".join(self.config.eeg.electrodes[index] for index in positions)
            self.training.electrode_status.setText(
                f"Model uses {len(positions)} electrode(s): {names}. Diagnostics show all eight."
                if self.extractor.names else "This selection leaves no configured model features. Select more electrodes.")
            self.update_start_availability()
            return
        self.model_channel_indices = positions
        self.extractor = FeatureExtractor(self.config.eeg.select_electrodes(positions))
        self.epoch_spec = self.extractor.epoch_spec(round(self.training.image_seconds.value() * 1000))
        names = ", ".join(self.config.eeg.electrodes[index] for index in positions)
        self.training.electrode_status.setText(
            f"Model uses {len(positions)} electrode(s): {names}. Diagnostics show all eight."
            if self.extractor.names else "This selection leaves no configured model features. Select more electrodes.")
        self.eval_models = {}
        self.eval_info = {}
        self.training.feedback_classifier.clear()
        self.training.models_status.setText("Electrodes changed. Load matching models or train again.")
        self._refresh_validation_availability()
        self.update_start_availability()

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
        if not connected:
            self.features.clear()
            self.last_raw_samples = 0
            self.last_raw_time = 0.0
            self.last_quality_time = 0.0
            self.features.note.setText("No live EEG stream. Diagnostics and feature values have been cleared.")
            if self.active:
                self.stop_run("EEG stream disconnected; session stopped.")
        elif not self.active:
            self.features.note.setText("Unicorn opened. Waiting for current EEG samples.")
        self.update_start_availability()

    def configure_device_from_controls(self):
        if self.active or self.finishing:
            return
        oscar = self.training.oscar.isChecked()
        serial = self.training.serial.text().strip() or None
        if self.device.oscar != oscar or self.device.serial != serial:
            self.features.clear()
            self.last_raw_samples = 0
            self.last_raw_time = 0.0
            self.last_quality_time = 0.0
            self.features.note.setText("Reconnecting EEG with updated device settings…")
        self.device.configure(oscar, serial)
        self.update_start_availability()

    def update_start_availability(self):
        session = self.session_type
        valid = (self.evaluation_valid if session == SESSION_TRAINING else
                 self._validation_eligible() if session == SESSION_VALIDATION else bool(self.eval_models))
        self.training.start.setEnabled(
            self.ready_for_new_run and not self.active and not self.finishing and valid
            and bool(self.training.selected_electrode_positions()) and bool(self.extractor.names)
            and self.device.stream_ready())

    def ensure_eeg_stream(self) -> bool:
        if self.device.stream_live():
            return True
        if self.active:
            self.stop_run("EEG stream stopped; session stopped.")
        return False

    # ------------------------------------------------------------------ sessions
    @property
    def session_type(self) -> str:
        return self.training.session.currentData()

    def _on_session_changed(self):
        session = self.session_type
        if not (self.active or self.finishing):
            self.training.count.setEnabled(session == SESSION_TRAINING and not self.training.persistent.isChecked())
            self.training.persistent.setEnabled(session == SESSION_TRAINING)
            self.training.evaluation.setEnabled(session == SESSION_TRAINING)
            for box in self.training.electrode_boxes:
                box.setEnabled(session == SESSION_TRAINING)
            for box in self.training.train_boxes.values():
                box.setEnabled(session == SESSION_TRAINING)
            if session == SESSION_TRAINING:
                self.training.update_model_limit()
        self.presentation.plane.setVisible(session != SESSION_TRAINING)
        self.training.heading_session(session)
        if hasattr(self, "evaluation_valid"):
            self.update_start_availability()

    def _set_eval_models(self, models: dict, info: dict, source: str):
        """Models used (frozen) by validation and test sessions."""
        self.eval_models = models
        self.eval_info = {**info, "source": source}
        classifiers_present = sorted({clf for _, clf in models})
        combo = self.training.feedback_classifier
        combo.clear()
        for clf in classifiers_present:
            combo.addItem(classifiers.CLASSIFIERS[clf].label if clf in classifiers.CLASSIFIERS else clf, clf)
        preferred = next((c for c in self.config.classifiers if c in classifiers_present), None)
        if preferred is not None:
            combo.setCurrentIndex(classifiers_present.index(preferred))
        duration = info.get("image_duration_s")
        self.training.models_status.setText(
            f"{len(models)} model(s) from {source}" + (f" · image {duration:g} s" if duration else ""))
        self._refresh_validation_availability()
        self.update_start_availability()

    def load_saved_models(self):
        start = str(self.root / "saved_models")
        path, _ = QFileDialog.getOpenFileName(self, "Load trained models (manifest .json)", start, "Manifest (*.json)")
        if not path:
            return
        try:
            models, manifest = load_models(Path(path))
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Load models", str(exc)); return
        if tuple(manifest.get("feature_names", ())) != self.extractor.names:
            QMessageBox.warning(self, "Load models", "These models were trained on different features than the "
                                "current config produces. Retrain or restore the matching config.")
            return
        model_channels = tuple(manifest.get("model_device_channels", manifest.get("device_channels", ())))
        if model_channels != self.extractor.eeg.device_channels:
            QMessageBox.warning(self, "Load models", "These models used a different electrode selection.")
            return
        self._set_eval_models(models, {"image_duration_s": manifest.get("image_duration_s"),
                                       "manifest": Path(path).name,
                                       "evaluation_mode": manifest.get("evaluation_mode"),
                                       "training_image_files": manifest.get("training_image_files", ())}, Path(path).name)

    def _display_targets(self) -> tuple[str, str]:
        names = [t.name for t in self.config.targets]
        return (names[0], names[1] if len(names) > 1 else names[0])

    def _target_point(self, dataset: Dataset, item: StimulusImage) -> tuple[float, float]:
        x_name, y_name = self._display_targets()
        return dataset.normalized(x_name, item.values[x_name]), dataset.normalized(y_name, item.values[y_name])

    def _session_images(self, session: str) -> dict[str, list[StimulusImage]] | None:
        if session == SESSION_TRAINING:
            return self.dataset.groups if self.training.evaluation.currentData() == EVAL_PREQUENTIAL else self.split_train
        if session == SESSION_VALIDATION:
            return self.split_validation
        try:
            self.test_dataset = load_dataset(self.config, self.config.test_datasets, reference=self.dataset)
            return self.test_dataset.groups
        except (OSError, ValueError) as exc:
            if not self._validation_eligible():
                QMessageBox.warning(self, "Test images", f"Test images are not available ({exc}). Add a separate test dataset.")
                return None
            answer = QMessageBox.question(
                self, "Test images",
                f"Test images are not available yet ({exc}).\n\nAdd them to data/{self.config.test_datasets[0]} "
                f"with data/{self.config.test_datasets[0]}.csv (columns image_name, valence, arousal).\n\n"
                "Run this test session with the validation images instead?")
            if answer != QMessageBox.StandardButton.Yes:
                return None
            self.test_dataset = self.dataset
            return self.split_validation

    def start_run(self):
        if self.active or self.finishing or not self.ready_for_new_run:
            return
        self.configure_device_from_controls()
        if not self.device.stream_ready():
            self.training.status.setText("Waiting for current EEG samples. A session cannot start without a live stream.")
            self.update_start_availability()
            return
        self.dataset = self.load_dataset()
        if self.dataset is None:
            return
        self._refresh_evaluation_options()
        session = self.session_type
        if session == SESSION_TRAINING and not self.evaluation_valid:
            self.training.status.setText("Choose an available evaluation mode and trial count.")
            return
        if session == SESSION_VALIDATION and not self._validation_eligible():
            self.training.status.setText("Validation needs models trained without the held-out images.")
            return
        if not self.training.selected_electrode_positions():
            self.training.status.setText("Select at least one model electrode.")
            return
        if not self.extractor.names:
            self.training.status.setText("The selected electrodes produce no configured model features.")
            return
        fixation_min = self.training.fixation_min.value()
        fixation_max = self.training.fixation_max.value()
        if fixation_min > fixation_max:
            QMessageBox.warning(self, "Fixation timing", "Fixation minimum must be at most the maximum.")
            return
        model_pairs = self.training.selected_model_pairs()
        if session == SESSION_TRAINING and not model_pairs:
            QMessageBox.warning(self, "Models", "Select at least one target and model.")
            return
        if session != SESSION_TRAINING:
            if not self.eval_models:
                QMessageBox.warning(self, "Models", "Run a training session first or load saved models.")
                return
            trained_s = self.eval_info.get("image_duration_s")
            if trained_s and abs(trained_s - self.training.image_seconds.value()) > 1e-6:
                QMessageBox.warning(self, "Image duration", f"The models were trained with {trained_s:g} s images; "
                                    "set the same image duration so the features match.")
                return
        try:
            self.epoch_spec = self.extractor.epoch_spec(round(self.training.image_seconds.value() * 1000))
        except ValueError as exc:
            QMessageBox.warning(self, "Image duration", str(exc)); return
        groups = self._session_images(session)
        if groups is None:
            return
        for items in groups.values():
            for item in items:
                if not QImageReader(str(item.path)).canRead():
                    QMessageBox.warning(self, "Image error", f"Cannot read {item.path}")
                    return
        total = None if self.training.persistent.isChecked() else self.training.count.value()
        try:
            self.plan = (TrialPlan(groups, total, allow_repeats=self.training.evaluation.currentData() == EVAL_SPLIT)
                         if session == SESSION_TRAINING else OncePlan(groups))
        except ValueError as exc:
            QMessageBox.warning(self, "Session images", str(exc)); return
        self.running_session = session
        retention = self.training.image_seconds.value() + 5
        if session == SESSION_TEST:
            retention = max(retention, self.config.test_trial_s + 5)
        self.device.set_retention(retention)
        self.training.plot.setVisible(session == SESSION_TRAINING)
        self.training.weights.setVisible(session == SESSION_TRAINING)
        if session == SESSION_TRAINING:
            target_names = list(dict.fromkeys(target for target, _ in model_pairs))
            classifier_keys = list(dict.fromkeys(clf for _, clf in model_pairs))
            self.trainer = MultiTrainer(target_names, classifier_keys, len(self.extractor.names),
                                        selected_pairs=model_pairs)
            self.training.configure_models(list(self.trainer.models), self.extractor.names)
            self.training.plot.setLabel("left", "Prequential correct" if self.training.evaluation.currentData() == EVAL_PREQUENTIAL
                                        else "Training stream correct", units="%")
        else:
            target_names = []
            classifier_keys = sorted({clf for _, clf in self.eval_models})
            self.training.reset_save_selection()
            self.validation = ValidationScores(dict(self.eval_models))
            self.training.show_validation(self.validation)
        self.presentation.plane.clear()
        self.presentation.plane.set_axes(*self._display_targets())
        self.smoother.reset()
        self.decode_log = []
        self.current_target = None
        self.current_trial = None
        self.pending.clear()
        self.training_queue.clear()
        self.training_future = None
        self.training_item = None
        self.shown = self.usable = self.skipped = 0
        self.ready_for_new_run = False
        self.stop_reason = None
        self.training.start.setEnabled(False)
        self.training.stop.setEnabled(True)
        self.training.settings_scroll.setVisible(False)
        self.stop_shortcut.setEnabled(True)
        self.training.save.setEnabled(False)
        self.training.discard.setEnabled(False)
        for widget in self._settings_widgets:
            widget.setEnabled(False)
        self.features.oscar_toggle.setEnabled(False)
        self.active = True
        self.finishing = False
        self.features.clear()
        self.last_raw_samples = 0
        self.last_raw_time = 0.0
        self.features.note.setText("Training in progress. Live EEG plot paused to protect image timing.")
        self.last_feature_time = 0.0
        self.last_quality_time = 0.0
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
        session_dataset = self.test_dataset if session == SESSION_TEST else self.dataset
        evaluation_mode = self.training.evaluation.currentData() if session == SESSION_TRAINING else self.eval_info.get("evaluation_mode")
        model_eeg = self.extractor.eeg
        self.run_metadata = {"session": session,
                             "started_at": datetime.now().astimezone().isoformat(),
                             "participant_id": self.training.participant.text().strip() or "unnamed",
                             "eeg_source": "unicorn",
                             "unicorn_serial": self.training.serial.text().strip() or None,
                             "mode": ("persistent" if total is None else "balanced_finite") if session == SESSION_TRAINING else "each_image_once",
                             "requested_trials": total if session == SESSION_TRAINING else self.plan.total,
                             "evaluation_mode": evaluation_mode,
                             "oscar_enabled": self.training.oscar.isChecked(),
                             "image_duration_s": self.epoch_spec.image_duration_ms / 1000,
                             "fixation_min_s": fixation_min, "fixation_max_s": fixation_max,
                             "test_trial_s": self.config.test_trial_s if session == SESSION_TEST else None,
                             "test_rest_s": self.config.test_rest_s if session == SESSION_TEST else None,
                             "epoch_start_ms": START_MS, "epoch_end_ms": self.epoch_spec.end_ms,
                             "epoch_sample_count": self.epoch_spec.n_samples,
                             "unicorn_device_delay_ms": 40,
                             "device_delay_samples": self.device.ring.device_delay_samples,
                             "fs": eeg.fs, "channels": eeg.electrodes, "device_channels": eeg.device_channels,
                             "model_electrodes": model_eeg.electrodes,
                             "model_device_channels": model_eeg.device_channels,
                             "feature_names": self.extractor.names,
                             "classifiers": classifier_keys if session == SESSION_TRAINING else sorted({c for _, c in self.eval_models}),
                             "models_source": None if session == SESSION_TRAINING else self.eval_info.get("source"),
                             "live_display_classifier": None if session == SESSION_TRAINING else self.training.feedback_classifier.currentData(),
                             "training_targets": target_names,
                             "training_model_pairs": [f"{t}/{c}" for t, c in model_pairs] if session == SESSION_TRAINING else [],
                             "config": self.config.raw,
                             "split": {"validation_fraction": self.config.validation_fraction if evaluation_mode == EVAL_SPLIT else 0,
                                       "seed": self.config.split_seed,
                                       "training_images": sum(len(v) for v in groups.values()) if session == SESSION_TRAINING else None,
                                       "validation_images": sum(len(v) for v in self.split_validation.values()) if evaluation_mode == EVAL_SPLIT else 0},
                             "oscar_settling_requested_seconds": self.training.warmup_seconds.value() if self.training.warmup.isChecked() else 0,
                             "oscar_settling_applied_seconds": 0,
                             "dataset": session_dataset.manifest(),
                             "session_images": [i.path.name for items in groups.values() for i in items],
                             "training_image_files": sorted(self._image_files(groups)) if session == SESSION_TRAINING else [],
                             "timing_note": "g.Pype places trial markers on the Unicorn sample grid; epoch indices add the documented 40 ms device acquisition-to-host delay. OSCAR delay is handled by the pipeline. The Qt paint callback estimates visible onset; verify screen latency with a photodiode."}
        if self.training.oscar.isChecked() and self.training.warmup.isChecked():
            seconds = self.training.warmup_seconds.value()
            self.run_metadata["oscar_settling_applied_seconds"] = seconds
            self.training.status.setText(f"Pretrial waiting: {seconds} s before first trial")
            self.presentation.stimulus.fixation()
            self.warmup_timer.start(seconds * 1000)
        else:
            self.begin_fixation()

    def begin_fixation(self):
        if not self.active or not self.ensure_eeg_stream():
            return
        self.next_trial = self.plan.next() if self.plan else None
        if self.next_trial is None:
            self.stop_run(); return
        self.presentation.stimulus.fixation()
        self.training.status.setText(f"Fixation before trial {self.shown + 1}")
        if self.running_session == SESSION_TEST:
            delay = round(self.config.test_rest_s * 1000)
        else:
            delay = random.randint(round(self.training.fixation_min.value() * 1000),
                                   round(self.training.fixation_max.value() * 1000))
        self.fixation_timer.start(delay)

    def show_trial(self):
        if not self.active or not self.ensure_eeg_stream() or self.next_trial is None:
            return
        self.presentation.stimulus.image(self.next_trial.path)
        # The paint callback is the closest software-side onset observation.

    def on_image_onset(self, stamp: float):
        if not self.active or not self.ensure_eeg_stream() or self.next_trial is None:
            return
        try:
            marker_code, marker_delay_ms = self.device.mark_image(stamp)
        except Exception as exc:
            self.stop_run(f"Could not place image marker: {exc}.")
            return
        item = self.next_trial
        self.shown += 1
        session = self.running_session
        shown_s = self.config.test_trial_s if session == SESSION_TEST else self.epoch_spec.image_duration_ms / 1000
        if session != SESSION_TEST:
            self.pending.append(PendingEpoch(self.shown, item, stamp, stamp + self.epoch_spec.end_ms / 1000 + 2.0,
                                             marker_code, marker_delay_ms))
        dataset = self.test_dataset if session == SESSION_TEST else self.dataset
        entry = {"number": self.shown, "group": item.group, "image": item.path.name,
                 "labels": {t: dataset.class_name(t, v) for t, v in item.labels.items()},
                 "target_values": dict(item.values),
                 "onset_s_from_run_start": round(stamp - self.run_start_clock, 6),
                 "marker_code": marker_code,
                 "paint_to_marker_ms": round(marker_delay_ms, 3),
                 "eeg": "continuous" if session == SESSION_TEST else "pending"}
        if session != SESSION_TRAINING:
            self.current_target = self._target_point(dataset, item)
            self.current_trial = self.shown
            entry["target_point"] = list(self.current_target)
            plane = self.presentation.plane
            plane.trail.clear()
            plane.set_target(self.current_target)
            if session == SESSION_VALIDATION:
                plane.set_decoded(None)
                plane.set_caption(f"Validation image {self.shown}/{self.plan.total}: waiting for EEG…")
            else:
                plane.set_caption(f"Test image {self.shown}/{self.plan.total}: move ● onto ◯")
        self.trial_log.append(entry)
        self.training.status.setText(f"Image {self.shown}: {item.group}/{item.path.name}")
        self._update_counts()
        self.next_trial = None
        self.image_timer.start(round(shown_s * 1000))

    def end_image(self):
        if not self.active or not self.ensure_eeg_stream():
            return
        if self.running_session == SESSION_TEST:
            self._summarize_test_trial()
            self.current_target = None
            self.current_trial = None
            self.presentation.plane.set_target(None)
            self.presentation.plane.set_caption("Rest")
        self.presentation.stimulus.fixation()
        self.begin_fixation()

    def _summarize_test_trial(self):
        """Mean distance between decoded and target point during the trial (0 = perfect, 2.83 = opposite corners)."""
        rows = [r for r in self.decode_log if r["trial"] == self.current_trial and r["smoothed"][0] is not None]
        if not rows or self.current_trial is None:
            return
        target = np.asarray(self.current_target)
        distances = [float(np.linalg.norm(np.asarray(r["smoothed"]) - target)) for r in rows]
        self.trial_log[self.current_trial - 1].update({"mean_distance": float(np.mean(distances)),
                                                       "final_distance": distances[-1], "n_estimates": len(rows)})

    def _decode_live(self, now: float):
        """Continuous valence/arousal estimate from the latest EEG window (test session)."""
        spec = self.epoch_spec
        _, samples = self.device.ring.recent(max(2, spec.n_samples / spec.fs + 1))
        if len(samples) < spec.n_samples:
            return
        window = samples[-spec.n_samples:, self.model_channel_indices]
        if not np.isfinite(window).all() or np.ptp(window, axis=0).max() > 500:
            self.presentation.plane.set_caption("Artifact or lost samples in EEG window: estimate paused")
            return
        try:
            vector = self.extractor.extract(window, spec)
        except ValueError:
            return
        raw = affect_point(self.eval_models, self.training.feedback_classifier.currentData(), vector,
                           self._display_targets())
        smoothed = self.smoother.update(raw)
        self.presentation.plane.set_decoded(smoothed)
        self.decode_log.append({"t": round(now - self.run_start_clock, 4), "trial": self.current_trial,
                                "target": None if self.current_target is None else list(self.current_target),
                                "raw": list(raw), "smoothed": list(smoothed)})
        if self.current_target is not None and smoothed[0] is not None:
            distance = float(np.linalg.norm(np.asarray(smoothed) - np.asarray(self.current_target)))
            self.presentation.plane.set_caption(
                f"Test image {self.current_trial}/{self.plan.total} · distance {distance:.2f}"
                + ("  ✓ on target" if distance < 0.35 else ""))

    def poll(self):
        now = time.perf_counter()
        self._collect_training_result()
        stream_live = self.device.stream_live()
        start_ready = self.device.stream_ready()
        if self.active and not stream_live:
            self.stop_run("EEG stream stopped or timing became unreliable; training stopped.")
        self.update_start_availability()
        if stream_live:
            device_text = (f"Unicorn {self.device.serial or '(first available)'} streaming EEG · OSCAR "
                           + ("on" if self.device.oscar else "off"))
            if self.training.device_status.text() != device_text:
                self.training.device_status.setText(device_text)
            if start_ready and self.ready_for_new_run and self.training.status.text().startswith("Waiting for"):
                self.training.status.setText("EEG stream ready. Start training.")
        elif self.device.connected:
            issue = self.device.readiness_issue() if hasattr(self.device, "readiness_issue") else "EEG timing unavailable"
            self.training.device_status.setText(f"Unicorn opened, but {issue}. Training disabled.")
        if not stream_live:
            if (self.features.samples or self.features.raw_sample_count
                    or any(rating.good is not None for rating in self.features.quality_map.ratings)):
                self.features.clear()
            self.last_raw_samples = 0
            self.features.note.setText("No live EEG stream. Diagnostics and feature values have been cleared.")
        elif self.active:
            self.features.note.setText("Training in progress. Live EEG plot paused to protect image timing.")
        elif self.finishing:
            self.features.note.setText("Finishing pending EEG trials. Live plot paused.")
        else:
            self.features.note.setText("Live EEG diagnostic. Features remain frozen outside training. OSCAR "
                                       + ("on." if self.device.oscar else "off (unprocessed stream)."))
            if now - self.last_raw_time >= .25:
                self.last_raw_time = now
                count = self.device.ring.received_samples
                if count != self.last_raw_samples:
                    _, raw_samples = self.device.ring.recent(4)
                    self.features.set_raw(raw_samples)
                    self.last_raw_samples = count
        if stream_live and not self.finishing and now - self.last_quality_time >= .5:
            self.last_quality_time = now
            _, quality_samples = self.device.ring.recent(3)
            self.features.set_quality(self.quality.assess(quality_samples))
        if self.active and stream_live and now - self.last_feature_time >= .25:
            spec = self.epoch_spec
            times, samples = self.device.ring.recent(max(2, spec.n_samples / spec.fs + 1))
            if len(samples) >= spec.n_samples and times[-1] > self.last_feature_time:
                try:
                    values = self.display_extractor.extract(samples[-spec.n_samples:], spec)
                    self.features.add(times[-1], values)
                    self.last_feature_time = now
                except ValueError:
                    pass
        if (self.active and stream_live and self.running_session == SESSION_TEST
                and now - self.last_decode_time >= self.config.feedback_update_s):
            self.last_decode_time = now
            self._decode_live(now)
        if self.finishing and not stream_live and self.pending:
            for item in self.pending:
                log = self.trial_log[item.number - 1]
                log["marker_sample_index"] = self.device.ring.marker_sample(item.marker_code)
                log["eeg_onset_sample_index"] = self.device.ring.eeg_onset_sample(
                    item.marker_code, item.paint_to_marker_ms)
                log["eeg"] = "timing_unreliable"
                self.skipped += 1
            self.pending.clear()
            self._update_counts()
        for item in self.pending[:]:
            epoch, epoch_status = (self.device.ring.epoch_result(
                item.marker_code, self.epoch_spec, item.paint_to_marker_ms)
                if self.device.connected else (None, "disconnected"))
            if epoch is not None:
                log = self.trial_log[item.number - 1]
                log["marker_sample_index"] = self.device.ring.marker_sample(item.marker_code)
                log["eeg_onset_sample_index"] = self.device.ring.eeg_onset_sample(
                    item.marker_code, item.paint_to_marker_ms)
                log["eeg"] = "processing"
                self.training_queue.append((item, epoch, time.perf_counter()))
                self.pending.remove(item)
            elif epoch_status in ("invalid_scans", "expired_epoch", "incomplete_epoch") or now >= item.deadline:
                self.skipped += 1
                log = self.trial_log[item.number - 1]
                log["marker_sample_index"] = self.device.ring.marker_sample(item.marker_code)
                log["eeg_onset_sample_index"] = self.device.ring.eeg_onset_sample(
                    item.marker_code, item.paint_to_marker_ms)
                log["eeg"] = epoch_status if epoch_status in (
                    "invalid_scans", "expired_epoch", "incomplete_epoch") else "missing_marker_or_incomplete_epoch"
                self.pending.remove(item)
                self._update_counts()
            else:
                # Keep test-before-train predictions in presentation order.
                break
        self._start_next_training()
        if self.finishing and not self._training_busy():
            self.finish_run()

    def _validate_on(self, vector: np.ndarray, item: PendingEpoch, log: dict):
        results = self.validation.add(vector, item.image.labels)
        log["predictions"] = {f"{t}/{c}": {"predicted": self.dataset.class_name(t, predicted), "p_class1": p,
                                           "correct": predicted == item.image.labels[t]}
                              for (t, c), (predicted, p) in results.items()}
        self.training.show_validation(self.validation)
        point = affect_point(self.eval_models, self.training.feedback_classifier.currentData(), vector,
                             self._display_targets())
        log["decoded_point"] = list(point)
        if item.number == self.current_trial:
            plane = self.presentation.plane
            plane.set_decoded(point)
            x_name, y_name = self._display_targets()
            clf = self.training.feedback_classifier.currentData()
            hits = [name for name in (x_name, y_name) if (name, clf) in results
                    and results[(name, clf)][0] == item.image.labels[name]]
            plane.set_caption(f"Validation image {item.number}/{self.plan.total} · correct: "
                              + (", ".join(hits) if hits else "none"))
    def _training_busy(self) -> bool:
        return bool(self.pending or self.training_queue or self.training_future)

    def _start_next_training(self):
        if self.training_future is not None or not self.training_queue:
            return
        self.training_item = self.training_queue.popleft()
        item, epoch, queued_at = self.training_item
        self.trial_log[item.number - 1]["training_queue_ms"] = round(
            (time.perf_counter() - queued_at) * 1000, 2)
        self.training_future = self.training_executor.submit(
            train_epoch, self.extractor, self.epoch_spec,
            self.trainer if self.running_session == SESSION_TRAINING else None,
            epoch[:, self.model_channel_indices], dict(item.image.labels))

    def _collect_training_result(self):
        future = self.training_future
        if future is None or not future.done():
            return
        item, epoch, _ = self.training_item
        self.training_future = None
        self.training_item = None
        log = self.trial_log[item.number - 1]
        try:
            vector, predictions, elapsed_ms = future.result()
        except Exception as exc:
            self.skipped += 1
            log["eeg"] = "training_error"
            log["training_error"] = f"{type(exc).__name__}: {exc}"
            # A partially updated trainer cannot safely continue the ordered run.
            for queued, _, _ in self.training_queue:
                self.trial_log[queued.number - 1]["eeg"] = "training_aborted"
                self.skipped += 1
            self.training_queue.clear()
            for waiting in self.pending:
                self.trial_log[waiting.number - 1]["eeg"] = "training_aborted"
                self.skipped += 1
            self.pending.clear()
            self.stop_run(f"EEG model training failed: {exc}.")
        else:
            log["training_ms"] = round(elapsed_ms, 2)
            if vector is None:
                self.skipped += 1
                log["eeg"] = "rejected_artifact"
            else:
                self.usable += 1
                self.epochs.append(epoch.astype(np.float32))
                self.epoch_trial_numbers.append(item.number)
                self.epoch_labels.append(dict(item.image.labels))
                self.epoch_values.append(dict(item.image.values))
                self.epoch_groups.append(item.image.group)
                log["eeg"] = "usable"
                if self.running_session == SESSION_TRAINING:
                    if predictions:
                        log["predictions"] = {
                            f"{t}/{c}": {"predicted": self.dataset.class_name(t, p.label),
                                         "p_class1": p.probability, "correct": p.correct}
                            for (t, c), p in predictions.items()}
                        for key in predictions:
                            self.training.add_accuracy(key, item.number, self.trainer.models[key].rolling_accuracy)
                    self.training.update_models(self.trainer)
                elif self.running_session == SESSION_VALIDATION:
                    self._validate_on(vector, item, log)
        self._update_counts()

    def _update_counts(self):
        self.training.count_status.setText(f"Trials: {self.shown} · usable EEG: {self.usable} · skipped: {self.skipped}")

    def stop_run(self, reason: str | None = None):
        if not self.active:
            return
        if self.running_session == SESSION_TEST and self.current_trial is not None:
            self._summarize_test_trial()
        self.active = False
        self.finishing = True
        self.warmup_timer.stop()
        self.fixation_timer.stop()
        self.image_timer.stop()
        self.stop_reason = reason
        self.presentation.stimulus.idle()
        self.features.note.setText("Finishing pending EEG trials. Live plot paused.")
        self.training.stop.setEnabled(False)
        self.stop_shortcut.setEnabled(False)
        self.training.status.setText((reason + " " if reason else "") + "Finishing pending EEG epochs…")
        if not self._training_busy():
            self.finish_run()

    def finish_run(self):
        self.finishing = False
        session = self.running_session
        self.last_raw_time = 0.0
        self.last_quality_time = 0.0
        self.features.oscar_toggle.setEnabled(True)
        self.training.status.setText((self.stop_reason + " " if self.stop_reason else "")
                                     + f"{session.capitalize()} stopped after {self.shown} presentations; "
                                     f"{self.usable} usable EEG epochs.")
        self.training.discard.setEnabled(True)
        self.run_metadata.update({"stopped_at": datetime.now().astimezone().isoformat(),
                                  "presented_trials": self.shown, "skipped_eeg_trials": self.skipped,
                                  "trial_log": self.trial_log})
        if session == SESSION_TRAINING:
            self.training.prepare_save_selection(self.trainer)
            if self.trainer.any_fitted:
                models = {k: t.model for k, t in self.trainer.models.items() if t.model is not None}
                self._set_eval_models(models, {"image_duration_s": self.epoch_spec.image_duration_ms / 1000,
                                               "evaluation_mode": self.run_metadata["evaluation_mode"],
                                               "training_image_files": self.run_metadata["training_image_files"]},
                                      "this training run")
            else:
                self.training.status.setText(self.training.status.text() + " A model needs at least 10 usable EEG trials with both labels.")
        else:
            summary = {f"{t}/{c}": dict(zip(("accuracy", "balanced_accuracy", "n"), self.validation.accuracy((t, c))),
                                         majority_baseline=self.validation.majority_baseline((t, c)))
                       for t, c in sorted(self.validation.models)}
            self.run_metadata["scores"] = summary
            if session == SESSION_TEST:
                distances = [e["mean_distance"] for e in self.trial_log if "mean_distance" in e]
                self.run_metadata["mean_distance"] = float(np.mean(distances)) if distances else None
            self.training.save.setEnabled(True)

    def save_model(self):
        """Save the finished session: selected trained models (training) or the session log (validation/test)."""
        session = self.run_metadata.get("session", SESSION_TRAINING)
        selected = self.training.selected_save_models() if session == SESSION_TRAINING else set()
        if session == SESSION_TRAINING and not selected:
            QMessageBox.warning(self, "Cannot save", "Select at least one fitted model in the Save column.")
            return
        directory = self.root / "saved_models"
        directory.mkdir(exist_ok=True)
        participant = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.run_metadata["participant_id"])
        name = {SESSION_TRAINING: "scene", SESSION_VALIDATION: "validation", SESSION_TEST: "test"}[session]
        stem = directory / (f"{name}_{participant}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        if session == SESSION_TRAINING:
            epochs_path = stem.with_suffix(".epochs.npz")
            self.run_metadata["epochs_file"] = epochs_path.name
            try:
                written = self.trainer.save(stem, self.run_metadata, selected)
            except ValueError as exc:
                QMessageBox.warning(self, "Cannot save", str(exc)); return
            self._save_epochs(epochs_path, self.trainer.targets)
            self.training.status.setText(f"Saved {len(written) + 1} files to saved_models/{stem.name}.*")
        else:
            targets = [t.name for t in self.config.targets]
            if self.epochs:
                epochs_path = stem.with_suffix(".epochs.npz")
                self.run_metadata["epochs_file"] = epochs_path.name
                self._save_epochs(epochs_path, targets)
            if self.decode_log:
                self.run_metadata["live_decoding"] = self.decode_log
            stem.with_suffix(".json").write_text(json.dumps(self.run_metadata, indent=2, default=str), encoding="utf-8")
            self.training.status.setText(f"Saved {session} log to saved_models/{stem.name}.json")
        self.training.reset_save_selection()
        self.training.save.setEnabled(False)
        self.ready_for_new_run = True
        self.update_start_availability()
        self.training.settings_scroll.setVisible(True)
        self.training.discard.setEnabled(False)
        self._enable_settings()

    def _save_epochs(self, path: Path, targets: list[str]):
        if not self.epochs:
            return
        labels = {f"label_{t}": np.asarray([row[t] for row in self.epoch_labels], dtype=np.uint8) for t in targets}
        labels.update({f"value_{t}": np.asarray([row[t] for row in self.epoch_values], dtype=np.float32)
                       for t in targets})
        np.savez_compressed(path, eeg_uV=np.stack(self.epochs),
                            trial_number=np.asarray(self.epoch_trial_numbers, dtype=np.int32),
                            group=np.asarray(self.epoch_groups, dtype=str),
                            channels=np.asarray(self.config.eeg.electrodes, dtype=str), **labels)

    def discard_run(self):
        if self.running_session == SESSION_TRAINING:
            self.trainer = MultiTrainer([], [], len(self.extractor.names))
        self.epochs = []
        self.epoch_trial_numbers = []
        self.epoch_labels = []
        self.epoch_values = []
        self.epoch_groups = []
        self.decode_log = []
        self.training.status.setText("Session data discarded. Ready to start again.")
        self.ready_for_new_run = True
        self.update_start_availability()
        self.training.settings_scroll.setVisible(True)
        self.training.reset_save_selection()
        self.training.save.setEnabled(False)
        self.training.discard.setEnabled(False)
        self._enable_settings()

    def _enable_settings(self):
        for widget in self._settings_widgets:
            widget.setEnabled(True)
        self.features.oscar_toggle.setEnabled(True)
        self.training.count.setEnabled(not self.training.persistent.isChecked())
        self._on_session_changed()

    def closeEvent(self, event):
        self.active = False
        self.clock.stop()
        self.warmup_timer.stop()
        self.fixation_timer.stop()
        self.image_timer.stop()
        self.device.close()
        self.training_executor.shutdown(wait=False, cancel_futures=True)
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
