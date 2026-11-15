"""Operator, presentation and EEG windows of the taste experiment (sweet vs salty).

Opened from the scene trainer by choosing the Configuration
"Gorzki smak porażki i słodko-słony sukces (?)"; the Unicorn connection is
handed over, so the headset does not reconnect. The procedure and its
rationale are in ``hotbrain.taste`` and docs/smak_procedura.md.
"""
from __future__ import annotations

import copy
import json
import random
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from . import classifiers
from .acquisition import DeviceManager
from .features import EpochSpec, FeatureExtractor
from .gui import (CURVE_COLORS, STYLE, FeaturePanel, PanelWindow, Stimulus, TargetEditor, _hint, _row, _section,
                  _seconds)
from .quality import SignalQuality
from .report import clone_trainers
from .sessions import ValidationScores, load_models
from .taste import (SALTY, SWEET, TASTE_CONFIG, TASTE_EXPERIMENT, TASTE_NAME, TASTE_TARGET, UNSUPPORTED_CLASSIFIERS,
                    UNSUPPORTED_FEATURES, TasteConfig, TasteFeatures, TasteTiming, build_taste_report, decode_trial,
                    learn_trial, taste_sequence, write_taste_report)
from .training import MultiTrainer

SESSION_TRAINING = "taste_training"
SESSION_TEST = "taste_test"
EXPERIMENT_SCENE = "scene"

# Top-level windows opened by switching experiments; Qt does not own them, so keep them alive here.
_OPEN_WINDOWS: list[QMainWindow] = []

TASTE_COLORS = {SALTY: "#8bb9ff", SWEET: "#ff9ed8"}


def taste_config_for(scene_config: dict | None) -> dict:
    """TASTE_CONFIG on the same headset wiring as the scene experiment's config."""
    config = copy.deepcopy(TASTE_CONFIG)
    if scene_config and "eeg" in scene_config:
        for key in ("sampling_rate", "electrodes", "device_channels", "regions", "asymmetry_pairs", "notch_hz"):
            if key in scene_config["eeg"]:
                config["eeg"][key] = copy.deepcopy(scene_config["eeg"][key])
    return config


# ---------------------------------------------------------------------------------------------- participant view
class TasteStimulus(Stimulus):
    """Fixation cross, cue word, tasting dot and rinse text. The cue's first paint is the trial onset."""

    def __init__(self):
        super().__init__()
        self.text = ""

    def cue(self, text: str):
        self.text = text
        self.state = "cue"
        self._emit_on_paint = True
        self.update()

    def tasting(self):
        """Identical for both tastes: a small fixation dot while the tastant is held on the tongue."""
        self.state = "tasting"
        self._emit_on_paint = False
        self.update()

    def message(self, text: str):
        self.text = text
        self.state = "message"
        self._emit_on_paint = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101827"))
        painter.setPen(QColor("#eaf1ff"))
        rect = self.rect()
        if self.state == "fixation":
            painter.setFont(QFont("Arial", 52))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "+")
        elif self.state == "cue":
            font = QFont("Arial", max(20, min(rect.height() // 7, rect.width() // 9)))
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text)
        elif self.state == "tasting":
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QColor("#eaf1ff"))
            painter.drawEllipse(rect.center(), 7, 7)
        elif self.state == "message":
            painter.setFont(QFont("Arial", max(16, min(rect.height() // 14, 30))))
            painter.drawText(rect.adjusted(30, 30, -30, -30),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.text)
        else:
            painter.setFont(QFont("Arial", 22))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Gotowe")
        painter.end()
        if self._emit_on_paint:
            self._emit_on_paint = False
            self.onset.emit(time.perf_counter())


class TasteGauge(QWidget):
    """Decoded P(sweet) between SŁONE (left) and SŁODKIE (right)."""

    def __init__(self, classes: tuple[str, str]):
        super().__init__()
        self.setMinimumSize(220, 120)
        self.classes = classes
        self.value: float | None = None
        self.caption = ""

    def set_value(self, value: float | None, caption: str | None = None):
        self.value = None if value is None else float(np.clip(value, 0, 1))
        if caption is not None:
            self.caption = caption
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#141d2b"))
        margin, height = 24, 26
        bar = QRect(margin, self.height() // 2 - height // 2, self.width() - 2 * margin, height)
        painter.setPen(QColor("#3e5372"))
        painter.setBrush(QColor("#223249"))
        painter.drawRoundedRect(bar, 8, 8)
        painter.setPen(QColor("#5b6b84"))
        painter.drawLine(bar.center().x(), bar.top() - 6, bar.center().x(), bar.bottom() + 6)
        painter.setFont(QFont("Arial", 11))
        painter.setPen(QColor(TASTE_COLORS[SALTY]))
        painter.drawText(QRect(margin, bar.bottom() + 6, 160, 20), Qt.AlignmentFlag.AlignLeft,
                         self.classes[SALTY].upper())
        painter.setPen(QColor(TASTE_COLORS[SWEET]))
        painter.drawText(QRect(bar.right() - 160, bar.bottom() + 6, 160, 20), Qt.AlignmentFlag.AlignRight,
                         self.classes[SWEET].upper())
        if self.value is not None:
            x = bar.left() + self.value * bar.width()
            colour = QColor(TASTE_COLORS[SWEET if self.value >= .5 else SALTY])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colour)
            painter.drawEllipse(round(x) - 11, bar.center().y() - 11, 22, 22)
        painter.setPen(QColor("#c9d6ea"))
        painter.drawText(QRect(8, 4, self.width() - 16, bar.top() - 8),
                         Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.caption)
        painter.end()


class TastePresentation(QWidget):
    def __init__(self, classes: tuple[str, str]):
        super().__init__()
        layout = QVBoxLayout(self)
        heading = QLabel("Taste presentation")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        self.stimulus = TasteStimulus()
        layout.addWidget(self.stimulus, 4)
        self.gauge = TasteGauge(classes)
        self.gauge.setVisible(False)
        layout.addWidget(self.gauge, 1)
        self.note = _hint()
        self.note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.note)


# ---------------------------------------------------------------------------------------------- operator panel
class TastePanel(QWidget):
    def __init__(self, config: TasteConfig):
        super().__init__()
        timing = config.timing
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(8)
        heading = QLabel(TASTE_NAME)
        heading.setObjectName("heading")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(_hint("Sweet vs salty taste from EEG. Fixation → cue word → the participant places the "
                               "tastant on the tongue → identical fixation dot → swallow and rinse. "
                               "Only the EEG well after the cue (default +5…+15 s) is classified."))
        self.settings_widget = QWidget()
        settings = QVBoxLayout(self.settings_widget)
        settings.setContentsMargins(0, 0, 6, 0)
        settings.setSpacing(12)

        # --- Experiment and session ---------------------------------------------------------------
        box, form = _section("Experiment")
        self.experiment = QComboBox()
        self.experiment.addItem(TASTE_NAME, TASTE_EXPERIMENT)
        self.experiment.addItem("Scene emotion trainer (configurations 1–3)", EXPERIMENT_SCENE)
        self.experiment.setToolTip("Switch back to the scene-emotion experiment; the headset stays connected.")
        form.addRow("Configuration", self.experiment)
        self.session = QComboBox()
        self.session.addItem("Training (models learn trial by trial)", SESSION_TRAINING)
        self.session.addItem("Test (frozen models, live P(sweet))", SESSION_TEST)
        form.addRow("Session type", self.session)
        self.participant = QLineEdit("debug")
        form.addRow("Participant ID", self.participant)
        self.load_models = QPushButton("Load saved taste models…")
        self.models_status = _hint("No trained taste models yet.")
        form.addRow("Saved models", _row(self.load_models))
        form.addRow("", self.models_status)
        self.feedback_classifier = QComboBox()
        self.feedback_classifier.setToolTip("Model that drives the live P(sweet) gauge in test sessions.")
        form.addRow("Live gauge uses", self.feedback_classifier)
        self.blind = QCheckBox("Blind trials: the participant sees only “%s”" % config.blind_cue_text)
        self.blind.setToolTip("The operator panel shows which taste to give. Use it to check that the model "
                              "decodes the taste and not the cue word or the expectation.")
        form.addRow("", self.blind)
        settings.addWidget(box)

        # --- Trial timing -------------------------------------------------------------------------
        box, form = _section("Trial timing (seconds from cue onset)")
        self.per_taste = QSpinBox()
        self.per_taste.setRange(2, 200)
        self.per_taste.setValue(config.per_taste)
        self.max_run = QSpinBox()
        self.max_run.setRange(1, 6)
        self.max_run.setValue(config.max_run)
        form.addRow("Trials per taste", _row(self.per_taste, "at most", self.max_run, "equal tastes in a row"))
        self.fixation_min = _seconds(1.0, 30.0, timing.fixation_min_s, .5)
        self.fixation_max = _seconds(1.0, 30.0, timing.fixation_max_s, .5)
        form.addRow("Fixation before cue", _row(self.fixation_min, "to", self.fixation_max))
        self.trial_s = _seconds(5.0, 60.0, timing.trial_s, 1.0)
        form.addRow("Tasting period", self.trial_s)
        self.cue_s = _seconds(.5, 60.0, timing.cue_visible_s, .5)
        form.addRow("Cue word visible", _row(self.cue_s, _hint("then the same dot for both tastes")))
        self.analysis_start = _seconds(1.0, 60.0, timing.analysis_start_s, .5)
        self.analysis_end = _seconds(2.0, 60.0, timing.analysis_end_s, .5)
        form.addRow("Analysed EEG", _row(self.analysis_start, "to", self.analysis_end))
        self.window_s = _seconds(1.0, 8.0, timing.window_s, .5)
        self.step_s = _seconds(.25, 8.0, timing.step_s, .25)
        form.addRow("Windows", _row(self.window_s, "every", self.step_s))
        self.rinse_s = _seconds(0.0, 120.0, timing.rinse_s, 1.0)
        form.addRow("Swallow and rinse", self.rinse_s)
        self.warmup = QCheckBox("Wait before the first trial")
        self.warmup.setChecked(True)
        self.warmup_s = _seconds(2.0, 120.0, config.warmup_s, 1.0)
        form.addRow("Start of run", _row(self.warmup, self.warmup_s))
        self.timing_status = _hint()
        form.addRow("", self.timing_status)
        settings.addWidget(box)

        # --- Features and classifiers -------------------------------------------------------------
        box = QGroupBox("Features and classifiers")
        features_layout = QVBoxLayout(box)
        self.preset = QComboBox()
        for profile in config.profiles:
            self.preset.addItem(f"{profile.number} · {profile.name}", profile.number)
        self.preset.setCurrentIndex(self.preset.findData(config.default_profile))
        self.reset_preset = QPushButton("Reset to preset")
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset"))
        preset_row.addWidget(self.preset, 1)
        preset_row.addWidget(self.reset_preset)
        features_layout.addLayout(preset_row)
        self.editor = TargetEditor(SimpleNamespace(name=TASTE_TARGET), config.eeg, config.classifiers, True)
        # The analysis interval comes from the trial timing; ERP and the continuous-target model do not apply.
        self.editor.layout().itemAt(0).layout().setRowVisible(self.editor.window, False)
        for family in UNSUPPORTED_FEATURES:
            self.editor.feature_boxes[family].setChecked(False)
            self.editor.feature_boxes[family].setVisible(False)
        for key in UNSUPPORTED_CLASSIFIERS:
            if key in self.editor.model_boxes:
                self.editor.model_boxes[key].setChecked(False)
                self.editor.model_boxes[key].setVisible(False)
        self.editor.load(config.profile(config.default_profile).target(TASTE_TARGET))
        features_layout.addWidget(self.editor)
        self.baseline_correction = QCheckBox("Relative to the trial's pre-cue baseline (%g…%g s)"
                                             % (timing.baseline_start_s, timing.baseline_end_s))
        self.baseline_correction.setChecked(config.baseline_correction)
        self.baseline_correction.setToolTip("Subtract the features of the fixation just before the cue from every "
                                            "analysis window (ERD/ERS); removes slow drifts of electrode contact.")
        features_layout.addWidget(self.baseline_correction)
        self.emg_limit = QDoubleSpinBox()
        self.emg_limit.setRange(0, 200)
        self.emg_limit.setDecimals(1)
        self.emg_limit.setSuffix(" µV RMS")
        self.emg_limit.setSpecialValueText("off")
        self.emg_limit.setValue(config.emg_max_rms_uv or 0)
        band = config.emg_band_hz or (30, 45)
        self.emg_limit.setToolTip(f"Reject a window when any model channel's {band[0]:g}–{band[1]:g} Hz RMS exceeds "
                                  "this (jaw, tongue and swallowing muscle activity).")
        features_layout.addWidget(_row(QLabel(f"Muscle (EMG {band[0]:g}–{band[1]:g} Hz) limit"), self.emg_limit))
        self.features_status = _hint()
        features_layout.addWidget(self.features_status)
        settings.addWidget(box)

        # --- Headset ------------------------------------------------------------------------------
        box, form = _section("Headset")
        self.serial = QLineEdit("UN-2019.03.01")
        self.serial.setPlaceholderText("Blank = first available Unicorn")
        form.addRow("Unicorn serial", self.serial)
        self.oscar = QCheckBox("g.Pype OSCAR artifact removal")
        self.oscar.setChecked(True)
        form.addRow("EEG processing", self.oscar)
        settings.addWidget(box)
        settings.addStretch()

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setWidget(self.settings_widget)
        layout.addWidget(self.settings_scroll, 3)

        # --- Run controls -------------------------------------------------------------------------
        buttons = QHBoxLayout()
        self.start = QPushButton("▶  Start session")
        self.start.setObjectName("primary")
        self.stop = QPushButton("■  Stop")
        self.stop.setObjectName("danger")
        self.save = QPushButton("Save results")
        self.discard = QPushButton("Discard run")
        self.bad = QPushButton("✗ Mark trial bad (B)")
        self.bad.setToolTip("Spilled, swallowed early, coughed, talked…: the current trial is not learned or scored.")
        for button in (self.start, self.stop, self.bad, self.save, self.discard):
            button.setEnabled(button is self.start)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.start_hint = QLabel()
        self.start_hint.setObjectName("blocker")
        self.start_hint.setWordWrap(True)
        layout.addWidget(self.start_hint)
        self.operator_cue = QLabel()
        self.operator_cue.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.operator_cue.setWordWrap(True)
        self.operator_cue.setStyleSheet("font-size: 20px; font-weight: 700; padding: 6px;")
        layout.addWidget(self.operator_cue)
        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.device_status = _hint("Checking for Unicorn…")
        layout.addWidget(self.device_status)
        self.count_status = _hint("Trials: 0 · usable: 0 · skipped: 0")
        layout.addWidget(_row(self.count_status, _hint("Escape stops · B marks the current trial bad")))

        # --- Results ------------------------------------------------------------------------------
        self.results = QWidget()
        results = QVBoxLayout(self.results)
        results.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(("Model", "Trials", "Accuracy", "Balanced", "Last P(sweet)", "Last"))
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(150)
        results.addWidget(self.table)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0f1724")
        self.plot.setLabel("left", "Trial-level correct (cumulative)", units="%")
        self.plot.setLabel("bottom", "Trial")
        self.plot.setYRange(0, 100)
        self.plot.showGrid(x=True, y=True, alpha=.2)
        self.plot.addLegend(offset=(-10, 10))
        self.plot.addLine(y=50, pen=pg.mkPen("#5b6b84", width=1, style=Qt.PenStyle.DashLine))
        results.addWidget(self.plot, 1)
        self.results.setVisible(False)
        layout.addWidget(self.results, 2)
        self.curves: dict = {}

    def input_widgets(self) -> list[QWidget]:
        return [self.experiment, self.session, self.participant, self.load_models, self.feedback_classifier,
                self.blind, self.per_taste, self.max_run, self.fixation_min, self.fixation_max, self.trial_s,
                self.cue_s, self.analysis_start, self.analysis_end, self.window_s, self.step_s, self.rinse_s,
                self.warmup, self.warmup_s, self.preset, self.reset_preset, self.baseline_correction,
                self.emg_limit, self.serial, self.oscar, *self.editor.widgets()]

    def timing(self, config: TasteConfig) -> TasteTiming:
        return replace(config.timing, trial_s=self.trial_s.value(), cue_visible_s=self.cue_s.value(),
                       analysis_start_s=self.analysis_start.value(), analysis_end_s=self.analysis_end.value(),
                       rinse_s=self.rinse_s.value(), fixation_min_s=self.fixation_min.value(),
                       fixation_max_s=self.fixation_max.value(), window_s=self.window_s.value(),
                       step_s=self.step_s.value())

    def selected_pairs(self) -> list[tuple[str, str]]:
        return [(TASTE_TARGET, key) for key in self.editor.selected_models() if key not in UNSUPPORTED_CLASSIFIERS]

    def configure_results(self, keys: list[tuple[str, str]]):
        self.table.setRowCount(len(keys))
        self.plot.clear()
        self.plot.addLine(y=50, pen=pg.mkPen("#5b6b84", width=1, style=Qt.PenStyle.DashLine))
        self.curves = {}
        for row, key in enumerate(keys):
            label = classifiers.CLASSIFIERS[key[1]].label if key[1] in classifiers.CLASSIFIERS else key[1]
            self.table.setItem(row, 0, QTableWidgetItem(label))
            for column in range(1, 6):
                self.table.setItem(row, column, QTableWidgetItem("—"))
            pen = pg.mkPen(CURVE_COLORS[row % len(CURVE_COLORS)], width=2)
            self.curves[key] = (self.plot.plot([], [], pen=pen, name=label), [], [])
        self.results.setVisible(True)

    def update_results(self, keys: list[tuple[str, str]], scores: ValidationScores, last: dict, number: int,
                       classes: tuple[str, str]):
        for row, key in enumerate(keys):
            accuracy, balanced, n = scores.accuracy(key)
            p = last.get(key)
            values = (str(n), "—" if accuracy is None else f"{accuracy:.0f}%",
                      "—" if balanced is None else f"{balanced:.0f}%",
                      "learning…" if p is None else f"{p:.2f}",
                      "—" if p is None else classes[int(p >= .5)])
            for column, text in enumerate(values, start=1):
                self.table.item(row, column).setText(text)
            if key in last and accuracy is not None:
                curve, xs, ys = self.curves[key]
                xs.append(number)
                ys.append(accuracy)
                curve.setData(xs, ys)


# ---------------------------------------------------------------------------------------------- main window
class TasteWindow(QMainWindow):
    def __init__(self, root: Path, config: dict | None = None, device=None, back=None):
        super().__init__()
        self.root = root
        self.config = TasteConfig.from_dict(TASTE_CONFIG if config is None else config)
        self.back = back                      # callable(device) -> scene window, or None
        self.handed_over = False
        self.setWindowTitle(f"Hot Brain · {TASTE_NAME}")
        self.resize(720, 900)
        self.panel = TastePanel(self.config)
        self.setCentralWidget(self.panel)
        self.presentation = TastePresentation(self.config.classes)
        self.presentation_window = PanelWindow("Hot Brain · taste presentation", self.presentation, self.stop_run)
        self.display_extractor = FeatureExtractor(self.config.eeg)
        self.display_spec = EpochSpec(1500, self.config.eeg.fs, 1500, 0, None)
        self.features = FeaturePanel(self.display_extractor)
        self.features_window = PanelWindow("Hot Brain · EEG diagnostics and features", self.features)
        self.quality = SignalQuality(self.config.eeg)
        self.profile_number = self.config.default_profile
        self.taste = self._build_taste()
        self.device = None
        self._attach_device(device)
        self.training_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="taste-training")
        self.report_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="taste-report")
        self.future: Future | None = None
        self.future_item: dict | None = None
        self.report_future: Future | None = None
        self.report_stem: str | None = None
        self.queue: deque[tuple[dict, np.ndarray]] = deque()
        self.pending: list[dict] = []
        self.trainer = MultiTrainer([], [], self.taste.n_features())
        self.eval_models: dict = {}
        self.eval_info: dict = {}
        self.scores = ValidationScores({})
        self.prequential: dict[tuple[str, str], list[tuple[int, float]]] = {}
        self.last_p: dict[tuple[str, str], float] = {}
        self.result_keys: list[tuple[str, str]] = []
        self.sequence: list[int] = []
        self.index = 0
        self.current: dict | None = None
        self.next_label: int | None = None
        self.live_baseline = None
        self.live_value: float | None = None
        self.running_session = SESSION_TRAINING
        self.active = self.finishing = False
        self.ready_for_new_run = True
        self.stop_reason: str | None = None
        self.shown = self.usable = self.skipped = 0
        self.trial_log: list[dict] = []
        self.epochs: list[np.ndarray] = []
        self.epoch_numbers: list[int] = []
        self.epoch_labels: list[int] = []
        self.run_metadata: dict = {}
        self.run_start = 0.0
        self.last_raw_time = self.last_quality_time = self.last_feature_time = self.last_decode_time = 0.0
        self.last_raw_samples = 0

        self.clock = QTimer(self)
        self.clock.setInterval(100)
        self.clock.timeout.connect(self.poll)
        self.warmup_timer = self._precise_timer(self.begin_fixation)
        self.fixation_timer = self._precise_timer(self.show_cue)
        self.cue_timer = self._precise_timer(self.hide_cue)
        self.trial_timer = self._precise_timer(self.end_trial)
        self.rinse_timer = self._precise_timer(self.begin_fixation)
        self.stop_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.stop_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.stop_shortcut.setEnabled(False)
        self.stop_shortcut.activated.connect(self.stop_run)
        self.bad_shortcut = QShortcut(QKeySequence("B"), self)
        self.bad_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.bad_shortcut.setEnabled(False)
        self.bad_shortcut.activated.connect(self.mark_bad)

        p = self.panel
        p.start.clicked.connect(self.start_run)
        p.stop.clicked.connect(self.stop_run)
        p.bad.clicked.connect(self.mark_bad)
        p.save.clicked.connect(self.save_results)
        p.discard.clicked.connect(self.discard_run)
        p.load_models.clicked.connect(self.load_saved_models)
        p.session.currentIndexChanged.connect(self._on_session_changed)
        p.experiment.currentIndexChanged.connect(self._on_experiment_changed)
        p.preset.currentIndexChanged.connect(self._on_preset_changed)
        p.reset_preset.clicked.connect(self._on_preset_changed)
        p.editor.changed.connect(self._on_features_changed)
        p.editor.models_changed.connect(self.update_start_availability)
        p.baseline_correction.toggled.connect(self._on_features_changed)
        p.emg_limit.valueChanged.connect(self._on_features_changed)
        for control in (p.per_taste, p.max_run, p.fixation_min, p.fixation_max, p.trial_s, p.cue_s,
                        p.analysis_start, p.analysis_end, p.window_s, p.step_s, p.rinse_s):
            control.valueChanged.connect(self._on_timing_changed)
        p.serial.editingFinished.connect(self.configure_device_from_controls)
        p.oscar.toggled.connect(self.configure_device_from_controls)
        self.features.oscar_toggle.setChecked(p.oscar.isChecked())
        self.features.oscar_toggle.toggled.connect(p.oscar.setChecked)
        p.oscar.toggled.connect(self.features.oscar_toggle.setChecked)
        self.presentation.stimulus.onset.connect(self.on_cue_onset)
        self._describe()
        self._on_session_changed()
        self.clock.start()
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(STYLE)

    # ------------------------------------------------------------------ setup helpers
    def _precise_timer(self, callback) -> QTimer:
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.timeout.connect(callback)
        return timer

    def _attach_device(self, device):
        """Use a running device from the scene window when its wiring matches; otherwise open the Unicorn."""
        eeg = self.config.eeg
        if device is not None and (device.eeg.fs != eeg.fs or tuple(device.eeg.device_channels) != eeg.device_channels):
            device.close()
            device = None
        if device is None:
            device = DeviceManager(eeg, self.panel.serial.text().strip() or None)
        else:
            self.panel.serial.setText(device.serial or "")
            self.panel.oscar.setChecked(device.oscar)
        self.device = device
        device.status_changed.connect(self.panel.device_status.setText)
        device.connected_changed.connect(self.on_connection)
        self.panel.device_status.setText("Unicorn connected (handed over)." if device.connected
                                         else "Checking for Unicorn…")

    def _profile(self):
        preset = self.config.profile(self.profile_number)
        return replace(preset, targets=((TASTE_TARGET, self.panel.editor.spec()),))

    def _build_taste(self) -> TasteFeatures:
        p = self.panel
        emg = p.emg_limit.value() or None
        return TasteFeatures(self.config, self._profile(), p.timing(self.config), p.baseline_correction.isChecked(),
                             emg)

    def _describe(self):
        timing = self.panel.timing(self.config)
        problems = timing.problems()
        trials = 2 * self.panel.per_taste.value()
        minutes = round(trials * timing.mean_trial_s() / 60)
        cue = "the whole trial" if timing.cue_kept else f"{timing.cue_visible_s:g} s, then the dot"
        self.panel.timing_status.setText(
            ("⚠ " + "; ".join(problems) + "." if problems else
             f"Fixation {timing.fixation_min_s:g}–{timing.fixation_max_s:g} s (baseline {timing.baseline_start_s:g}…"
             f"{timing.baseline_end_s:g} s) → cue word {cue} → analysis {timing.analysis_start_s:g}–"
             f"{timing.analysis_end_s:g} s ({timing.n_windows()} windows of {timing.window_s:g} s) → end of tasting at "
             f"{timing.trial_s:g} s → rinse {timing.rinse_s:g} s. {trials} trials ≈ {minutes} min."))
        self.presentation.note.setText(f"{trials} trials · cue → +{timing.analysis_start_s:g} s … "
                                       f"+{timing.analysis_end_s:g} s analysed · ≈ {minutes} min")
        described = self.taste.describe()
        if TASTE_TARGET in self.taste.problems:
            text = f"⚠ Not available: {self.taste.problems[TASTE_TARGET]}."
        elif TASTE_TARGET in described:
            spec = described[TASTE_TARGET]
            text = (f"{spec['n_features']} features · {', '.join(spec['electrodes'])} · {', '.join(spec['bands'])} · "
                    f"{', '.join(spec['features'])}")
        else:
            text = "No features."
        self.panel.editor.summary.setText(text)
        eeg = self.config.eeg
        self.panel.features_status.setText(
            f"Bandpass {eeg.bandpass_hz[0]:g}–{eeg.bandpass_hz[1]:g} Hz · notch {eeg.notch_hz or 'off'} · windows "
            f"rejected above ±{eeg.artifact_max_abs_uv:g} µV or at the EMG limit · trial P(sweet) = mean over its "
            f"windows · at least {self.config.min_windows} clean windows per trial.")

    # ------------------------------------------------------------------ settings changes
    def _on_preset_changed(self):
        if self.active or self.finishing:
            return
        self.profile_number = self.panel.preset.currentData()
        self.panel.editor.load(self.config.profile(self.profile_number).target(TASTE_TARGET))
        self._on_features_changed()

    def _on_features_changed(self):
        if self.active or self.finishing:
            return
        self.taste = self._build_taste()
        if self.eval_models:
            self.eval_models, self.eval_info = {}, {}
            self.panel.feedback_classifier.clear()
            self.panel.models_status.setText("Feature settings changed: loaded models no longer match and were "
                                             "unloaded. Train again or load models saved with these settings.")
        self._describe()
        self.update_start_availability()

    def _on_timing_changed(self):
        if self.active or self.finishing:
            return
        self.taste = self._build_taste()
        self._describe()
        self.update_start_availability()

    def _on_session_changed(self):
        session = self.panel.session.currentData()
        if not (self.active or self.finishing):
            learning = session == SESSION_TRAINING
            self.panel.editor.set_editable(learning)
            for widget in (self.panel.preset, self.panel.reset_preset, self.panel.baseline_correction,
                           self.panel.emg_limit, self.panel.window_s, self.panel.step_s):
                widget.setEnabled(learning)
        self.presentation.gauge.setVisible(session == SESSION_TEST)
        self.update_start_availability()

    def _on_experiment_changed(self):
        if self.panel.experiment.currentData() == TASTE_EXPERIMENT:
            return
        self.panel.experiment.blockSignals(True)
        self.panel.experiment.setCurrentIndex(0)
        self.panel.experiment.blockSignals(False)
        QTimer.singleShot(0, self.switch_to_scene)

    def switch_to_scene(self):
        if self.active or self.finishing or self.back is None:
            return
        if not self.ready_for_new_run and QMessageBox.question(
                self, "Switch experiment", "The last taste run is not saved. Discard it and switch?"
        ) != QMessageBox.StandardButton.Yes:
            return
        device = self.release_device()
        window = self.back(device)
        _hand_over_windows(self, window)

    def release_device(self):
        """Give the running EEG connection to another window (it is then not closed with this one)."""
        device = self.device
        device.status_changed.disconnect(self.panel.device_status.setText)
        device.connected_changed.disconnect(self.on_connection)
        self.handed_over = True
        return device

    # ------------------------------------------------------------------ device
    def on_connection(self, connected: bool):
        if not connected:
            self.features.clear()
            self.last_raw_samples = 0
            if self.active:
                self.stop_run("EEG stream disconnected; session stopped.")
        self.update_start_availability()

    def configure_device_from_controls(self):
        if self.active or self.finishing or self.handed_over:
            return
        oscar, serial = self.panel.oscar.isChecked(), self.panel.serial.text().strip() or None
        if self.device.oscar != oscar or self.device.serial != serial:
            self.features.clear()
            self.last_raw_samples = 0
            self.features.note.setText("Reconnecting EEG with updated device settings…")
        self.device.configure(oscar, serial)
        self.update_start_availability()

    def update_start_availability(self):
        if not hasattr(self, "ready_for_new_run"):
            return
        blocker = self._start_blocker()
        enabled = blocker is None and not self.active and not self.finishing
        self.panel.start.setEnabled(enabled)
        self.panel.start_hint.setText("" if enabled or self.active or self.finishing else f"Start disabled: {blocker}")

    def _start_blocker(self) -> str | None:
        if not self.ready_for_new_run:
            return "save or discard the previous run first."
        if self.device is None or not self.device.stream_ready():
            return "waiting for a live EEG stream from the headset."
        problems = self.panel.timing(self.config).problems()
        if problems:
            return problems[0] + "."
        if TASTE_TARGET not in self.taste.targets:
            return "no usable features: tick electrodes, bands and feature families."
        if self.panel.session.currentData() == SESSION_TRAINING:
            if not self.panel.selected_pairs():
                return "tick at least one classifier."
        elif not self.eval_models:
            return "a test needs taste models: run a training session or load saved taste models."
        return None

    # ------------------------------------------------------------------ models
    def _set_eval_models(self, models: dict, info: dict, source: str):
        self.eval_models = models
        self.eval_info = {**info, "source": source}
        combo = self.panel.feedback_classifier
        combo.clear()
        for _, key in sorted(models):
            combo.addItem(classifiers.CLASSIFIERS[key].label if key in classifiers.CLASSIFIERS else key, key)
        self.panel.models_status.setText(f"{len(models)} taste model(s) from {source}.")
        self.update_start_availability()

    def load_saved_models(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load taste models (manifest .json)",
                                              str(self.root / "saved_models"), "Manifest (*.json)")
        if path:
            self.load_models_from(Path(path))

    def load_models_from(self, path: Path) -> bool:
        try:
            models, manifest = load_models(path)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Load models", str(exc))
            return False
        if manifest.get("experiment") != TASTE_EXPERIMENT:
            QMessageBox.warning(self, "Load models", "This manifest is not from the taste experiment.")
            return False
        saved = tuple((manifest.get("feature_names_by_target") or {}).get(TASTE_TARGET, ()))
        if saved != tuple(self.taste.names.get(TASTE_TARGET, ())):
            QMessageBox.warning(self, "Load models", "These models were trained on other features. Choose the same "
                                f"preset ({manifest.get('configuration')}) and electrodes, bands and families.")
            return False
        self._set_eval_models(models, {"manifest": path.name, "timing": manifest.get("timing"),
                                       "baseline_correction": manifest.get("baseline_correction")}, path.name)
        return True

    # ------------------------------------------------------------------ run
    def start_run(self):
        if self.active or self.finishing or self._start_blocker() is not None:
            self.update_start_availability()
            return
        self.configure_device_from_controls()
        p = self.panel
        session = p.session.currentData()
        timing = p.timing(self.config)
        self.taste = self._build_taste()     # fresh baseline history for this run
        try:
            self.sequence = taste_sequence(p.per_taste.value(), p.max_run.value())
        except ValueError as exc:
            QMessageBox.warning(self, "Trial order", str(exc))
            return
        self.running_session = session
        if session == SESSION_TRAINING:
            pairs = p.selected_pairs()
            self.trainer = MultiTrainer([TASTE_TARGET], [k for _, k in pairs], self.taste.n_features(), pairs)
            self.result_keys = list(self.trainer.models)
        else:
            self.result_keys = sorted(self.eval_models)
        self.scores = ValidationScores({})
        self.prequential = {key: [] for key in self.result_keys}
        self.last_p = {}
        p.configure_results(self.result_keys)
        self.device.set_retention(timing.trial_s - timing.baseline_start_s + timing.rinse_s + 15)
        self.index = 0
        self.current = None
        self.pending.clear()
        self.queue.clear()
        self.future = self.future_item = None
        self.shown = self.usable = self.skipped = 0
        self.trial_log, self.epochs, self.epoch_numbers, self.epoch_labels = [], [], [], []
        self.stop_reason = None
        self.live_baseline, self.live_value = None, None
        self.presentation.gauge.set_value(None, "Waiting for the first trial")
        self.active, self.finishing, self.ready_for_new_run = True, False, False
        for widget in p.input_widgets():
            widget.setEnabled(False)
        self.features.oscar_toggle.setEnabled(False)
        p.settings_scroll.setVisible(False)
        p.start.setEnabled(False)
        p.stop.setEnabled(True)
        p.bad.setEnabled(True)
        p.save.setEnabled(False)
        p.discard.setEnabled(False)
        self.stop_shortcut.setEnabled(True)
        self.bad_shortcut.setEnabled(True)
        self.features.clear()
        self.features.note.setText("Session in progress. Live EEG plot paused to protect cue timing.")
        if not self.presentation_window.isVisible():
            self.presentation_window.show()
        self.run_start = time.perf_counter()
        eeg = self.config.eeg
        self.run_metadata = {
            "experiment": TASTE_EXPERIMENT, "experiment_name": TASTE_NAME,
            "session": "taste_training" if session == SESSION_TRAINING else "taste_test",
            "started_at": datetime.now().astimezone().isoformat(),
            "participant_id": p.participant.text().strip() or "unnamed",
            "unicorn_serial": p.serial.text().strip() or None, "oscar_enabled": p.oscar.isChecked(),
            "classes": list(self.config.classes), "label_1": self.config.classes[SWEET],
            "blind": p.blind.isChecked(),
            "cue_text": list(self.config.cue_text), "blind_cue_text": self.config.blind_cue_text,
            "sequence": [self.config.classes[label] for label in self.sequence],
            "timing": {"trial_s": timing.trial_s, "cue_visible_s": timing.cue_visible_s,
                       "analysis_s": [timing.analysis_start_s, timing.analysis_end_s],
                       "baseline_s": [timing.baseline_start_s, timing.baseline_end_s],
                       "fixation_s": [timing.fixation_min_s, timing.fixation_max_s], "rinse_s": timing.rinse_s,
                       "window_s": timing.window_s, "step_s": timing.step_s,
                       "windows_per_trial": timing.n_windows()},
            "baseline_correction": self.taste.baseline_correction,
            "emg": {"band_hz": self.config.emg_band_hz, "max_rms_uv": self.taste.emg_max_rms_uv},
            "configuration": self.profile_number, "configuration_name": self.config.profile(self.profile_number).name,
            "target_features": self.taste.describe(),
            "feature_names_by_target": {t: list(n) for t, n in self.taste.names.items()},
            "model_kind": "taste_state", "state_window_s": timing.window_s,
            "classifiers": [k for _, k in self.result_keys],
            "models_source": None if session == SESSION_TRAINING else self.eval_info.get("source"),
            "fs": eeg.fs, "channels": eeg.electrodes, "device_channels": eeg.device_channels,
            "psd_method": eeg.psd_method, "artifact_max_abs_uv": eeg.artifact_max_abs_uv,
            "artifact_reasons": {}, "baseline_sources": {}, "window_counts": {"total": 0, "usable": 0},
            "marked_bad_trials": 0, "config": self.config.raw,
            "timing_note": "Cue onsets are g.Pype markers on the Unicorn sample grid (+40 ms device delay); the "
                           "tastant contact time is not measured, so only sustained activity well after the cue "
                           "is analysed."}
        if p.oscar.isChecked() and p.warmup.isChecked():
            self.presentation.stimulus.fixation()
            p.status.setText(f"Settling: {p.warmup_s.value():g} s before the first trial.")
            self._operator_prepare(self.sequence[0])
            self.warmup_timer.start(round(p.warmup_s.value() * 1000))
        else:
            self.begin_fixation()

    def _operator_prepare(self, label: int):
        name = self.config.classes[label].upper()
        self.panel.operator_cue.setStyleSheet(f"font-size: 20px; font-weight: 700; padding: 6px; "
                                              f"color: {TASTE_COLORS[label]};")
        self.panel.operator_cue.setText(f"Next: {name}" + (" (give it when the cue appears)"
                                                           if self.panel.blind.isChecked() else ""))

    def ensure_stream(self) -> bool:
        if self.device.stream_live():
            return True
        if self.active:
            self.stop_run("EEG stream stopped; session stopped.")
        return False

    def begin_fixation(self):
        if not self.active or not self.ensure_stream():
            return
        if self.index >= len(self.sequence):
            self.stop_run()
            return
        timing = self.panel.timing(self.config)
        self.next_label = self.sequence[self.index]
        self.presentation.stimulus.fixation()
        self._operator_prepare(self.next_label)
        self.panel.status.setText(f"Fixation before trial {self.index + 1}/{len(self.sequence)}")
        delay = random.uniform(timing.fixation_min_s, timing.fixation_max_s)
        self.fixation_timer.start(round(delay * 1000))

    def _cue_text(self, label: int) -> str:
        return self.config.blind_cue_text if self.panel.blind.isChecked() else self.config.cue_text[label]

    def show_cue(self):
        if not self.active or not self.ensure_stream() or self.next_label is None:
            return
        self.presentation.stimulus.cue(self._cue_text(self.next_label))

    def on_cue_onset(self, stamp: float):
        if not self.active or self.next_label is None or not self.ensure_stream():
            return
        try:
            marker_code, delay_ms = self.device.mark_image(stamp)
        except Exception as exc:
            self.stop_run(f"Could not place the cue marker: {exc}.")
            return
        timing = self.panel.timing(self.config)
        label = self.next_label
        self.next_label = None
        self.index += 1
        self.shown += 1
        trial = {"number": self.shown, "label": label, "marker": marker_code, "delay_ms": delay_ms,
                 "onset": stamp, "ended": False, "ready_at": None, "bad": False,
                 "deadline": stamp + timing.trial_s + 15.0}
        self.pending.append(trial)
        self.current = trial
        self.trial_log.append({"number": self.shown, "taste": self.config.classes[label],
                               "cue": self._cue_text(label),
                               "onset_s_from_run_start": round(stamp - self.run_start, 6),
                               "marker_code": marker_code, "paint_to_marker_ms": round(delay_ms, 3),
                               "eeg": "pending"})
        name = self.config.classes[label].upper()
        self.panel.operator_cue.setText(f"NOW: {name} on the tongue · hold, don't swallow")
        self.panel.status.setText(f"Trial {self.shown}/{len(self.sequence)}: {name}")
        if not timing.cue_kept:
            self.cue_timer.start(round(timing.cue_visible_s * 1000))
        self.trial_timer.start(round(timing.trial_s * 1000))
        self._update_counts()
        if self.running_session == SESSION_TEST:
            self.live_baseline = self._live_baseline(timing)
            self.live_value = None
            self.presentation.gauge.set_value(None, f"Trial {self.shown}: decoding starts at "
                                                    f"+{timing.analysis_start_s:g} s")

    def hide_cue(self):
        if self.active:
            self.presentation.stimulus.tasting()

    def end_trial(self):
        if not self.active or self.current is None:
            return
        self.current["ended"] = True
        self.current["ready_at"] = time.perf_counter() + 2.0   # grace period to mark the trial bad
        self.current = None
        self.cue_timer.stop()
        self.presentation.stimulus.message(self.config.rinse_text)
        self.panel.operator_cue.setText("Swallow and rinse with water")
        timing = self.panel.timing(self.config)
        if timing.rinse_s > 0:
            self.rinse_timer.start(round(timing.rinse_s * 1000))
        else:
            self.begin_fixation()

    def mark_bad(self):
        """Exclude the running trial (or the one that just ended) from learning and scoring."""
        candidates = [t for t in self.pending if not t["bad"]]
        if not (self.active or self.finishing) or not candidates:
            return
        trial = candidates[-1]
        trial["bad"] = True
        self.trial_log[trial["number"] - 1]["marked_bad"] = True
        self.run_metadata["marked_bad_trials"] = self.run_metadata.get("marked_bad_trials", 0) + 1
        self.panel.status.setText(f"Trial {trial['number']} marked bad: it will not be learned or scored.")

    # ------------------------------------------------------------------ live decoding (test)
    def _live_baseline(self, timing: TasteTiming):
        if not self.taste.baseline_correction:
            return None
        fs = self.config.eeg.fs
        _, samples = self.device.ring.recent(-timing.baseline_start_s)
        cut = round(-timing.baseline_end_s * fs)
        segment = samples[:len(samples) - cut] if cut else samples
        windows = self.taste._cut(segment, 0, 0, len(segment) / fs, min(timing.step_s, .5))
        return self.taste.baseline(windows) or self.taste.fallback_baseline()

    def _decode_live(self, now: float):
        trial = self.current
        if trial is None:
            return
        timing = self.panel.timing(self.config)
        since = now - trial["onset"]
        if since < timing.analysis_start_s or since > timing.analysis_end_s:
            return
        key = (TASTE_TARGET, self.panel.feedback_classifier.currentData())
        model = self.eval_models.get(key)
        if model is None:
            return
        if self.taste.baseline_correction and self.live_baseline is None:
            self.presentation.gauge.set_value(None, "No clean pre-cue baseline: live estimate paused")
            return
        n = round(timing.window_s * self.config.eeg.fs)
        _, samples = self.device.ring.recent(timing.window_s + .5)
        if len(samples) < n:
            return
        try:
            vector = self.taste.window_vector(samples[-n:])[TASTE_TARGET]
        except ValueError as exc:
            self.presentation.gauge.set_value(self.live_value, f"Artifact ({str(exc).split(' ')[0]}): estimate paused")
            return
        if self.live_baseline is not None:
            vector = vector - self.live_baseline[TASTE_TARGET]
        p = float(model.predict_proba(vector[None, :])[0, 1])
        w = self.config.feedback_smoothing
        self.live_value = p if self.live_value is None else w * p + (1 - w) * self.live_value
        self.presentation.gauge.set_value(self.live_value, f"Trial {trial['number']} · live P(sweet) "
                                                           f"{self.live_value:.2f}")

    # ------------------------------------------------------------------ poll
    def poll(self):
        now = time.perf_counter()
        self._collect_result()
        self._collect_report()
        live = self.device.stream_live()
        if self.active and not live:
            self.stop_run("EEG stream stopped or became unreliable; session stopped.")
        self.update_start_availability()
        if live:
            text = (f"Unicorn {self.device.serial or '(first available)'} streaming EEG · OSCAR "
                    + ("on" if self.device.oscar else "off"))
            if self.panel.device_status.text() != text:
                self.panel.device_status.setText(text)
        if not live:
            if self.features.samples or self.features.raw_sample_count:
                self.features.clear()
            self.last_raw_samples = 0
            self.features.note.setText("No live EEG stream. Diagnostics and feature values have been cleared.")
        elif self.active:
            self.features.note.setText("Session in progress. Live EEG plot paused to protect cue timing.")
        elif not self.finishing:
            self.features.note.setText("Live EEG diagnostic. Check that Fz, C3, Cz, C4 and Pz are good: they carry "
                                       "the taste features. Ask the participant to clench the jaw to see EMG.")
            if now - self.last_raw_time >= .25:
                self.last_raw_time = now
                count = self.device.ring.received_samples
                if count != self.last_raw_samples:
                    self.features.set_raw(self.device.ring.recent(4)[1])
                    self.last_raw_samples = count
        if live and not self.finishing and now - self.last_quality_time >= .5:
            self.last_quality_time = now
            self.features.set_quality(self.quality.assess(self.device.ring.recent(3)[1]))
        if self.active and live and now - self.last_feature_time >= .25:
            spec = self.display_spec
            times, samples = self.device.ring.recent(max(2, spec.n_samples / spec.fs + 1))
            if len(samples) >= spec.n_samples:
                try:
                    self.features.add(times[-1], self.display_extractor.extract(samples[-spec.n_samples:], spec))
                    self.last_feature_time = now
                except ValueError:
                    pass
        if (self.active and live and self.running_session == SESSION_TEST
                and now - self.last_decode_time >= self.config.feedback_update_s):
            self.last_decode_time = now
            self._decode_live(now)
        self._collect_epochs(now, live)
        self._start_next()
        if self.finishing and not self._busy():
            self.finish_run()

    def _skip(self, trial: dict, reason: str):
        self.skipped += 1
        self.trial_log[trial["number"] - 1]["eeg"] = reason
        self.pending.remove(trial)
        self._update_counts()

    def _collect_epochs(self, now: float, live: bool):
        spec = self.taste.epoch_spec()
        for trial in self.pending[:]:
            if trial["ready_at"] is None or now < trial["ready_at"]:
                break     # keep presentation order (test-before-train)
            log = self.trial_log[trial["number"] - 1]
            if trial["bad"]:
                self._skip(trial, "marked_bad")
                continue
            if trial.get("stopped_early"):
                self._skip(trial, "stopped_before_analysis_end")
                continue
            if not live and self.finishing:
                self._skip(trial, "timing_unreliable")
                continue
            epoch, status = self.device.ring.epoch_result(trial["marker"], spec, trial["delay_ms"])
            log["marker_sample_index"] = self.device.ring.marker_sample(trial["marker"])
            if epoch is not None:
                log["eeg"] = "processing"
                self.queue.append((trial, epoch))
                self.pending.remove(trial)
            elif status in ("invalid_scans", "expired_epoch", "incomplete_epoch") or now >= trial["deadline"]:
                self._skip(trial, status if status != "waiting_samples" else "missing_samples")
            else:
                break

    def _busy(self) -> bool:
        return bool(self.pending or self.queue or self.future)

    def _start_next(self):
        if self.future is not None or not self.queue:
            return
        trial, epoch = self.queue.popleft()
        self.future_item = {"trial": trial, "epoch": epoch}
        if self.running_session == SESSION_TRAINING:
            self.future = self.training_executor.submit(learn_trial, self.taste, self.trainer, epoch, trial["label"],
                                                        trial["number"])
        else:
            self.future = self.training_executor.submit(decode_trial, self.taste, dict(self.eval_models), epoch,
                                                        trial["label"], trial["number"])

    def _collect_result(self):
        future = self.future
        if future is None or not future.done():
            return
        item, self.future, self.future_item = self.future_item, None, None
        trial, epoch = item["trial"], item["epoch"]
        log = self.trial_log[trial["number"] - 1]
        try:
            outcome = future.result()
        except Exception as exc:
            self.skipped += 1
            log["eeg"] = "processing_error"
            log["error"] = f"{type(exc).__name__}: {exc}"
            self.queue.clear()
            self.stop_run(f"Taste model update failed: {exc}.")
            return
        features = outcome.features
        log.update({"processing_ms": round(outcome.elapsed_ms, 2), "baseline": features.baseline_source,
                    "windows": {"total": features.n_windows, "usable": len(features.rows),
                                "rejected": features.reasons}})
        counts = self.run_metadata["window_counts"]
        counts["total"] += features.n_windows
        counts["usable"] += len(features.rows)
        sources = self.run_metadata["baseline_sources"]
        sources[features.baseline_source] = sources.get(features.baseline_source, 0) + 1
        for reason, n in features.reasons.items():
            self.run_metadata["artifact_reasons"][reason] = self.run_metadata["artifact_reasons"].get(reason, 0) + n
        if features.rejected:
            self.skipped += 1
            log["eeg"] = "rejected"
            log["rejection_reason"] = features.rejected
            self.panel.status.setText(f"Trial {trial['number']} rejected: {features.rejected}.")
            self._update_counts()
            return
        self.usable += 1
        log["eeg"] = "usable"
        self.epochs.append(epoch.astype(np.float32))
        self.epoch_numbers.append(trial["number"])
        self.epoch_labels.append(trial["label"])
        label = trial["label"]
        log["predictions"] = {}
        for key, p in outcome.probabilities.items():
            self.scores.records.setdefault(key, []).append((label, int(p >= .5), p))
            self.prequential.setdefault(key, []).append((label, p))
            self.last_p[key] = p
            log["predictions"][f"{key[0]}/{key[1]}"] = {"p_sweet": round(p, 4),
                                                        "predicted": self.config.classes[int(p >= .5)],
                                                        "correct": int(p >= .5) == label}
        self.panel.update_results(self.result_keys, self.scores, dict(outcome.probabilities), trial["number"],
                                  self.config.classes)
        if self.running_session == SESSION_TEST:
            key = (TASTE_TARGET, self.panel.feedback_classifier.currentData())
            if key in outcome.probabilities:
                p = outcome.probabilities[key]
                decided = self.config.classes[int(p >= .5)].upper()
                verdict = "✓" if int(p >= .5) == label else "✗"
                self.presentation.gauge.set_value(p, f"Trial {trial['number']}: decoded {decided} "
                                                     f"(P(sweet) {p:.2f}) {verdict}")
        self._update_counts()

    def _update_counts(self):
        self.panel.count_status.setText(f"Trials: {self.shown} · usable: {self.usable} · skipped: {self.skipped}")

    # ------------------------------------------------------------------ stop / finish / save
    def stop_run(self, reason: str | None = None):
        if not self.active:
            return
        now = time.perf_counter()
        timing = self.panel.timing(self.config)
        if self.current is not None:
            self.current["ended"] = True
            self.current["ready_at"] = now
            # Data after the stop are not a held taste.
            if now - self.current["onset"] < timing.analysis_end_s:
                self.current["stopped_early"] = True
            self.current = None
        self.active, self.finishing = False, True
        for timer in (self.warmup_timer, self.fixation_timer, self.cue_timer, self.trial_timer, self.rinse_timer):
            timer.stop()
        self.stop_reason = reason
        self.presentation.stimulus.idle()
        self.panel.operator_cue.setText("")
        self.panel.stop.setEnabled(False)
        self.stop_shortcut.setEnabled(False)
        self.panel.status.setText((reason + " " if reason else "") + "Finishing pending trials…")
        if not self._busy():
            self.finish_run()

    def finish_run(self):
        self.finishing = False
        self.bad_shortcut.setEnabled(False)
        p = self.panel
        p.bad.setEnabled(False)
        p.discard.setEnabled(True)
        self.features.oscar_toggle.setEnabled(True)
        session = self.running_session
        self.run_metadata.update({"stopped_at": datetime.now().astimezone().isoformat(),
                                  "presented_trials": self.shown, "usable_trials": self.usable,
                                  "rejected_trials": self.skipped, "trial_log": self.trial_log,
                                  "stop_reason": self.stop_reason})
        scores = {}
        for key in self.result_keys:
            accuracy, balanced, n = self.scores.accuracy(key)
            scores[f"{key[0]}/{key[1]}"] = {"accuracy": accuracy, "balanced_accuracy": balanced, "n_trials": n,
                                           "majority_baseline": self.scores.majority_baseline(key),
                                           "permutation_p": self.scores.permutation_p(key)}
        self.run_metadata["trial_scores"] = scores
        text = ((self.stop_reason + " " if self.stop_reason else "")
                + f"Stopped after {self.shown} trials; {self.usable} usable.")
        if session == SESSION_TRAINING:
            if self.trainer.any_fitted:
                models = {k: t.model for k, t in self.trainer.models.items() if t.model is not None}
                self._set_eval_models(models, {"timing": self.run_metadata["timing"]}, "this training run")
                p.save.setEnabled(True)
                p.session.setCurrentIndex(p.session.findData(SESSION_TEST))
                text += " Models ready for a Test session."
            else:
                text += " Not enough clean windows of both tastes to fit a model."
            self._start_report()
        else:
            p.save.setEnabled(bool(self.trial_log))
        p.status.setText(text)

    def _start_report(self):
        if not any(t.x for t in self.trainer.models.values()):
            return
        participant = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.run_metadata["participant_id"])
        self.report_stem = f"taste_{participant}_cfg{self.profile_number}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        meta = {k: v for k, v in self.run_metadata.items() if k != "config"}
        self.report_future = self.report_executor.submit(build_taste_report, clone_trainers(self.trainer.models),
                                                         meta, dict(self.config.evaluation),
                                                         {k: list(v) for k, v in self.prequential.items()})
        self.panel.status.setText(self.panel.status.text() + " Computing the trial-level report…")

    def _collect_report(self):
        future = self.report_future
        if future is None or not future.done():
            return
        self.report_future = None
        try:
            md_path, _ = write_taste_report(future.result(), self.root / "training_reports", self.report_stem)
        except Exception as exc:
            self.panel.status.setText(f"Taste report failed: {type(exc).__name__}: {exc}")
            return
        self.run_metadata["report_files"] = [md_path.name, md_path.with_suffix(".json").name]
        self.panel.status.setText(self.panel.status.text().split(" Computing")[0]
                                  + f" Report: training_reports/{md_path.name}")

    def save_results(self) -> list[Path]:
        directory = self.root / "saved_models"
        directory.mkdir(exist_ok=True)
        participant = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.run_metadata["participant_id"])
        name = "taste" if self.running_session == SESSION_TRAINING else "taste_test"
        stem = directory / (f"{name}_{participant}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        written: list[Path] = []
        if self.epochs:
            epochs_path = stem.with_suffix(".epochs.npz")
            self.run_metadata["epochs_file"] = epochs_path.name
            np.savez_compressed(epochs_path, eeg_uV=np.stack(self.epochs),
                                trial_number=np.asarray(self.epoch_numbers, dtype=np.int32),
                                label_taste=np.asarray(self.epoch_labels, dtype=np.int8),
                                classes=np.asarray(self.config.classes, dtype=str),
                                channels=np.asarray(self.config.eeg.electrodes, dtype=str),
                                epoch_start_s=self.panel.timing(self.config).baseline_start_s)
            written.append(epochs_path)
        if self.running_session == SESSION_TRAINING:
            try:
                written += self.trainer.save(stem, self.run_metadata)
            except ValueError as exc:
                QMessageBox.warning(self, "Cannot save", str(exc))
                return []
        else:
            path = stem.with_suffix(".json")
            path.write_text(json.dumps(self.run_metadata, indent=2, default=str), encoding="utf-8")
            written.append(path)
        self.panel.status.setText(f"Saved {len(written)} file(s) to saved_models/{stem.name}.*")
        self._ready_again()
        return written

    def discard_run(self):
        if self.running_session == SESSION_TRAINING:
            self.trainer = MultiTrainer([], [], self.taste.n_features())
        self.epochs, self.epoch_numbers, self.epoch_labels = [], [], []
        self.panel.status.setText("Run discarded. Ready to start again.")
        self._ready_again()

    def _ready_again(self):
        self.ready_for_new_run = True
        p = self.panel
        p.save.setEnabled(False)
        p.discard.setEnabled(False)
        p.settings_scroll.setVisible(True)
        for widget in p.input_widgets():
            widget.setEnabled(True)
        self._on_session_changed()
        self.update_start_availability()

    # ------------------------------------------------------------------ windows
    def show_windows(self, available: QRect | None = None):
        if available is None:
            screen = QApplication.primaryScreen()
            available = screen.availableGeometry() if screen is not None else QRect(0, 0, 1600, 900)
        area = available.adjusted(8, 8, -8, -8)
        gap = 10
        width = area.width() - 2 * gap
        left, middle = round(width * .40), round(width * .32)
        self.setGeometry(area.x(), area.y(), left, area.height())
        self.presentation_window.setGeometry(area.x() + left + gap, area.y(), middle, area.height())
        self.features_window.setGeometry(area.x() + left + middle + 2 * gap, area.y(), width - left - middle,
                                         area.height())
        self.show()
        self.presentation_window.show()
        self.features_window.show()

    def closeEvent(self, event):
        self.active = False
        self.clock.stop()
        for timer in (self.warmup_timer, self.fixation_timer, self.cue_timer, self.trial_timer, self.rinse_timer):
            timer.stop()
        if not self.handed_over:
            self.device.close()
        self.training_executor.shutdown(wait=False, cancel_futures=True)
        self.report_executor.shutdown(wait=False, cancel_futures=True)
        self.presentation_window.close()
        self.features_window.close()
        if self in _OPEN_WINDOWS:
            _OPEN_WINDOWS.remove(self)
        event.accept()


# ---------------------------------------------------------------------------------------------- switching
def _hand_over_windows(old, new):
    """Show ``new`` where ``old``'s three windows were, then close ``old``."""
    for attr in ("", "presentation_window", "features_window"):
        source = getattr(old, attr) if attr else old
        target = getattr(new, attr) if attr else new
        target.setGeometry(source.geometry())
        target.show()
    _OPEN_WINDOWS.append(new)
    old.close()


def open_taste_experiment(scene_window) -> TasteWindow:
    """Replace the scene-emotion windows with the taste experiment, keeping the EEG connection."""
    from .gui import MainWindow

    root, scene_config = scene_window.root, scene_window.config.raw
    device = scene_window.release_device()

    def back(device_):
        return MainWindow(root, scene_config, device=device_)

    window = TasteWindow(root, taste_config_for(scene_config), device=device, back=back)
    _hand_over_windows(scene_window, window)
    return window
