"""Automatic, real-EEG facial-expression frequency-tagging pilot.

The scene classifier remains available with ``run.py --scenes``. This session
records every real scan and makes no simulated EEG available to the user.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from .acquisition import DeviceManager
from .config import AppConfig, CONFIG
from .face_signal import BASE_HZ, BLOCK_S, tagged_response
from .face_stimuli import LEVELS, prepare_faces
from .features import FeatureExtractor
from .gui import FeaturePanel, PanelWindow
from .quality import SignalQuality
from .recording import SessionRecorder

MAX_SESSION_S = 15 * 60
FACE_COUNT = round(BASE_HZ * BLOCK_S)
QUIZ_FACE_MS = 500


@dataclass
class Block:
    phase: str
    emotion: str
    level: int
    identity: int
    control: bool = False


class FaceScreen(QWidget):
    painted = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(540, 550)
        self.picture = QPixmap()
        self.message = "Waiting for real Unicorn EEG…"
        self.report_paint = False

    def show_face(self, picture: QPixmap, message: str = "", report_paint: bool = False):
        self.picture = picture
        if self.picture.isNull():
            raise ValueError("Could not load generated face")
        self.message = message
        self.report_paint = report_paint
        self.update()

    def show_message(self, message: str):
        self.picture = QPixmap()
        self.message = message
        self.report_paint = False
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#101827"))
        if not self.picture.isNull():
            area = self.rect().adjusted(20, 20, -20, -86)
            image = self.picture.scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            x = area.center().x() - image.width() // 2
            y = area.center().y() - image.height() // 2
            p.drawPixmap(x, y, image)
        p.setPen(QColor("#eaf1ff"))
        p.setFont(QFont("Arial", 18))
        p.drawText(self.rect().adjusted(20, 0, -20, -15),
                   Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
                   self.message)
        p.end()
        if self.report_paint:
            self.report_paint = False
            self.painted.emit(time.perf_counter())


class FacePilot(QMainWindow):
    """Manual-start pilot that requires a current real EEG stream."""

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.config = AppConfig.from_dict(CONFIG, root)
        self.eeg = self.config.eeg
        self.assets = prepare_faces(root / "output" / "face_pilot_assets")
        self.pixmaps = {key: QPixmap(str(path)) for key, path in self.assets.items()}
        if any(picture.isNull() for picture in self.pixmaps.values()):
            raise RuntimeError("A generated face could not be loaded")
        self.extractor = FeatureExtractor(self.eeg)
        self.quality = SignalQuality(self.eeg)
        self.device = DeviceManager(self.eeg, serial="UN-2019.03.01")
        self.device.set_retention(60)
        self.device.configure(False, "UN-2019.03.01")  # wet-electrode pilot: OSCAR off
        self.setWindowTitle("Hot Brain · face EEG pilot")
        self.resize(470, 720)
        content = QWidget()
        layout = QVBoxLayout(content)
        title = QLabel("Facial expression EEG pilot")
        title.setStyleSheet("font-size: 22px; font-weight: bold")
        layout.addWidget(title)
        self.status = QLabel("Connecting to real Unicorn EEG. Press Start when ready.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.device_status = QLabel("Checking device…")
        self.device_status.setWordWrap(True)
        layout.addWidget(self.device_status)
        self.time_label = QLabel("Pilot limit: 15 minutes · press Escape to stop")
        layout.addWidget(self.time_label)
        self.plot = pg.PlotWidget(title="Expression-change EEG response by block")
        self.plot.setLabel("left", "Tagged response / nearby EEG noise")
        self.plot.setLabel("bottom", "Face block")
        self.plot.showGrid(x=True, y=True, alpha=.2)
        self.plot.addLine(y=1.7, pen=pg.mkPen("#e2ac5c", style=Qt.PenStyle.DashLine))
        self.happy_curve = self.plot.plot(pen=pg.mkPen("#6dd7bf", width=2), symbol="o", name="happy")
        self.sad_curve = self.plot.plot(pen=pg.mkPen("#ff727a", width=2), symbol="o", name="sad")
        self.plot.addLegend()
        layout.addWidget(self.plot, 1)
        self.quiz_plot = pg.PlotWidget(title="Five-face quiz: correct (1) or incorrect (0)")
        self.quiz_plot.setLabel("left", "Answer")
        self.quiz_plot.setLabel("bottom", "Face block")
        self.quiz_plot.setYRange(-0.1, 1.1)
        self.quiz_plot.showGrid(x=True, y=True, alpha=.2)
        self.quiz_curve = self.quiz_plot.plot(pen=pg.mkPen("#8bb9ff", width=2), symbol="o")
        layout.addWidget(self.quiz_plot, 1)
        self.summary = QLabel("EEG responses and quiz answers will appear here.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.start_button = QPushButton("Start face pilot")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)
        layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop pilot (Escape)")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: self.finish("Stopped by participant"))
        layout.addWidget(self.stop_button)
        self.setCentralWidget(content)
        self.screen = FaceScreen()
        self.presentation_window = PanelWindow("Hot Brain · participant face screen", self.screen)
        self.features = FeaturePanel(self.extractor)
        self.features.oscar_toggle.setChecked(False)
        self.features.note.setText("Real Unicorn EEG only. The face pilot uses the posterior tagged response; these are diagnostic plots.")
        self.features.reconnect.clicked.connect(self.device.retry_now)
        self.features.oscar_toggle.toggled.connect(self._set_oscar)
        self.features_window = PanelWindow("Hot Brain · live EEG and quality", self.features)
        self.device.status_changed.connect(self.device_status.setText)
        self.screen.painted.connect(self._painted)
        self.clock = QTimer(self)
        self.clock.setInterval(5)
        self.clock.setTimerType(Qt.TimerType.PreciseTimer)
        self.clock.timeout.connect(self._tick)
        self.clock.start()
        self.started = False
        self.active = False
        self.finished = False
        self.run_started = 0.0
        self.started_at = ""
        self.recorder: SessionRecorder | None = None
        self.plan: list[Block] = []
        self.block_index = -1
        self.block: Block | None = None
        self.block_started = 0.0
        self.frame_index = 0
        self.frame_times: list[float] = []
        self.marker_code: int | None = None
        self.marker_delay = 0.0
        self.awaiting_eeg = False
        self.block_metric: dict | None = None
        self.quiz_target: int | None = None
        self.quiz_step = ""
        self.quiz_shown = 0
        self.answer: dict = {}
        self.quiz_timer = QTimer(self)
        self.quiz_timer.setSingleShot(True)
        self.quiz_timer.timeout.connect(self._quiz_timeout)
        self.feedback_timer = QTimer(self)
        self.feedback_timer.setSingleShot(True)
        self.feedback_timer.timeout.connect(self._next_block)
        self.rows: list[dict] = []
        self.level = {"happy": 70, "sad": 70}
        self.control_ratio: float | None = None
        self.posterior_good = False
        self.last_diagnostics = 0.0
        QApplication.instance().installEventFilter(self)

    def show_windows(self):
        bounds = QApplication.primaryScreen().availableGeometry()
        widths = [int(bounds.width() * x) for x in (.26, .40, .34)]
        x = bounds.x()
        for window, width in zip((self, self.presentation_window, self.features_window), widths):
            window.setGeometry(x, bounds.y(), width, bounds.height())
            window.show()
            x += width

    def _set_oscar(self, checked: bool):
        if self.active:
            self.features.oscar_toggle.blockSignals(True)
            self.features.oscar_toggle.setChecked(self.device.oscar)
            self.features.oscar_toggle.blockSignals(False)
            return
        self.device.configure(checked, "UN-2019.03.01")
        self.posterior_good = False

    def _make_plan(self):
        rng = random.Random()
        pre = [Block("pre", emotion, level, 0) for emotion in ("happy", "sad") for level in (40, 80)]
        post = [Block("post", emotion, level, 3) for emotion in ("happy", "sad") for level in (40, 80)]
        rng.shuffle(pre)
        rng.shuffle(post)
        training = [Block("train", emotion, 70, 1 if i % 2 == 0 else 2)
                    for i in range(5) for emotion in ("happy", "sad")]
        rng.shuffle(training)
        return [Block("control", "neutral", 0, 0, True), *pre, *training, *post]

    def _start(self):
        if self.started or not self.device.stream_ready() or not self.posterior_good:
            return
        self.started = self.active = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.run_started = time.perf_counter()
        self.started_at = datetime.now().astimezone().isoformat()
        self.plan = self._make_plan()
        metadata = {"participant_id": "pilot", "session": "face_pilot",
                    "started_at": self.started_at, "fs": self.eeg.fs,
                    "oscar_enabled": self.device.oscar, "protocol": "controlled schematic faces; 6 Hz base, 1.2 Hz expression change; 20 s blocks",
                    "max_session_s": MAX_SESSION_S, "blocks_planned": len(self.plan),
                    "eeg_source": "real Unicorn only", "electrodes": self.eeg.electrodes}
        try:
            self.recorder = SessionRecorder(self.root / "recordings", "pilot", "face_pilot", metadata)
        except OSError as exc:
            self.started = False
            self.active = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status.setText(f"Cannot start recording: {exc}")
            return
        self.device.recorder = self.recorder
        self.features.oscar_toggle.setEnabled(False)
        self.features.reconnect.setEnabled(False)
        self._next_block()

    def _next_block(self):
        if not self.active:
            return
        self.block_index += 1
        if self.block_index >= len(self.plan):
            self.finish("Pilot completed")
            return
        self.block = self.plan[self.block_index]
        if self.block.phase == "train":
            self.block.level = min(LEVELS[1:], key=lambda x: abs(x - self.level[self.block.emotion]))
        self.frame_index = 0
        self.frame_times = []
        self.marker_code = None
        self.marker_delay = 0.0
        self.block_metric = None
        self.awaiting_eeg = False
        self.answer = {}
        self.quiz_target = None
        self.quiz_step = ""
        self.block_started = time.perf_counter()
        self.status.setText(f"{self.block.phase.upper()} · block {self.block_index + 1}/{len(self.plan)} · "
                            f"{self.block.emotion} {self.block.level}%")
        self._show_frame(0)

    def _show_frame(self, index: int):
        block = self.block
        oddball = not block.control and index % 5 == 4
        key = (block.identity, block.emotion, block.level) if oddball else (block.identity, "neutral", 0)
        self.screen.show_face(self.pixmaps[key], "Look at the face · keep still", report_paint=True)

    def _painted(self, stamp: float):
        if not self.active or self.block is None or self.quiz_step:
            return
        self.frame_times.append(stamp)
        if self.marker_code is None:
            try:
                self.block_started = stamp
                self.marker_code, self.marker_delay = self.device.mark_image(stamp)
                self.recorder.event({"type": "face_block", "block": self.block_index,
                                     "phase": self.block.phase, "identity": self.block.identity,
                                     "emotion": self.block.emotion, "level": self.block.level,
                                     "marker": self.marker_code, "paint_time": stamp})
            except Exception as exc:
                self.finish(f"Marker or recording failed: {exc}")

    def _end_block(self):
        if self.marker_code is None:
            self.finish("No block marker was placed")
            return
        self.awaiting_eeg = True
        self.screen.show_message("Preparing the five-face question…")
        if self.block.control:
            self.quiz_step = "control"
        else:
            self.quiz_target = random.choice((None, 0, 1, 2, 3, 4))
            self.quiz_shown = 0
            self.quiz_step = "sequence"
            self._quiz_face()

    def _quiz_face(self):
        if self.quiz_shown >= 5:
            self.quiz_step = "position"
            self.screen.show_message("Which face changed? Press 1–5, or 0 for none.")
            self.quiz_timer.timeout.disconnect()
            self.quiz_timer.timeout.connect(self._quiz_timeout)
            self.quiz_timer.start(6000)
            return
        b = self.block
        key = ((b.identity, b.emotion, b.level) if self.quiz_shown == self.quiz_target
               else (b.identity, "neutral", 0))
        self.screen.show_face(self.pixmaps[key], f"Face {self.quiz_shown + 1} of 5")
        self.quiz_shown += 1
        self.quiz_timer.timeout.disconnect()
        self.quiz_timer.timeout.connect(self._quiz_face)
        self.quiz_timer.start(QUIZ_FACE_MS)

    def _quiz_timeout(self):
        if self.quiz_step == "sequence":
            self._quiz_face()
        elif self.quiz_step in ("position", "direction", "strength"):
            self.answer[self.quiz_step] = None
            self._advance_answer()

    def eventFilter(self, watched: QObject, event: QEvent):
        if event.type() != QEvent.Type.KeyPress:
            return False
        key = event.key()
        if key == Qt.Key.Key_Escape and self.active:
            self.finish("Stopped by participant")
            return True
        if not self.active or self.quiz_step not in ("position", "direction", "strength"):
            return False
        if self.quiz_step in ("position", "strength") and Qt.Key.Key_0 <= key <= Qt.Key.Key_5:
            value = key - Qt.Key.Key_0
            if self.quiz_step == "strength" and value == 0:
                return True
            self.answer[self.quiz_step] = value
        elif self.quiz_step == "direction" and key in (Qt.Key.Key_H, Qt.Key.Key_S):
            self.answer["direction"] = "happy" if key == Qt.Key.Key_H else "sad"
        else:
            return False
        self.quiz_timer.stop()
        self._advance_answer()
        return True

    def _advance_answer(self):
        if self.quiz_step == "position":
            if self.answer.get("position") in (None, 0):
                self._finish_question()
                return
            self.quiz_step = "direction"
            self.screen.show_message("Which direction? Press H for happy or S for sad.")
        elif self.quiz_step == "direction":
            self.quiz_step = "strength"
            self.screen.show_message("How strong? Press 1 (very subtle) to 5 (very strong).")
        else:
            self._finish_question()
            return
        self.quiz_timer.timeout.disconnect()
        self.quiz_timer.timeout.connect(self._quiz_timeout)
        self.quiz_timer.start(6000)

    def _finish_question(self):
        if self.awaiting_eeg:
            self.quiz_step = "waiting_eeg"
            self.screen.show_message("Waiting for the last real EEG samples…")
            return
        self.quiz_step = "feedback"
        self.quiz_timer.stop()
        self._record_block()
        if self.block.phase == "train":
            actual = "none" if self.quiz_target is None else f"{self.quiz_target + 1}: {self.block.emotion} {self.block.level}%"
            self.screen.show_message(f"Answer: {actual}\nEEG tagged response: "
                                     f"{self.block_metric['ratio']:.2f}" if self.block_metric and "ratio" in self.block_metric else
                                     f"Answer: {actual}\nEEG block still being checked")
        else:
            self.screen.show_message("Next block…")
        self.feedback_timer.start(1500)

    def _record_block(self):
        b = self.block
        answer_position = self.answer.get("position")
        expected_position = 0 if self.quiz_target is None else self.quiz_target + 1
        correct_position = answer_position == expected_position
        correct_direction = self.quiz_target is not None and self.answer.get("direction") == b.emotion
        correct = bool(correct_position and (self.quiz_target is None or correct_direction))
        row = {"block": self.block_index, "phase": b.phase, "emotion": b.emotion,
               "level": b.level, "identity": b.identity, "control": b.control,
               "marker_code": self.marker_code, "marker_delay_ms": self.marker_delay,
               "eeg_onset_sample": self.device.ring.eeg_onset_sample(self.marker_code, self.marker_delay),
               "frame_times": list(self.frame_times), "frames_painted": len(self.frame_times),
               "quiz_changed_position": expected_position, "answer": dict(self.answer),
               "correct_position": correct_position, "correct_direction": correct_direction,
               "correct": correct, "eeg": self.block_metric}
        self.rows.append(row)
        if self.recorder:
            try:
                self.recorder.event({"type": "face_block_result", **row})
            except Exception as exc:
                self.finish(f"Recording failed: {exc}")
                return
        if b.phase == "train" and self.quiz_target is not None and self.block_metric and "ratio" in self.block_metric:
            threshold = max(1.7, (self.control_ratio or 1.0) + 0.5)
            if self.block_metric["ratio"] >= threshold and correct:
                self.level[b.emotion] = max(20, b.level - 15)
            else:
                self.level[b.emotion] = min(100, b.level + 15)
        self._plot_results()

    def _plot_results(self):
        for emotion, curve in (("happy", self.happy_curve), ("sad", self.sad_curve)):
            rows = [r for r in self.rows if r["emotion"] == emotion and r["eeg"] and "ratio" in r["eeg"]]
            curve.setData([r["block"] + 1 for r in rows], [r["eeg"]["ratio"] for r in rows])
        quiz_rows = [r for r in self.rows if not r["control"]]
        self.quiz_curve.setData([r["block"] + 1 for r in quiz_rows],
                                [int(r["correct"]) for r in quiz_rows])
        def performance(phase):
            rows = [r for r in self.rows if r["phase"] == phase and r["quiz_changed_position"] != 0]
            return "—" if not rows else f"{sum(r['correct'] for r in rows)}/{len(rows)}"
        self.summary.setText(f"Five-face quiz correct: baseline {performance('pre')} · "
                             f"training {performance('train')} · final {performance('post')}\n"
                             f"Current training strength: happy {self.level['happy']}% · sad {self.level['sad']}%")

    def _collect_eeg(self):
        if not self.awaiting_eeg or self.marker_code is None:
            return
        intervals = np.diff(self.frame_times)
        if (len(self.frame_times) != FACE_COUNT or len(intervals) != FACE_COUNT - 1
                or np.any(abs(intervals - 1 / BASE_HZ) > 0.04)):
            self.block_metric = {"error": "frame_timing_unreliable",
                                 "frames_painted": len(self.frame_times)}
            self.awaiting_eeg = False
            if self.quiz_step == "control":
                self.answer = {}
                self.quiz_target = None
                self._record_block()
                self.quiz_step = "feedback"
                self.feedback_timer.start(500)
            elif self.quiz_step == "waiting_eeg":
                self._finish_question()
            return
        from .features import EpochSpec
        # Existing ring epoch spans -200 to +20,000 ms for this 20.3 s spec.
        epoch, reason = self.device.ring.epoch_result(self.marker_code, EpochSpec(20300, self.eeg.fs), self.marker_delay)
        if epoch is None:
            if reason in ("waiting_marker", "waiting_samples"):
                return
            self.block_metric = {"error": reason}
        else:
            try:
                self.block_metric = tagged_response(epoch[round(.2 * self.eeg.fs):])
            except ValueError as exc:
                self.block_metric = {"error": str(exc)}
        self.awaiting_eeg = False
        if self.quiz_step == "control":
            if self.block_metric and "ratio" in self.block_metric:
                self.control_ratio = self.block_metric["ratio"]
            self.answer = {}
            self.quiz_target = None
            self._record_block()
            self.quiz_step = "feedback"
            self.feedback_timer.start(500)
        elif self.quiz_step == "waiting_eeg":
            self._finish_question()

    def _diagnostics(self, now: float):
        if now - self.last_diagnostics < 1.0:
            return
        self.last_diagnostics = now
        if not self.device.stream_live():
            self.posterior_good = False
            self.features.source_status.setText("NO LIVE EEG · no simulated data")
            self.features.clear()
            return
        self.features.source_status.setText("REAL UNICORN EEG · OSCAR " + ("on" if self.device.oscar else "off"))
        _, recent = self.device.ring.recent(4)
        if self.features.live_raw.isChecked():
            self.features.set_raw(recent)
        if len(recent):
            ratings = self.quality.assess(recent)
            self.features.set_quality(ratings)
            self.posterior_good = sum(ratings[i].good is True for i in (5, 6, 7)) >= 2
        if self.features.live_features.isChecked():
            spec = self.extractor.epoch_spec(1000)
            if len(recent) >= spec.n_samples and np.isfinite(recent[-spec.n_samples:]).all():
                try:
                    self.features.add(now, self.extractor.extract(recent[-spec.n_samples:], spec))
                except ValueError:
                    pass

    def _tick(self):
        now = time.perf_counter()
        if self.active and now - self.run_started >= MAX_SESSION_S:
            self.finish("15-minute limit reached")
            return
        if self.active and not self.device.stream_live():
            self.finish("Real EEG stream lost")
            return
        if self.active and self.recorder and (self.recorder.error or self.recorder.dropped_blocks):
            self.finish("EEG recording writer failed")
            return
        if not self.started and not self.finished:
            connected = self.device.stream_ready()
            ready = connected and self.posterior_good
            self.start_button.setEnabled(ready)
            message = ("Real EEG ready. Press Start face pilot."
                       if ready else
                       "Real EEG connected. Adjust the back electrodes until at least two lights are green."
                       if connected else "Waiting for real Unicorn EEG. Press Start when it is ready.")
            if self.status.text() != message:
                self.status.setText(message)
        if self.active and self.block and not self.quiz_step:
            elapsed = now - self.block_started
            target_frame = min(FACE_COUNT - 1, int(elapsed * BASE_HZ))
            if target_frame > self.frame_index and elapsed < BLOCK_S:
                self.frame_index = target_frame
                self._show_frame(target_frame)
            if elapsed >= BLOCK_S:
                self._end_block()
        if self.active:
            self._collect_eeg()
            self.time_label.setText(f"Elapsed {now - self.run_started:.0f} s / 900 s · Escape stops")
        self._diagnostics(now)

    def finish(self, reason: str):
        if self.finished:
            return
        self.finished = True
        self.active = False
        self.quiz_step = ""
        self.quiz_timer.stop()
        self.feedback_timer.stop()
        self.screen.show_message(f"Pilot ended. {reason}\nYou can remove the headset.")
        self.status.setText(reason)
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.features.oscar_toggle.setEnabled(True)
        self.features.reconnect.setEnabled(True)
        if self.recorder is not None:
            self.device.recorder = None
            phases = {}
            for phase in ("pre", "train", "post"):
                rows = [r for r in self.rows if r["phase"] == phase]
                response = [r["eeg"]["ratio"] for r in rows if r["eeg"] and "ratio" in r["eeg"]]
                phases[phase] = {"blocks": len(rows), "quiz_correct": sum(bool(r["correct"]) for r in rows),
                                 "mean_eeg_ratio": float(np.mean(response)) if response else None,
                                 "valid_eeg_blocks": len(response)}
            metadata = {"participant_id": "pilot", "session": "face_pilot",
                        "started_at": self.started_at,
                        "stopped_at": datetime.now().astimezone().isoformat(),
                        "stop_reason": reason, "fs": self.eeg.fs,
                        "oscar_enabled": self.device.oscar, "protocol": "6 Hz faces / 1.2 Hz changes",
                        "blocks": self.rows, "blocks_completed": len(self.rows),
                        "summary_by_phase": phases,
                        "stimuli": "parametric schematic faces generated by hotbrain.face_stimuli; 4 identities",
                        "control_eeg_ratio": self.control_ratio}
            self.recorder.close(metadata)
            self.status.setText(f"{reason}. Results and EEG saved in {self.recorder.stem.parent}.")

    def closeEvent(self, event):
        if self.active:
            self.finish("Application closed")
        self.clock.stop()
        QApplication.instance().removeEventFilter(self)
        self.device.close()
        self.presentation_window.close()
        self.features_window.close()
        event.accept()


def launch_face_pilot(root: Path) -> int:
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(antialias=True)
    window = FacePilot(root)
    window.show_windows()
    return app.exec()
