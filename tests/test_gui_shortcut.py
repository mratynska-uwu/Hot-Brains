"""The three startup windows remain independent and share the stop shortcut."""

import copy
import os
import time
from pathlib import Path

import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QRect, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import hotbrain.gui as gui
from hotbrain.acquisition import SampleRing
from hotbrain.config import CONFIG
from hotbrain.training import MultiTrainer


class NoDevice(QObject):
    status_changed = Signal(str)
    connected_changed = Signal(bool)

    def __init__(self, eeg, serial):
        super().__init__()
        self.eeg = eeg
        self.connected = False
        self.oscar = True
        self.serial = serial
        self.ring = SampleRing(fs=eeg.fs, channels=eeg.device_channels)

    def provide_test_frames(self):
        # Isolated unit-test fixture; the application never synthesizes EEG.
        samples = 500
        t = np.arange(samples) / self.eeg.fs
        frame = np.column_stack([20 * np.sin(2 * np.pi * (8 + i) * t) for i in range(8)])
        self.ring.append(frame, 0)
        was_connected = self.connected
        self.connected = True
        if not was_connected:
            self.connected_changed.emit(True)

    def stream_ready(self, max_age_s=1.0):
        return (self.connected and self.ring.received_samples >= self.eeg.fs // 2
                and time.perf_counter() - self.ring.last_arrival <= max_age_s)

    def configure(self, oscar, serial):
        self.oscar = oscar
        self.serial = serial

    def set_retention(self, seconds):
        self.retention_s = seconds

    def close(self):
        pass


def test_chosen_key_stops_from_presentation_window(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gui, "DeviceManager", NoDevice)
    for folder_name, positive, aroused in (("set/happy", 1, (1.0, 3.0)), ("set/sad", 0, (2.0, 4.0))):
        folder = tmp_path / "data" / folder_name
        folder.mkdir(parents=True)
        lines = ["image_name,valence,arousal"]
        for i, value in enumerate(aroused):
            assert QImage(12, 12, QImage.Format.Format_RGB32).save(str(folder / f"sample{i}.png"))
            lines.append(f"sample{i}.png,{3 if positive else -3},{value}")
        folder.with_suffix(".csv").write_text("\n".join(lines), encoding="utf-8")
    config = copy.deepcopy(CONFIG)
    config["datasets"] = ["set/happy", "set/sad"]

    app = QApplication.instance() or QApplication([])
    window = gui.MainWindow(tmp_path, config)
    window.show_windows(QRect(0, 0, 1800, 1000))
    window.training.warmup.setChecked(False)
    try:
        app.processEvents()
        assert window.isVisible()
        assert window.presentation_window.isVisible()
        assert window.features_window.isVisible()
        assert window.geometry().right() < window.presentation_window.geometry().left()
        assert window.presentation_window.geometry().right() < window.features_window.geometry().left()
        assert window.training.parentWidget() is window
        assert window.presentation.parentWidget() is window.presentation_window
        assert window.features.parentWidget() is window.features_window
        window.show_windows(QRect(0, 0, 1440, 900))
        app.processEvents()
        assert window.geometry().right() < window.presentation_window.geometry().left()
        assert window.presentation_window.geometry().right() < window.features_window.geometry().left()
        assert window.dataset is not None, window.training.dataset_status.text()
        assert not window.training.start.isEnabled()
        window.start_run()
        assert not window.active
        assert window.shown == 0
        assert window.presentation.stimulus.state == "idle"
        assert not window.features.samples
        assert window.training.image_seconds.value() == 4.5
        assert window.training.fixation_min.value() == 2.0
        assert window.training.fixation_max.value() == 3.0
        assert "+4200 ms" in window.presentation.note.text()
        assert window.training.selected_classifiers() == ["logreg", "svm", "forest"]
        assert len(window.features.quality_map.POSITIONS) == 8
        assert all(rating.good is None for rating in window.features.quality_map.ratings)
        assert not window.features.quality_map.grab().isNull()
        window.training.image_seconds.setValue(2.0)
        window.training.fixation_min.setValue(2.0)
        window.training.fixation_max.setValue(4.0)
        assert "+1700 ms" in window.presentation.note.text()
        for key_name, key_code in (("Escape", Qt.Key.Key_Escape), ("F8", Qt.Key.Key_F8)):
            window.device.provide_test_frames()
            window.poll()
            assert window.training.start.isEnabled()
            assert not window.features.samples
            if key_name == "Escape":
                for key, box in window.training.train_boxes.items():
                    box.setChecked(key == "svm")
            else:
                for key, box in window.training.train_boxes.items():
                    box.setChecked(key in ("logreg", "svm", "forest"))
            window.training.stop_key.setCurrentText(key_name)
            window.start_run()
            app.processEvents()
            assert window.active
            window.poll()
            assert window.features.samples
            assert window.training.settings_scroll.isHidden()
            assert window.training.accuracy.columnCount() == 6
            assert window.training.accuracy.rowCount() == (2 if key_name == "Escape" else 6)
            assert window.run_metadata["classifiers"] == (["svm"] if key_name == "Escape" else ["logreg", "svm", "forest"])
            assert window.run_metadata["epoch_end_ms"] == 1700
            assert window.run_metadata["fixation_min_s"] == 2.0
            assert window.run_metadata["fixation_max_s"] == 4.0
            presentation = window.presentation_window
            presentation.activateWindow()
            presentation.setFocus()
            app.processEvents()
            QTest.keyClick(presentation, key_code)
            app.processEvents()
            assert not window.active
            assert not window.stop_shortcut.isEnabled()
            assert not window.training.save.isEnabled()
            frozen_count = len(window.features.samples)
            frozen_value = window.features.current.text()
            frozen_lights = list(window.features.quality_map.ratings)
            window.device.provide_test_frames()
            window.poll()
            assert len(window.features.samples) == frozen_count
            assert window.features.current.text() == frozen_value
            assert window.features.quality_map.ratings == frozen_lights
            window.device.connected = False
            window.device.connected_changed.emit(False)
            assert not window.features.samples
            assert window.features.current.text() == "Current value: —"
            window.discard_run()
            assert not window.training.start.isEnabled()
            assert not window.training.settings_scroll.isHidden()
        window.device.provide_test_frames()
        window.poll()
        window.start_run()
        assert window.active
        with window.device.ring._lock:
            window.device.ring._last_arrival -= 2
        window.poll()
        assert not window.active
        assert "EEG stream stopped" in window.training.status.text()
        window.device.connected = False
        window.device.connected_changed.emit(False)
        assert not window.features.samples
        window.discard_run()
        trainer = MultiTrainer(["valence", "arousal"], ["logreg", "svm", "forest"],
                               len(window.extractor.names))
        for number in range(16):
            features = np.zeros(len(window.extractor.names))
            features[0] = number % 2
            features[1] = (number // 2) % 2
            trainer.add(features, {"valence": number % 2, "arousal": (number // 2) % 2})
        window.training.configure_models(list(trainer.models), window.extractor.names)
        assert not window.training.save.isEnabled()
        assert not window.training.selected_save_models()
        window.training.prepare_save_selection(trainer)
        assert window.training.save.isEnabled()
        for key, box in window.training.save_boxes.items():
            box.setChecked(key[1] == "forest")
        assert window.training.selected_save_models() == {("valence", "forest"), ("arousal", "forest")}
        for box in window.training.save_boxes.values():
            box.setChecked(False)
        assert not window.training.save.isEnabled()
        window.training.save_boxes[("valence", "forest")].setChecked(True)
        window.trainer = trainer
        window.epochs = [np.zeros((window.epoch_spec.n_samples, len(window.extractor.channels)), dtype=np.float32)]
        window.epoch_trial_numbers = [1]
        window.epoch_labels = [{"valence": 1, "arousal": 0}]
        window.epoch_values = [{"valence": 1.0, "arousal": 0.0}]
        window.epoch_groups = ["happy"]
        window.save_model()
        assert not window.training.save.isEnabled()
        assert len(list((tmp_path / "saved_models").glob("*.forest.joblib"))) == 1
        assert not list((tmp_path / "saved_models").glob("*.logreg.joblib"))
        assert not list((tmp_path / "saved_models").glob("*.svm.joblib"))
    finally:
        window.close()
