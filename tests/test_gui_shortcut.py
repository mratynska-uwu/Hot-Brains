"""The stop shortcut must work from a detached participant window."""

import copy
import os
from pathlib import Path

import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import hotbrain.gui as gui
from hotbrain.config import CONFIG
from hotbrain.training import MultiTrainer


class NoDevice(QObject):
    status_changed = Signal(str)
    connected_changed = Signal(bool)

    def __init__(self, eeg, serial):
        super().__init__()
        self.connected = False
        self.oscar = False

    def configure(self, oscar, serial):
        self.oscar = oscar

    def set_retention(self, seconds):
        self.retention_s = seconds

    def close(self):
        pass


def test_chosen_key_stops_from_detached_presentation(tmp_path: Path, monkeypatch):
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
    window.show()
    window.training.warmup.setChecked(False)
    try:
        assert window.dataset is not None, window.training.dataset_status.text()
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
            assert window.training.settings_widget.isHidden()
            assert window.training.accuracy.columnCount() == 6
            assert window.training.accuracy.rowCount() == (2 if key_name == "Escape" else 6)
            assert window.run_metadata["classifiers"] == (["svm"] if key_name == "Escape" else ["logreg", "svm", "forest"])
            assert window.run_metadata["epoch_end_ms"] == 1700
            assert window.run_metadata["fixation_min_s"] == 2.0
            assert window.run_metadata["fixation_max_s"] == 4.0
            presentation = window.detached[1]
            presentation.activateWindow()
            presentation.setFocus()
            app.processEvents()
            QTest.keyClick(presentation, key_code)
            app.processEvents()
            assert not window.active
            assert not window.stop_shortcut.isEnabled()
            assert not window.training.save.isEnabled()
            window.discard_run()
            assert not window.training.settings_widget.isHidden()
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
