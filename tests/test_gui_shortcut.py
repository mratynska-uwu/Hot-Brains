"""The stop shortcut must work from a detached participant window."""

import copy
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import hotbrain.gui as gui
from hotbrain.config import CONFIG


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
        assert len(window.features.quality_map.POSITIONS) == 8
        assert all(rating.good is None for rating in window.features.quality_map.ratings)
        assert not window.features.quality_map.grab().isNull()
        window.training.image_seconds.setValue(2.0)
        window.training.fixation_min.setValue(2.0)
        window.training.fixation_max.setValue(4.0)
        assert "+1700 ms" in window.presentation.note.text()
        for key_name, key_code in (("Escape", Qt.Key.Key_Escape), ("F8", Qt.Key.Key_F8)):
            if key_name == "Escape":
                svm_index = next(i for i in range(window.training.model_choice.count())
                                 if window.training.model_choice.itemData(i) == ["svm"])
                window.training.model_choice.setCurrentIndex(svm_index)
            else:
                window.training.model_choice.setCurrentIndex(0)
            window.training.stop_key.setCurrentText(key_name)
            window.start_run()
            app.processEvents()
            assert window.active
            assert window.training.accuracy.columnCount() == 5
            assert window.training.accuracy.rowCount() == (2 if key_name == "Escape" else 4)
            assert window.run_metadata["classifiers"] == (["svm"] if key_name == "Escape" else ["logreg", "svm"])
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
            window.discard_run()
    finally:
        window.close()
