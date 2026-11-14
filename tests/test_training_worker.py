"""Trial fitting cannot hold up Qt stimulus events or reorder predictions."""

import os
import copy
import threading
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from hotbrain.dataset import StimulusImage
from hotbrain.config import CONFIG
from hotbrain.gui import MainWindow, PendingEpoch
from hotbrain.training import MultiTrainer


def test_training_worker_keeps_gui_responsive_and_trials_ordered(tmp_path: Path, monkeypatch):
    from hotbrain import gui
    from test_gui_shortcut import NoDevice

    monkeypatch.setattr(gui, "DeviceManager", NoDevice)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    window.trainer = MultiTrainer(["valence"], ["logreg"], len(window.extractor.names))
    window.training.configure_models(list(window.trainer.models), window.extractor.names)
    window.shown = 2
    window.active = True
    main_thread = threading.get_ident()
    worker_threads = []
    painted = []

    def slow_training(extractor, spec, trainer, epoch, labels):
        worker_threads.append(threading.get_ident())
        time.sleep(.15)
        vector = np.full(len(window.extractor.names), epoch[0, 0])
        return vector, trainer.add(vector, labels), 150.0

    monkeypatch.setattr(gui, "train_epoch", slow_training)
    for number, label in ((1, 0), (2, 1)):
        image = StimulusImage(tmp_path / f"{number}.png", "test", {"valence": label})
        item = PendingEpoch(number, image, 0.0, 100.0, number, 0.0)
        window.training_queue.append((item, np.full((10, 8), number, dtype=float), time.perf_counter()))
        window.trial_log.append({"number": number, "eeg": "processing"})
    try:
        started = time.perf_counter()
        window._start_next_training()
        assert time.perf_counter() - started < .05
        QTimer.singleShot(0, lambda: painted.append(threading.get_ident()))
        app.processEvents()
        assert painted == [main_thread]
        assert window._training_busy()
        window.stop_run()
        assert window.finishing
        assert not window.training.discard.isEnabled()
        deadline = time.perf_counter() + 3
        while window.finishing and time.perf_counter() < deadline:
            app.processEvents()
            window.poll()
            time.sleep(.01)
        assert not window.finishing
        assert not window._training_busy()
        assert window.training.discard.isEnabled()
        assert worker_threads == [worker_threads[0], worker_threads[0]]
        assert worker_threads[0] != main_thread
        assert [int(x[0]) for x in window.trainer.models[("valence", "logreg")].x] == [1, 2]
        assert [log["eeg"] for log in window.trial_log] == ["usable", "usable"]
        assert window.usable == 2
    finally:
        window.close()


def test_gui_does_not_stop_on_isolated_invalid_scan(tmp_path: Path, monkeypatch):
    from hotbrain import gui
    from test_gui_shortcut import NoDevice

    monkeypatch.setattr(gui, "DeviceManager", NoDevice)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    try:
        window.device.provide_test_frames()
        window.active = True
        # A recent invalid scan blocks a clean new start, but EEG continues.
        monkeypatch.setattr(window.device, "stream_ready", lambda: False)
        monkeypatch.setattr(window.device, "stream_live", lambda: True)
        window.poll()
        assert window.active
        assert not window.finishing
    finally:
        window.close()


def test_validation_uses_frozen_models_when_no_training_pairs_are_selected(tmp_path: Path, monkeypatch):
    from hotbrain import gui
    from test_gui_shortcut import NoDevice

    class FixedModel:
        def predict(self, x):
            return np.ones(len(x), dtype=int)

        def predict_proba(self, x):
            return np.tile([0.2, 0.8], (len(x), 1))

    monkeypatch.setattr(gui, "DeviceManager", NoDevice)
    for name, valence in (("happy", 1), ("sad", -1)):
        folder = tmp_path / "data" / name
        folder.mkdir(parents=True)
        rows = ["image_name,valence,arousal"]
        for index in range(2):
            image = f"{index}.png"
            assert QImage(12, 12, QImage.Format.Format_RGB32).save(str(folder / image))
            rows.append(f"{image},{valence},{index + 1}")
        folder.with_suffix(".csv").write_text("\n".join(rows), encoding="utf-8")
    config = copy.deepcopy(CONFIG)
    config["datasets"] = ["happy", "sad"]
    config["balance_by"] = "valence"
    config["split"]["validation_fraction"] = 0.5
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path, config)
    try:
        for index, box in enumerate(window.training.electrode_boxes):
            box.setChecked(index in (0, 2, 7))
        window.training.evaluation.setCurrentIndex(1)
        window._set_eval_models({("valence", "logreg"): FixedModel()},
                                {"image_duration_s": 3.0, "evaluation_mode": "split",
                                 "training_image_files": sorted(window._image_files(window.split_train))},
                                "test fixture")
        for box in window.training.train_boxes.values():
            box.setChecked(False)
        window.training.session.setCurrentIndex(1)
        window.training.warmup.setChecked(False)
        window.device.provide_test_frames()
        window.start_run()
        assert window.active
        assert window.training.plot.isHidden()
        assert window.run_metadata["training_model_pairs"] == []
        image = window.next_trial
        window.shown = 1
        window.trial_log.append({"number": 1, "eeg": "processing"})
        item = PendingEpoch(1, image, 0.0, 100.0, 1, 0.0)
        t = np.arange(window.epoch_spec.n_samples) / window.config.eeg.fs
        epoch = np.column_stack([20 * np.sin(2 * np.pi * (8 + i) * t) for i in range(8)])
        window.training_queue.append((item, epoch, time.perf_counter()))
        window._start_next_training()
        deadline = time.perf_counter() + 3
        while window.training_future is not None and time.perf_counter() < deadline:
            app.processEvents()
            window._collect_training_result()
            time.sleep(.01)
        assert window.training_future is None
        assert window.trial_log[0]["eeg"] == "usable"
        assert len(window.extractor.names) == 12
        assert window.validation.accuracy(("valence", "logreg"))[2] == 1
        assert window.run_metadata["training_model_pairs"] == []
    finally:
        window.close()
