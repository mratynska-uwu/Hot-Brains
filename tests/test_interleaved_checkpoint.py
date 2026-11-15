"""Held-out EEG stays out of fitting and selects a saved checkpoint."""
import os
import random
import time
import json
import copy

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage

from hotbrain.config import CONFIG
from hotbrain.dataset import StimulusImage
from hotbrain.gui import MainWindow, PendingEpoch
from hotbrain.training import InterleavedPlan, MultiTrainer, select_best_checkpoints


def test_interleaved_plan_balances_roles_and_never_crosses_image_identity(tmp_path):
    train = {group: [StimulusImage(tmp_path / f"train_{group}_{i}.png", group, {"valence": label})
                     for i in range(3)] for group, label in (("negative", 0), ("positive", 1))}
    heldout = {group: [StimulusImage(tmp_path / f"heldout_{group}_{i}.png", group, {"valence": label})
                       for i in range(2)] for group, label in (("negative", 0), ("positive", 1))}
    plan = InterleavedPlan(train, heldout, 20, .2, random.Random(4))
    seen = []
    for _ in range(20):
        image = plan.next()
        seen.append((plan.last_role, image))
    assert plan.next() is None
    assert sum(role == "training" for role, _ in seen) == 16
    assert sum(role == "validation" for role, _ in seen) == 4
    for role in ("training", "validation"):
        assert sorted(image.labels["valence"] for current, image in seen if current == role).count(0) == (
            8 if role == "training" else 2)
    assert all(image.path.name.startswith("train_" if role == "training" else "heldout_")
               for role, image in seen)


def test_checkpoint_selection_uses_same_final_validation_set_for_all_candidates():
    class FixedModel:
        def __init__(self, correct):
            self.correct = correct

        def predict_proba(self, x):
            positive = np.where(x[:, 0] == 1, .9 if self.correct else .1,
                                .1 if self.correct else .9)
            return np.column_stack([1 - positive, positive])

    features = [np.array([i % 2], dtype=float) for i in range(20)]
    labels = [{"valence": i % 2} for i in range(20)]
    selected = select_best_checkpoints({("valence", "logreg"): [(20, FixedModel(True)),
                                                                 (40, FixedModel(False))]},
                                       features, labels)
    choice = selected[("valence", "logreg")]
    assert choice["training_trials"] == 20
    assert np.isclose(choice["validation_brier"], .01)
    assert choice["validation_auc"] == 1.0
    assert choice["validation_n"] == 20
    assert choice["beats_coinflip_brier"]
    poor = select_best_checkpoints({("valence", "logreg"): [(40, FixedModel(False))]},
                                   features, labels)
    assert not poor[("valence", "logreg")]["beats_coinflip_brier"]


def test_validation_epoch_does_not_enter_training_fit(tmp_path, monkeypatch):
    from hotbrain import gui
    from test_gui_shortcut import NoDevice

    monkeypatch.setattr(gui, "DeviceManager", NoDevice)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path)
    window.trainer = MultiTrainer(["valence"], ["logreg"], len(window.extractor.names), checkpoint_every=20)
    window.training.configure_models(list(window.trainer.models), window.extractor.names, split_mode=True)
    window.shown = 1
    image = StimulusImage(tmp_path / "heldout.png", "valence=positive",
                          {"valence": 1, "arousal": 1}, {"valence": 2.0, "arousal": 3.0})
    t = np.arange(window.epoch_spec.n_samples) / window.config.eeg.fs
    epoch = np.column_stack([20 * np.sin(2 * np.pi * (8 + i) * t) for i in range(8)])
    window.trial_log.append({"number": 1, "eeg": "processing"})
    window.training_queue.append((PendingEpoch(1, image, 0, 10, 1, 0, "validation"), epoch,
                                  time.perf_counter()))
    try:
        window._start_next_training()
        deadline = time.perf_counter() + 3
        while window.training_future is not None and time.perf_counter() < deadline:
            app.processEvents()
            window._collect_training_result()
        assert window.trial_log[0]["eeg"] == "usable"
        assert window.trainer.training_trials == 0
        assert not window.trainer.models[("valence", "logreg")].x
        assert len(window.heldout_vectors) == 1
        assert window.epoch_roles == ["validation"]
    finally:
        window.close()


def test_save_uses_selected_checkpoint_instead_of_latest_fit(tmp_path):
    trainer = MultiTrainer(["valence"], ["logreg"], 2)
    for i in range(12):
        trainer.add(np.array([i % 2, i / 12]), {"valence": i % 2})
    assert trainer.predict(np.array([1.0, .5]), {"valence": 1})
    assert trainer.training_trials == 12
    assert len(trainer.models[("valence", "logreg")].x) == 12
    checkpoint = DummyClassifier(strategy="constant", constant=0).fit(
        np.array([[0, 0], [1, 1]]), np.array([0, 1]))
    selected = {("valence", "logreg"): {"model": checkpoint, "training_trials": 10,
                                        "validation_brier": .1, "validation_accuracy": .8,
                                        "validation_n": 20, "candidate_checkpoints": 2}}
    trainer.save(tmp_path / "run", {"evaluation_mode": "split"},
                 {("valence", "logreg")}, selected)
    saved = joblib.load(tmp_path / "run.valence.logreg.joblib")
    assert saved.predict(np.array([[1, 1]]))[0] == 0
    manifest = json.loads((tmp_path / "run.json").read_text())
    assert manifest["models"]["valence/logreg"]["n_eeg_trials"] == 10
    assert manifest["models"]["valence/logreg"]["checkpoint_selection"]["validation_brier"] == .1


def test_gui_interleaved_run_selects_checkpoint_before_save(tmp_path, monkeypatch):
    from hotbrain import gui
    from test_gui_shortcut import NoDevice

    monkeypatch.setattr(gui, "DeviceManager", NoDevice)
    for name, valence in (("negative", -1), ("positive", 1)):
        folder = tmp_path / "data" / name
        folder.mkdir(parents=True)
        rows = ["image_name,valence,arousal"]
        for i in range(2):
            assert QImage(12, 12, QImage.Format.Format_RGB32).save(str(folder / f"{i}.png"))
            rows.append(f"{i}.png,{valence},{i + (3 if valence > 0 else 1)}")
        folder.with_suffix(".csv").write_text("\n".join(rows), encoding="utf-8")
    config = copy.deepcopy(CONFIG)
    config["datasets"] = ["negative", "positive"]
    config["split"]["validation_fraction"] = .5
    app = QApplication.instance() or QApplication([])
    window = MainWindow(tmp_path, config)
    try:
        window.training.warmup.setChecked(False)
        window.training.count.setValue(40)
        window.device.provide_test_frames()
        window.start_run()
        assert window.active and isinstance(window.plan, InterleavedPlan)
        window.fixation_timer.stop()
        for i in range(20):
            vector = np.zeros(len(window.extractor.names))
            vector[0] = i % 2
            vector[1] = i / 20
            window.trainer.add(vector, {"valence": i % 2})
            window.heldout_vectors.append(vector.copy())
            window.heldout_labels.append({"valence": i % 2})
        window.stop_run()
        deadline = time.perf_counter() + 5
        while window.finishing and time.perf_counter() < deadline:
            app.processEvents()
            window.poll()
            time.sleep(.01)
        assert not window.finishing
        assert ("valence", "logreg") in window.best_models
        assert window.training.save.isEnabled()
        window.save_model()
        manifest = json.loads(next((tmp_path / "saved_models").glob("scene_*.json")).read_text())
        assert manifest["models"]["valence/logreg"]["checkpoint_selection"]["validation_n"] == 20
        assert manifest["validation_used_for_selection"] is True
        assert (tmp_path / "recordings").is_dir()
    finally:
        window.close()
