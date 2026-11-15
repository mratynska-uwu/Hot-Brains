"""Per-target feature settings and the neutral-image control condition between trials."""

import copy
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

import hotbrain.gui as gui
from conftest import classic_config
from hotbrain.config import CONFIG, AppConfig
from hotbrain.features import TargetFeatures

ROOT = Path(__file__).resolve().parents[1]
APP = AppConfig.from_dict(CONFIG, ROOT)


def test_targets_can_use_their_own_analysis_window():
    preset = APP.profile(1)
    profile = replace(preset, targets=(("valence", replace(preset.target("valence"), window_ms=None)),
                                       ("arousal", replace(preset.target("arousal"), window_ms=600))))
    features = TargetFeatures(APP.eeg, profile)
    spec = features.epoch_spec(20000)
    assert spec.end_ms == 19700                          # the epoch spans the longest window
    assert features.target_spec("arousal", spec).end_ms == 600
    assert features.target_spec("valence", spec) is spec
    epoch = np.random.default_rng(1).normal(0, 5, (spec.n_samples, len(APP.eeg.electrodes)))
    short = TargetFeatures(APP.eeg, replace(profile, targets=(("valence", profile.target("valence")),
                                                              ("arousal", replace(profile.target("arousal"),
                                                                                  window_ms=None)))))
    # Arousal features from the 600 ms window differ from the whole-image ones; valence is unchanged.
    a, b = features.extract(epoch, spec), short.extract(epoch, spec)
    assert np.allclose(a["valence"], b["valence"]) and not np.allclose(a["arousal"], b["arousal"])
    assert features.describe()["arousal"]["window_ms"] == 600
    # An empty electrode selection is a problem for that target, not "all electrodes".
    empty = TargetFeatures(APP.eeg, replace(profile, targets=(("valence", replace(profile.target("valence"),
                                                                                electrodes=())),
                                                              ("arousal", profile.target("arousal")))))
    assert "valence" in empty.problems and empty.targets == ("arousal",)


def _window(tmp_path: Path, monkeypatch, neutral_images: int):
    from test_gui_shortcut import NoDevice

    monkeypatch.setattr(gui, "DeviceManager", NoDevice)
    for folder_name, valence in (("set/happy", 3), ("set/sad", -3)):
        folder = tmp_path / "data" / folder_name
        folder.mkdir(parents=True)
        lines = ["image_name,valence,arousal"]
        for i in range(4):
            assert QImage(12, 12, QImage.Format.Format_RGB32).save(str(folder / f"s{i}.png"))
            lines.append(f"s{i}.png,{valence},{i + 1}")
        folder.with_suffix(".csv").write_text("\n".join(lines), encoding="utf-8")
    neutral = tmp_path / "data" / "neutral"
    neutral.mkdir()
    for i in range(neutral_images):
        assert QImage(12, 12, QImage.Format.Format_RGB32).save(str(neutral / f"n{i}.png"))
    config = classic_config()   # synthetic CSVs: valence/arousal only
    config["datasets"] = ["set/happy", "set/sad"]
    config["neutral_images"] = "neutral"
    QApplication.instance() or QApplication([])
    return gui.MainWindow(tmp_path, config)


def test_neutral_images_fill_the_gap_between_trials(tmp_path: Path, monkeypatch):
    window = _window(tmp_path, monkeypatch, neutral_images=2)
    training = window.training
    try:
        assert training.image_seconds.value() == 3.0
        training.between.setCurrentIndex(training.between.findData(gui.BETWEEN_NEUTRAL))
        assert training.neutral_seconds.value() == 20.0
        assert training.presentation_form.isRowVisible(training.neutral_row)
        assert not training.presentation_form.isRowVisible(training.fixation_row)
        assert "neutral image 20 s" in window.presentation.note.text()
        training.image_seconds.setValue(2.0)
        training.neutral_seconds.setValue(1.0)
        training.warmup.setChecked(False)
        training.count.setValue(4)
        window.device.provide_test_frames()
        window.poll()
        assert training.start.isEnabled(), training.start_hint.text()
        window.start_run()
        assert window.active
        assert window.presentation.stimulus.state == "neutral"     # not "image": no onset marker
        assert window.pre_stimulus["kind"] == "neutral" and window.pre_stimulus["duration_s"] == 1.0
        assert window.run_metadata["between_images"] == "neutral"
        assert sorted(window.run_metadata["neutral_images"]) == ["n0.png", "n1.png"]
        drawn = [window._next_neutral().name for _ in range(6)]
        assert set(drawn) == {"n0.png", "n1.png"}                  # repeats are allowed
        window.stop_run("test")
    finally:
        window.close()


def test_missing_neutral_images_block_start_with_a_reason(tmp_path: Path, monkeypatch):
    window = _window(tmp_path, monkeypatch, neutral_images=0)
    training = window.training
    try:
        window.device.provide_test_frames()
        window.poll()
        assert training.start.isEnabled()
        training.between.setCurrentIndex(training.between.findData(gui.BETWEEN_NEUTRAL))
        window.poll()
        assert not training.start.isEnabled()
        assert "neutral images" in training.start_hint.text()
    finally:
        window.close()


def test_valence_and_arousal_settings_are_independent(tmp_path: Path, monkeypatch):
    window = _window(tmp_path, monkeypatch, neutral_images=1)
    editors = window.training.target_editors
    try:
        for index, box in enumerate(editors["arousal"].electrode_boxes):
            box.setChecked(index in (5, 6, 7))
        editors["arousal"].window.setValue(800)
        editors["arousal"].model_boxes["logreg"].setChecked(True)
        described = window.target_features.describe()
        assert described["arousal"]["electrodes"] == ["PO7", "Oz", "PO8"]
        assert described["arousal"]["window_ms"] == 800
        assert described["valence"]["electrodes"] == list(APP.eeg.electrodes)
        assert described["valence"]["window_ms"] is None
        assert ("arousal", "logreg") in window.training.selected_model_pairs()
        assert ("valence", "logreg") not in window.training.selected_model_pairs()
        assert "edited" in window.training.electrode_status.text()
        window.training.apply_preset.click()                       # back to the preset
        assert window.target_features.describe()["arousal"]["electrodes"] == list(APP.eeg.electrodes)
        assert "edited" not in window.training.electrode_status.text()
    finally:
        window.close()
