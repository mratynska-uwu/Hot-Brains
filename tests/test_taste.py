"""Taste experiment (sweet vs salty): timing, trial order, features, evaluation and the GUI run."""

import copy
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hotbrain.taste import (SALTY, SWEET, TASTE_CONFIG, TASTE_EXPERIMENT, TASTE_TARGET, TasteConfig, TasteFeatures,
                            evaluate_model, learn_trial, taste_sequence)
from hotbrain.training import MultiTrainer

FS = 250
CENTRAL = (1, 2, 3)   # C3, Cz, C4


def synthetic_epoch(config: TasteConfig, label: int, rng: np.random.Generator, emg_window: int | None = None,
                    effect: float = .45) -> np.ndarray:
    """Baseline-to-analysis-end epoch. Sweet trials desynchronise central alpha after +4 s;
    every trial has its own alpha level (slow contact drift) that only the baseline removes."""
    t = config.timing
    n = t.epoch_spec(FS).n_samples
    time_s = np.arange(n) / FS + t.baseline_start_s
    drift = rng.uniform(.5, 2.0)
    data = rng.normal(0, 2.0, (n, 8))
    for channel in range(8):
        amplitude = np.full(n, 10.0 * drift)
        if label == SWEET and channel in CENTRAL:
            amplitude[time_s > 4.0] *= 1 - effect
        data[:, channel] += amplitude * np.sin(2 * np.pi * 10 * time_s + rng.uniform(0, 2 * np.pi))
    if emg_window is not None:
        start = round((t.analysis_start_s + emg_window * t.step_s - t.baseline_start_s) * FS)
        data[start:start + round(t.window_s * FS), 2] += 25 * np.sin(2 * np.pi * 37 * np.arange(round(t.window_s * FS)) / FS)
    return data


def test_default_taste_config_and_presets():
    config = TasteConfig.from_dict(TASTE_CONFIG)
    assert config.classes == ("słone", "słodkie") and config.cue_text == ("SŁONE", "SŁODKIE")
    t = config.timing
    assert (t.trial_s, t.analysis_start_s, t.analysis_end_s) == (20.0, 5.0, 15.0)
    assert t.problems() == [] and t.n_windows() == 9
    assert {p.number: TasteFeatures(config, p).n_features() for p in config.profiles} == \
        {1: {TASTE_TARGET: 18}, 2: {TASTE_TARGET: 18}, 3: {TASTE_TARGET: 56}}
    # The default preset stays below the EMG-dominated bands and uses fronto-central-parietal sites.
    preset = config.profile(1).target(TASTE_TARGET)
    assert set(preset.bands) == {"theta", "alpha", "beta_low"} and "PO7" not in preset.electrodes


@pytest.mark.parametrize("change, message", [
    ({"cue_visible_s": 4.5}, "hide the cue"),
    ({"analysis_end_s": 21.0}, "inside the trial"),
    ({"fixation_min_s": 2.0}, "contain the whole baseline"),
    ({"analysis_end_s": 6.0}, "at least one"),
])
def test_invalid_timing_is_explained(change, message):
    from dataclasses import replace
    timing = replace(TasteConfig.from_dict(TASTE_CONFIG).timing, **change)
    assert any(message in problem for problem in timing.problems())


def test_cue_may_stay_for_the_whole_trial():
    from dataclasses import replace
    timing = replace(TasteConfig.from_dict(TASTE_CONFIG).timing, cue_visible_s=20.0)
    assert timing.problems() == [] and timing.cue_kept


@pytest.mark.parametrize("per_taste, max_run", [(20, 2), (5, 1), (13, 3), (2, 2)])
def test_sequence_is_balanced_with_short_runs(per_taste, max_run):
    for seed in range(30):
        sequence = taste_sequence(per_taste, max_run, random.Random(seed))
        assert len(sequence) == 2 * per_taste and sum(sequence) == per_taste
        longest = max(len(run) for run in "".join(map(str, sequence)).replace("01", "0 1").replace("10", "1 0").split())
        assert longest <= max_run


def test_trial_features_use_baseline_reject_emg_and_fall_back():
    config = TasteConfig.from_dict(TASTE_CONFIG)
    taste = TasteFeatures(config, config.profile(1))
    rng = np.random.default_rng(0)
    trial = taste.trial(synthetic_epoch(config, SWEET, rng, emg_window=3))
    assert trial.rejected is None and trial.baseline_source == "trial"
    # The 2 s burst overlaps three 2 s windows taken every 1 s.
    assert trial.n_windows == 9 and len(trial.rows) == 6 and trial.reasons == {"emg": 3}
    # Baseline-corrected central alpha power drops for sweet (log10 of (1 - .45)^2 ≈ -0.52).
    names = taste.names[TASTE_TARGET]
    alpha = [names.index(f"{e} alpha 8-13 Hz log power") for e in ("C3", "Cz", "C4")]
    assert np.mean([r[TASTE_TARGET][alpha] for r in trial.rows]) < -.35
    salty = taste.trial(synthetic_epoch(config, SALTY, rng))
    assert abs(np.mean([r[TASTE_TARGET][alpha] for r in salty.rows])) < .15
    # A trial whose baseline is all artifact uses the median of the earlier clean baselines.
    broken = synthetic_epoch(config, SALTY, rng)
    broken[:round(2.6 * FS), 0] += 300 * np.sin(2 * np.pi * 5 * np.arange(round(2.6 * FS)) / FS)
    assert taste.trial(broken).baseline_source == "session median"
    fresh = TasteFeatures(config, config.profile(1))
    assert fresh.trial(broken).rejected.startswith("no clean baseline")


def test_trial_level_learning_and_grouped_evaluation():
    config = TasteConfig.from_dict(TASTE_CONFIG)
    taste = TasteFeatures(config, config.profile(1))
    pairs = [(TASTE_TARGET, "lda_shrinkage")]
    trainer = MultiTrainer([TASTE_TARGET], ["lda_shrinkage"], taste.n_features(), pairs)
    rng = np.random.default_rng(1)
    prequential = []
    for number, label in enumerate(taste_sequence(12, 2, random.Random(1)), start=1):
        outcome = learn_trial(taste, trainer, synthetic_epoch(config, label, rng), label, number)
        if outcome.probabilities:
            prequential.append((label, outcome.probabilities[pairs[0]]))
    model = trainer.models[pairs[0]]
    assert len(set(model.tags)) == 24 and len(model.x) == 24 * 9
    late = prequential[-12:]
    assert np.mean([(p >= .5) == label for label, p in late]) >= .9
    report = evaluate_model("lda_shrinkage", model, {"cv_folds": 5, "permutations": 19, "seed": 0}, prequential)
    assert report["trials"] == 24 and report["cv"]["folds"] == 5
    assert report["cv"]["trial_level"]["balanced_accuracy"] >= .9
    assert report["permutation"]["p_value"] <= .1
    assert report["prequential"]["n"] == len(prequential)


def test_grouped_cv_does_not_leak_trial_identity():
    """Windows of a trial share an offset but the taste has no effect: trial-level CV must stay near chance."""
    rng = np.random.default_rng(2)
    x, y, tags = [], [], []
    for trial in range(30):
        offset = rng.normal(0, 3, 6)
        for _ in range(9):
            x.append(offset + rng.normal(0, .1, 6))
            y.append(trial % 2)
            tags.append(trial)
    trainer = SimpleNamespace(x=x, y=y, tags=tags, n_features=6)
    report = evaluate_model("lda_shrinkage", trainer, {"cv_folds": 5, "permutations": 0, "seed": 0})
    assert report["cv"]["trial_level"]["balanced_accuracy"] < .75


# ---------------------------------------------------------------------------------------------- GUI run
class MarkerDevice:
    """Unit-test stand-in for the Unicorn: frames are fed by the test and carry the trial markers."""

    def __init__(self, eeg, serial):
        from PySide6.QtCore import QObject, Signal

        class Signals(QObject):
            status_changed = Signal(str)
            connected_changed = Signal(bool)

        from hotbrain.acquisition import SampleRing
        self._signals = Signals()
        self.status_changed = self._signals.status_changed
        self.connected_changed = self._signals.connected_changed
        self.eeg, self.serial, self.oscar = eeg, serial, True
        self.connected = True
        self.ring = SampleRing(60, eeg.fs, eeg.device_channels)
        self._code, self._waiting = 0, None
        self.closed = False

    def feed(self, data: np.ndarray):
        markers = np.zeros((len(data), 1))
        if self._waiting is not None:
            markers[0, 0] = self._waiting
            self._waiting = None
        self.ring.append(np.hstack([data, markers]), marker_channel=8)

    def stream_ready(self, max_age_s=1.0):
        return self.connected and time.perf_counter() - self.ring.last_arrival <= 5

    stream_live = stream_ready

    def mark_image(self, stamp):
        self._code += 1
        self._waiting = self._code
        return self._code, 0.0

    def configure(self, oscar, serial):
        self.oscar, self.serial = oscar, serial

    def set_retention(self, seconds):
        self.ring.retention_s = seconds

    def close(self):
        self.closed = True


def _run_trials(app, window, device, config, rng):
    """Drive every trial of a started run through the real onset/end/poll path (no waiting on timers)."""
    t = config.timing
    pre = round(-t.baseline_start_s * FS)
    while window.active:
        window.fixation_timer.stop()
        label = window.next_label
        epoch = synthetic_epoch(config, label, rng)
        device.feed(np.random.default_rng(0).normal(0, 2, (FS, 8)))   # earlier fixation
        device.feed(epoch[:pre])                                           # baseline
        window.on_cue_onset(time.perf_counter())
        assert window.trial_log[-1]["cue"] in ("SŁONE", "SŁODKIE", "SMAKUJ")
        device.feed(epoch[pre:])
        device.feed(np.random.default_rng(1).normal(0, 2, (round(5.5 * FS), 8)))   # rest of the tasting period
        window.cue_timer.stop()
        window.trial_timer.stop()
        window.end_trial()
        window.rinse_timer.stop()
        window.pending[-1]["ready_at"] = 0.0
        deadline = time.perf_counter() + 20
        while (window.pending or window.queue or window.future) and time.perf_counter() < deadline:
            window.poll()
            app.processEvents()
            time.sleep(.01)
        assert not window.pending and not window.future
        window.begin_fixation()
    while window.finishing:
        window.poll()
        time.sleep(.01)


def test_taste_window_trains_saves_loads_and_tests(tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    import hotbrain.taste_gui as tg

    monkeypatch.setattr(tg, "DeviceManager", MarkerDevice)
    app = QApplication.instance() or QApplication([])
    window = tg.TasteWindow(tmp_path, copy.deepcopy(TASTE_CONFIG))
    config, device = window.config, window.device
    window.show_windows()
    rng = np.random.default_rng(3)
    try:
        device.feed(rng.normal(0, 2, (FS, 8)))
        window.poll()
        assert window.panel.start.isEnabled(), window.panel.start_hint.text()
        assert set(window.panel.selected_pairs()) == {(TASTE_TARGET, "lda_shrinkage"), (TASTE_TARGET, "logreg")}
        window.panel.per_taste.setValue(8)
        window.panel.warmup.setChecked(False)
        window.start_run()
        assert window.active and window.presentation.stimulus.state == "fixation"
        assert window.fixation_timer.isActive() and len(window.sequence) == 16
        _run_trials(app, window, device, config, rng)
        assert window.shown == 16 and window.usable == 16
        assert window.trainer.any_fitted and window.panel.save.isEnabled()
        accuracy, balanced, n = window.scores.accuracy((TASTE_TARGET, "lda_shrinkage"))
        assert n >= 12 and balanced >= 80
        files = window.save_results()
        manifest = next(p for p in files if p.suffix == ".json")
        assert json.loads(manifest.read_text(encoding="utf-8"))["experiment"] == TASTE_EXPERIMENT
        with np.load(next(p for p in files if p.name.endswith(".epochs.npz"))) as epochs:
            assert epochs["eeg_uV"].shape == (16, config.timing.epoch_spec(FS).n_samples, 8)
        deadline = time.perf_counter() + 60
        while window.report_future is not None and time.perf_counter() < deadline:
            window.poll()
            time.sleep(.05)
        assert list((tmp_path / "training_reports").glob("taste_*.md"))

        # Test session with the saved models, blind cues.
        window.eval_models = {}
        assert window.load_models_from(manifest)
        window.panel.session.setCurrentIndex(window.panel.session.findData(tg.SESSION_TEST))
        window.panel.blind.setChecked(True)
        window.panel.per_taste.setValue(2)
        device.feed(rng.normal(0, 2, (FS, 8)))
        window.poll()
        assert window.panel.start.isEnabled(), window.panel.start_hint.text()
        window.start_run()
        assert window.presentation.gauge.isVisibleTo(window.presentation)
        _run_trials(app, window, device, config, rng)
        assert {entry["cue"] for entry in window.trial_log} == {"SMAKUJ"}
        assert window.scores.accuracy((TASTE_TARGET, "lda_shrinkage"))[2] == 4
        assert "trial_scores" in window.run_metadata
    finally:
        window.close()
        app.processEvents()
    assert device.closed


def test_bad_trial_is_neither_learned_nor_scored(tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    import hotbrain.taste_gui as tg

    monkeypatch.setattr(tg, "DeviceManager", MarkerDevice)
    app = QApplication.instance() or QApplication([])
    window = tg.TasteWindow(tmp_path, copy.deepcopy(TASTE_CONFIG))
    device = window.device
    try:
        device.feed(np.zeros((FS, 8)) + np.random.default_rng(0).normal(0, 2, (FS, 8)))
        window.panel.per_taste.setValue(2)
        window.panel.warmup.setChecked(False)
        window.start_run()
        window.fixation_timer.stop()
        device.feed(np.random.default_rng(1).normal(0, 2, (3 * FS, 8)))
        window.on_cue_onset(time.perf_counter())
        window.mark_bad()
        window.stop_run()
        while window.finishing:
            window.poll()
            time.sleep(.01)
        assert window.trial_log[0]["eeg"] == "marked_bad" and window.usable == 0
        assert not any(t.x for t in window.trainer.models.values())
    finally:
        window.close()
        app.processEvents()
