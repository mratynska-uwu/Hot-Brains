"""Simulated EEG: a plausible, marker-aligned stream whose label effects the pipeline can learn (testing only)."""

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold, cross_val_score

import hotbrain.gui as gui
from hotbrain import classifiers
from hotbrain.config import CONFIG, AppConfig
from hotbrain.features import TargetFeatures
from hotbrain.quality import SignalQuality
from hotbrain.simulation import SimulatedDevice, is_simulated
from hotbrain.taste import SALTY, SWEET
from test_gui_shortcut import NoDevice

APP_CONFIG = AppConfig.from_dict(CONFIG, Path.cwd())
EEG = APP_CONFIG.eeg
CONDITIONS = {"negative": {"valence": 0, "arousal": 1}, "positive": {"valence": 1, "arousal": 1},
              "neutral": {"valence": None, "arousal": 0}}


def _simulate(effect_uv: float, stimuli: list[dict], duration_s: float, period_s: float, seed: int = 0):
    """Stream a whole run offline (1 s chunks, as the Qt timer would) and return the device and marker codes."""
    QApplication.instance() or QApplication([])
    device = SimulatedDevice(EEG, effect_uv, seed=seed, blinks=False, autostart=False)
    total = len(stimuli) * period_s + duration_s + 5
    device.set_retention(total + 10)
    now = time.perf_counter()
    t0 = now - total
    device.start(t0=t0)
    codes = []
    for i, labels in enumerate(stimuli):
        onset = t0 + 2 + i * period_s
        device.mark_stimulus(onset, duration_s, labels)
        codes.append(device.place_marker(onset))
    for step in np.arange(t0 + 1, now, 1.0):
        device.advance(step)
    device.advance(now)
    device.close()
    return device, codes


def _scene_features(effect_uv: float, profile: int = 3, per_condition: int = 30):
    rng = np.random.default_rng(1)
    order = [c for c in CONDITIONS for _ in range(per_condition)]
    rng.shuffle(order)
    device, codes = _simulate(effect_uv, [CONDITIONS[c] for c in order], 3.0, 4.75)
    features = TargetFeatures(EEG, APP_CONFIG.profile(profile))
    spec = features.epoch_spec(3000)
    rows = []
    for code, condition in zip(codes, order):
        epoch = device.ring.epoch(code, spec)
        assert epoch is not None and epoch.shape == (spec.n_samples, len(EEG.electrodes))
        rows.append((features.extract(epoch, spec), condition))   # no artifact rejection expected
    return features, rows


def test_stream_is_ready_marker_aligned_and_passes_quality_checks():
    QApplication.instance() or QApplication([])
    device = SimulatedDevice(EEG, seed=0, autostart=False)
    assert is_simulated(device) and not device.stream_ready()
    device.start(t0=time.perf_counter() - 3)
    assert device.stream_ready() and device.stream_live() and device.readiness_issue() == "ready"
    code, delay_ms = device.mark_image(time.perf_counter())
    assert code == 1 and delay_ms == 0.0
    # An onset whose sample was already streamed still lands on its exact sample.
    past = time.perf_counter() - 1.0
    code = device.place_marker(past)
    assert device.ring.marker_sample(code) == round((past - device.t0) * EEG.fs)
    assert device.ring.device_delay_samples == 0
    _, samples = device.ring.recent(3)
    assert samples.shape[1] == len(EEG.electrodes)
    assert all(rating.good for rating in SignalQuality(EEG).assess(samples[-2 * EEG.fs:]))
    device.close()
    assert not device.connected


def _t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t statistic of mean(a) - mean(b)."""
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def test_arousal_effect_raises_lpp_and_suppresses_posterior_alpha():
    for effect, expect_effect in ((15.0, True), (0.0, False)):
        features, rows = _scene_features(effect)
        names = features.names["arousal"]
        lpp = [names.index("central ERP 450-800 ms"), names.index("parietal ERP 450-800 ms")]
        alpha = names.index("posterior alpha 8-13 Hz log power")
        emotional = np.array([v["arousal"] for v, c in rows if CONDITIONS[c]["arousal"] == 1])
        neutral = np.array([v["arousal"] for v, c in rows if CONDITIONS[c]["arousal"] == 0])
        t_lpp = _t(emotional[:, lpp].mean(axis=1), neutral[:, lpp].mean(axis=1))
        t_alpha = _t(emotional[:, alpha], neutral[:, alpha])
        if expect_effect:
            assert t_lpp > 3 and t_alpha < -3, (t_lpp, t_alpha)
        else:
            assert abs(t_lpp) < 3 and abs(t_alpha) < 3, (t_lpp, t_alpha)


def test_classifier_learns_simulated_arousal_only_when_there_is_an_effect():
    def balanced_accuracy(effect: float) -> float:
        _, rows = _scene_features(effect)
        x = np.array([v["arousal"] for v, _ in rows])
        y = np.array([CONDITIONS[c]["arousal"] for _, c in rows])
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        return cross_val_score(classifiers.build("lda_shrinkage"), x, y, cv=cv, scoring="balanced_accuracy").mean()

    assert balanced_accuracy(4.0) > 0.7
    assert balanced_accuracy(0.0) < 0.65


def test_taste_effect_sweet_theta_salty_beta():
    sequence = [SWEET, SALTY] * 8
    device, codes = _simulate(6.0, [{"taste": label} for label in sequence], 20.0, 24.0)
    _, data = device.ring.recent(device.retention_s)
    first = device.ring.received_samples - len(data)
    fz, c3 = EEG.electrodes.index("Fz"), EEG.electrodes.index("C3")
    power = {SWEET: [], SALTY: []}
    for code, label in zip(codes, sequence):
        start = device.ring.marker_sample(code) - first
        segment = data[start + 5 * EEG.fs:start + 15 * EEG.fs]
        freqs, psd = welch(segment - segment.mean(axis=0), EEG.fs, nperseg=EEG.fs, axis=0)
        theta = np.log(psd[(freqs >= 5) & (freqs <= 7), fz].mean())
        beta = np.log(psd[(freqs >= 15) & (freqs <= 17), c3].mean())
        power[label].append((theta, beta))
    sweet, salty = np.mean(power[SWEET], axis=0), np.mean(power[SALTY], axis=0)
    assert sweet[0] > salty[0] + 0.3      # frontal-midline theta
    assert salty[1] > sweet[1] + 0.3      # central low beta


def test_gui_switches_source_marks_runs_and_hands_over(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(gui, "DeviceManager", NoDevice)
    app = QApplication.instance() or QApplication([])
    window = gui.MainWindow(tmp_path, CONFIG, simulate=True)
    try:
        assert isinstance(window.device, SimulatedDevice)
        assert window.training.source.currentData() == gui.SOURCE_SIMULATED
        app.processEvents()   # the simulator starts on the event loop
        deadline = time.perf_counter() + 2
        while not window.device.stream_ready() and time.perf_counter() < deadline:
            app.processEvents()
        assert window.device.stream_ready()
        window.poll()
        assert window.training.device_status.text().startswith("SIMULATED EEG")
        window.training.sim_effect.setValue(7.5)
        assert window.device.effect_uv == 7.5
        simulated = window.device
        window.training.source.setCurrentIndex(window.training.source.findData(gui.SOURCE_UNICORN))
        assert isinstance(window.device, NoDevice) and not simulated.connected
        window.training.source.setCurrentIndex(window.training.source.findData(gui.SOURCE_SIMULATED))
        assert isinstance(window.device, SimulatedDevice) and window.device.effect_uv == 7.5
        assert gui._simulated_prefix({"eeg_source": "simulated"}) == "SIMULATED_"
        assert gui._simulated_prefix({"eeg_source": "unicorn"}) == ""

        # The taste experiment adopts the running simulated stream and shows it as its source.
        from hotbrain.taste_gui import open_taste_experiment
        device = window.device
        taste = open_taste_experiment(window)
        try:
            assert taste.device is device
            assert taste.panel.source.currentData() == gui.SOURCE_SIMULATED
            assert taste.panel.sim_effect.value() == 7.5
        finally:
            taste.close()
    finally:
        window.close()
