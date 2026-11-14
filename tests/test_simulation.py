"""Simulated EEG streams plausible, marker-aligned data and encodes labels only when asked to."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from hotbrain.config import CONFIG, EEGSettings
from hotbrain.features import FeatureExtractor
from hotbrain.quality import SignalQuality
from hotbrain.simulation import SimulatedDevice

EEG = EEGSettings.from_dict(CONFIG["eeg"])


def _alpha_power(effect_uv: float) -> tuple[float, float]:
    """Mean Fz alpha power for valence=1 vs valence=0 images, epochs cut by marker like the real device."""
    QApplication.instance() or QApplication([])
    device = SimulatedDevice(EEG, ["valence", "arousal"], effect_uv, seed=1, autostart=False)
    device.set_retention(120)
    now = time.perf_counter()
    device.start(t0=now - 80)
    trials = []
    for i in range(36):
        onset = now - 78 + 2 * i
        valence = i % 2
        device.mark_stimulus(onset, 1.0, {"valence": valence, "arousal": 0})
        trials.append((device.place_marker(onset), valence))
    device.advance(now)
    device.close()
    extractor = FeatureExtractor(EEG)
    spec = extractor.epoch_spec(1000)
    alpha = extractor.names.index("Fz alpha 8-13 Hz log power")
    power = {0: [], 1: []}
    for code, valence in trials:
        epoch = device.ring.epoch(code, spec)
        assert epoch is not None and epoch.shape == (spec.n_samples, len(EEG.electrodes))
        power[valence].append(extractor.extract(epoch, spec)[alpha])
    return float(np.mean(power[1])), float(np.mean(power[0]))


def test_simulated_stream_is_ready_and_passes_quality_checks():
    QApplication.instance() or QApplication([])
    device = SimulatedDevice(EEG, ["valence"], seed=0, autostart=False)
    assert not device.stream_ready()
    device.start(t0=time.perf_counter() - 3)
    device.advance(time.perf_counter())
    assert device.stream_ready() and device.readiness_issue() == "ready"
    code, delay_ms = device.mark_image(time.perf_counter())
    assert code == 1 and delay_ms == 0.0
    # A paint stamp whose sample was already streamed still lands on its exact sample.
    past = time.perf_counter() - 1.0
    code = device.place_marker(past)
    assert device.ring.marker_sample(code) == round((past - device.t0) * EEG.fs)
    _, samples = device.ring.recent(3)
    assert samples.shape[1] == len(EEG.electrodes)
    assert all(rating.good for rating in SignalQuality(EEG).assess(samples[-2 * EEG.fs:]))
    device.close()


def test_effect_raises_alpha_only_for_label_one():
    with_effect = _alpha_power(8.0)
    assert with_effect[0] > with_effect[1] + 0.3
    no_effect = _alpha_power(0.0)
    assert abs(no_effect[0] - no_effect[1]) < 0.3
