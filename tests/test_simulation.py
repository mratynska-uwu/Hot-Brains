"""Simulated EEG streams plausible data and encodes labels only when asked to."""

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
    """Mean Fz alpha power for valence=1 vs valence=0 images."""
    QApplication.instance() or QApplication([])
    device = SimulatedDevice(EEG, ["valence", "arousal"], effect_uv, seed=1, autostart=False)
    device.set_retention(120)
    now = time.perf_counter()
    device.start(t0=now - 80)
    onsets = [(now - 78 + 2 * i, i % 2) for i in range(36)]
    for onset, valence in onsets:
        device.mark_stimulus(onset, 1.0, {"valence": valence, "arousal": 0})
    device.advance(now)
    device.close()
    extractor = FeatureExtractor(EEG)
    spec = extractor.epoch_spec(1000)
    alpha = extractor.names.index("Fz alpha 8-13 Hz log power")
    power = {0: [], 1: []}
    for onset, valence in onsets:
        epoch = device.ring.epoch(onset, spec)
        assert epoch is not None and epoch.shape == (spec.n_samples, len(EEG.electrodes))
        power[valence].append(extractor.extract(epoch, spec)[alpha])
    return float(np.mean(power[1])), float(np.mean(power[0]))


def test_simulated_signal_passes_quality_checks():
    QApplication.instance() or QApplication([])
    device = SimulatedDevice(EEG, ["valence"], seed=0, autostart=False)
    device.start(t0=time.perf_counter() - 3)
    device.advance(time.perf_counter())
    _, samples = device.ring.recent(3)
    assert samples.shape[1] == len(EEG.electrodes)
    assert all(rating.good for rating in SignalQuality(EEG).assess(samples))


def test_effect_raises_alpha_only_for_label_one():
    with_effect = _alpha_power(8.0)
    assert with_effect[0] > with_effect[1] + 0.3
    no_effect = _alpha_power(0.0)
    assert abs(no_effect[0] - no_effect[1]) < 0.3
