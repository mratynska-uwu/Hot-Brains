import random

import numpy as np

from hotbrain.features import EpochSpec, FEATURE_NAMES, N_SAMPLES, extract
from hotbrain.acquisition import SampleRing
from hotbrain.training import OnlineTrainer, TrialPlan, predict_saved


def test_finite_plan_balances_and_uses_images_before_repeating(tmp_path):
    images = {label: [tmp_path / f"{label}{i}.png" for i in range(3)] for label in ("happy", "sad")}
    plan = TrialPlan(images, 20, random.Random(42))
    trials = [plan.next() for _ in range(20)]
    assert plan.next() is None
    assert sum(label == "happy" for label, _ in trials) == 10
    assert sum(label == "sad" for label, _ in trials) == 10
    for label in images:
        first_three = [path for name, path in trials if name == label][:3]
        assert len(set(first_three)) == 3


def test_features_and_prequential_model_export(tmp_path):
    rng = np.random.default_rng(3)
    waveform = rng.normal(0, 4, (N_SAMPLES, 8))
    features = extract(waveform)
    assert features.shape == (len(FEATURE_NAMES),) == (20,)
    trainer = OnlineTrainer()
    for number in range(20):
        label = "happy" if number % 2 else "sad"
        prediction = trainer.add(features + rng.normal(0, .1, len(features)) + (number % 2), label)
        if number < 10:
            assert prediction is None
    assert trainer.model is not None
    assert len(trainer.predictions) == 10
    stem = tmp_path / "model"
    trainer.save(stem, {"participant": "test"})
    probability = predict_saved(stem, features)
    sklearn_probability = trainer.model.predict_proba(features[None, :])[0, 1]
    assert np.isclose(probability, sklearn_probability)


def test_bursty_frame_arrivals_preserve_a_complete_epoch(monkeypatch):
    ring = SampleRing()
    callback_times = iter(100.012 + .016 * i + (.03 if i % 2 else -.02) for i in range(63))
    monkeypatch.setattr("hotbrain.acquisition.time.perf_counter", lambda: next(callback_times))
    signal = np.tile(np.arange(250, dtype=float)[:, None], (1, 8))
    for first in range(0, 250, 4):
        ring.append(signal[first:first + 4], 0)
    epoch = ring.epoch(100.2)
    assert epoch is not None
    assert epoch.shape == (N_SAMPLES, 8)
    assert np.allclose(np.diff(ring.recent(2)[0]), 1 / 250)


def test_two_second_image_extends_epoch_and_features():
    spec = EpochSpec(2000)
    assert spec.end_ms == 1700
    assert spec.n_samples == 475
    rng = np.random.default_rng(4)
    values = rng.normal(0, 4, (spec.n_samples, 8))
    assert extract(values, spec).shape == (20,)


def test_ring_extracts_long_epoch(monkeypatch):
    ring = SampleRing()
    monkeypatch.setattr("hotbrain.acquisition.time.perf_counter", lambda: 104.0)
    ring.append(np.tile(np.arange(1000, dtype=float)[:, None], (1, 8)), 0)
    spec = EpochSpec(2000)
    epoch = ring.epoch(101.0, spec)
    assert epoch is not None
    assert epoch.shape == (475, 8)
