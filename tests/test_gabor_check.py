"""The preflight visual check stays balanced and separate from emotion fitting."""
import random

import numpy as np

from hotbrain.config import CONFIG, EEGSettings
from hotbrain.features import EpochSpec
from hotbrain.gabor_check import TRIALS, score_gabor, trial_plan


def test_gabor_plan_has_balanced_later_holdout():
    plan = trial_plan(random.Random(7))
    assert len(plan) == TRIALS
    assert [sum(label == value for label, _ in plan[:80]) for value in (0, 1)] == [40, 40]
    assert [sum(label == value for label, _ in plan[80:]) for value in (0, 1)] == [20, 20]
    assert all(role == "training" for _, role in plan[:80])
    assert all(role == "held_out" for _, role in plan[80:])


def test_gabor_check_detects_a_known_visual_evoked_response():
    eeg = EEGSettings.from_dict(CONFIG["eeg"])
    spec = EpochSpec(800, eeg.fs)
    rng = np.random.default_rng(3)
    trials = []
    for label, role in trial_plan(random.Random(9)):
        epoch = rng.normal(0, 5, (spec.n_samples, 8))
        if label:
            t = np.arange(spec.n_samples) / eeg.fs - .2
            epoch[:, 5:8] += (25 * np.exp(-((t - .12) / .045) ** 2))[:, None]
        trials.append((epoch, label, role))
    result = score_gabor(trials, eeg, spec)
    assert result["usable_training"] == 80
    assert result["usable_held_out"] == 40
    assert result["visual_response_detected"]
    assert result["balanced_accuracy"] > .9
