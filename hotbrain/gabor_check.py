"""Short visual-response check, kept separate from emotion training."""
from __future__ import annotations

import random

import numpy as np
from scipy.signal import sosfiltfilt
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import EpochSpec, START_MS, make_filter
from .config import EEGSettings

TRIALS = 120
TRAIN_TRIALS = 80
STIMULUS_MS = 800
FIXATION_MIN_MS = 500
FIXATION_MAX_MS = 800
BREAK_MS = 5000


def trial_plan(rng: random.Random | None = None) -> list[tuple[int, str]]:
    """Balanced, randomized train and held-out blocks of distinct presentations."""
    rng = rng or random.Random()
    train = [0, 1] * (TRAIN_TRIALS // 2)
    test = [0, 1] * ((TRIALS - TRAIN_TRIALS) // 2)
    rng.shuffle(train)
    rng.shuffle(test)
    return [(label, "training") for label in train] + [(label, "held_out") for label in test]


def gabor_features(epoch: np.ndarray, settings: EEGSettings, spec: EpochSpec) -> np.ndarray:
    """Predeclared visual ERP windows on PO7, Oz, PO8, relative to baseline."""
    data = np.asarray(epoch, dtype=np.float64)
    indices = [settings.electrodes.index(name) for name in ("PO7", "Oz", "PO8")]
    if data.shape != (spec.n_samples, len(settings.electrodes)) or not np.isfinite(data).all():
        raise ValueError("incomplete Gabor epoch")
    if np.ptp(data, axis=0).max() > 500 or np.ptp(data, axis=0).min() < .05:
        raise ValueError("artifact or flat channel")
    data = sosfiltfilt(make_filter(settings), data, axis=0)
    data -= data[:spec.baseline_samples].mean(axis=0)
    return np.asarray([data[round((start - START_MS) * spec.fs / 1000):
                            round((end - START_MS) * spec.fs / 1000), channel].mean()
                       for channel in indices for start, end in ((70, 150), (150, 250), (250, 400))])


def score_gabor(epochs: list[tuple[np.ndarray, int, str]], settings: EEGSettings,
                spec: EpochSpec) -> dict:
    """Fit only early trials; score the later held-out presentations once."""
    accepted = []
    for epoch, label, role in epochs:
        try:
            accepted.append((gabor_features(epoch, settings, spec), label, role))
        except ValueError:
            pass
    train = [(x, y) for x, y, role in accepted if role == "training"]
    test = [(x, y) for x, y, role in accepted if role == "held_out"]
    result = {"protocol": "Gabor versus uniform gray, 80 training and 40 held-out trials; posterior ERP 70–400 ms",
              "presented": TRIALS, "usable_training": len(train), "usable_held_out": len(test),
              "electrodes": ["PO7", "Oz", "PO8"], "status": "inconclusive"}
    if len(train) < 40 or len(test) < 20 or len({y for _, y in train}) < 2 or len({y for _, y in test}) < 2:
        result["reason"] = "Too few usable epochs of both classes"
        return result
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
    model.fit(np.stack([x for x, _ in train]), [y for _, y in train])
    truth = np.asarray([y for _, y in test])
    probabilities = model.predict_proba(np.stack([x for x, _ in test]))[:, 1]
    result.update(status="scored", accuracy=float(np.mean((probabilities >= .5) == truth)),
                  balanced_accuracy=float(balanced_accuracy_score(truth, probabilities >= .5)),
                  auc=float(roc_auc_score(truth, probabilities)),
                  brier=float(np.mean((probabilities - truth) ** 2)))
    correct = int(np.sum((probabilities >= .5) == truth))
    result["chance_p_one_sided"] = float(binomtest(correct, len(truth), .5, alternative="greater").pvalue)
    result["visual_response_detected"] = bool(result["chance_p_one_sided"] < .05
                                              and result["brier"] < .25)
    return result
