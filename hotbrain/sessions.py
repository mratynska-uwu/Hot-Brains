"""Training/validation split, validation scoring and live valence-arousal decoding.

Session types:
- training:   split runs interleave training and held-out images in one session;
              only training epochs fit the models. Prequential runs use unique images.
- validation: optional extra session for legacy models not selected on these
              held-out images; frozen models score each usable epoch.
- test:       test images with target valence/arousal; the participant tries to match
              them while the frozen models decode valence/arousal continuously.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .classifiers import probability_class1
from .dataset import Dataset, StimulusImage

SESSION_TRAINING = "training"
SESSION_VALIDATION = "validation"
SESSION_TEST = "test"


def split_dataset(dataset: Dataset, validation_fraction: float, seed: int
                  ) -> tuple[dict[str, list[StimulusImage]], dict[str, list[StimulusImage]]]:
    """Stratified per folder, reproducible: the same seed always gives the same split."""
    train, validation = {}, {}
    for group, items in dataset.groups.items():
        shuffled = sorted(items, key=lambda i: i.path.name)
        random.Random(f"{seed}:{group}").shuffle(shuffled)
        n_val = round(len(shuffled) * validation_fraction)
        if validation_fraction > 0 and len(shuffled) > 1:
            n_val = min(max(n_val, 1), len(shuffled) - 1)
        validation[group] = sorted(shuffled[:n_val], key=lambda i: i.path.name)
        train[group] = sorted(shuffled[n_val:], key=lambda i: i.path.name)
    return train, validation


class OncePlan:
    """Shows every image exactly once in random order (validation and test sessions)."""

    def __init__(self, groups: dict[str, list[StimulusImage]], rng: random.Random | None = None):
        self.items = [i for items in groups.values() for i in items]
        if not self.items:
            raise ValueError("No images for this session.")
        (rng or random.Random()).shuffle(self.items)
        self.total = len(self.items)
        self.index = 0

    def next(self) -> StimulusImage | None:
        if self.index >= self.total:
            return None
        self.index += 1
        return self.items[self.index - 1]


def affect_point(models: dict[tuple[str, str], object], classifier: str, features: np.ndarray,
                 targets: tuple[str, str]) -> tuple[float | None, float | None]:
    """Decoded (x, y) in [-1, 1]: 2 * P(label 1) - 1 for the two displayed targets."""
    out = []
    for target in targets:
        model = models.get((target, classifier))
        out.append(None if model is None else 2 * probability_class1(model, features) - 1)
    return out[0], out[1]


@dataclass
class ValidationScores:
    """Frozen-model predictions on held-out images."""
    models: dict[tuple[str, str], object]
    records: dict[tuple[str, str], list[tuple[int, int, float]]] = field(default_factory=dict)

    def add(self, features: np.ndarray, labels: dict[str, int]) -> dict[tuple[str, str], tuple[int, float]]:
        out = {}
        for (target, clf), model in self.models.items():
            p = probability_class1(model, features)
            predicted = int(model.predict(np.asarray(features, dtype=np.float64)[None, :])[0])
            self.records.setdefault((target, clf), []).append((labels[target], predicted, p))
            out[(target, clf)] = (predicted, p)
        return out

    def accuracy(self, key: tuple[str, str]) -> tuple[float | None, float | None, int]:
        """(accuracy %, balanced accuracy %, n). Balanced = mean recall of the classes present."""
        rows = self.records.get(key, [])
        if not rows:
            return None, None, 0
        truth = np.array([r[0] for r in rows])
        predicted = np.array([r[1] for r in rows])
        accuracy = 100 * float(np.mean(truth == predicted))
        recalls = [np.mean(predicted[truth == c] == c) for c in (0, 1) if np.any(truth == c)]
        return accuracy, 100 * float(np.mean(recalls)), len(rows)

    def majority_baseline(self, key: tuple[str, str]) -> float | None:
        rows = self.records.get(key, [])
        if not rows:
            return None
        ones = sum(r[0] for r in rows)
        return 100 * max(ones, len(rows) - ones) / len(rows)


class Smoother:
    """Exponential smoothing of the live decoded point."""

    def __init__(self, weight: float):
        self.weight = weight
        self.value: tuple[float, float] | None = None

    def reset(self):
        self.value = None

    def update(self, point: tuple[float | None, float | None]) -> tuple[float | None, float | None]:
        if point[0] is None or point[1] is None:
            return point
        if self.value is None:
            self.value = (float(point[0]), float(point[1]))
        else:
            w = self.weight
            self.value = (w * point[0] + (1 - w) * self.value[0], w * point[1] + (1 - w) * self.value[1])
        return self.value


def load_models(manifest_path: Path) -> tuple[dict[tuple[str, str], object], dict]:
    """Load the fitted models listed in a saved training manifest (joblib = pickle: own files only)."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    models = {}
    for key, entry in manifest.get("models", {}).items():
        if entry.get("model_file"):
            target, clf = key.split("/", 1)
            models[(target, clf)] = joblib.load(Path(manifest_path).with_name(entry["model_file"]))
    if not models:
        raise ValueError(f"{Path(manifest_path).name} lists no fitted models")
    return models, manifest
