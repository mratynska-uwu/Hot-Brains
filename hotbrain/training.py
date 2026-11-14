"""Trial allocation and prequential (test-before-train) classifier."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import CHANNELS, FEATURE_NAMES, FS

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}


def image_files(root: Path) -> dict[str, list[Path]]:
    groups = {}
    for label in ("happy", "sad"):
        folder = root / "training_dataset" / f"{label}_images"
        groups[label] = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if folder.exists() else []
    return groups


class TrialPlan:
    def __init__(self, images: dict[str, list[Path]], total: int | None, rng: random.Random | None = None):
        if not all(images.values()):
            raise ValueError("Put at least one image in each happy_images and sad_images folder.")
        if total is not None and (total < 2 or total % 2):
            raise ValueError("The finite trial count must be positive and even for exact balance.")
        self.rng = rng or random.Random()
        self.images = images
        self.total = total
        self.labels = (["happy", "sad"] * (total // 2)) if total else []
        self.rng.shuffle(self.labels)
        self.queues: dict[str, list[Path]] = {k: [] for k in images}
        self.previous: dict[str, Path | None] = {k: None for k in images}
        self.index = 0

    def next(self) -> tuple[str, Path] | None:
        if self.total is not None and self.index >= self.total:
            return None
        label = self.labels[self.index] if self.total else self.rng.choice(("happy", "sad"))
        self.index += 1
        queue = self.queues[label]
        if not queue:
            queue.extend(self.images[label])
            self.rng.shuffle(queue)
            if len(queue) > 1 and queue[-1] == self.previous[label]:
                queue[0], queue[-1] = queue[-1], queue[0]
        path = queue.pop()
        self.previous[label] = path
        return label, path


@dataclass
class Prediction:
    label: str
    probability_happy: float
    correct: bool


class OnlineTrainer:
    """Refit fast L2 logistic regression after each held-out prediction."""

    def __init__(self):
        self.x: list[np.ndarray] = []
        self.y: list[int] = []
        self.model = None
        self.predictions: list[Prediction] = []

    def add(self, features: np.ndarray, label: str) -> Prediction | None:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise ValueError("Invalid EEG feature vector")
        target = int(label == "happy")
        prediction = None
        if self.model is not None:
            p = float(self.model.predict_proba(features[None, :])[0, 1])
            prediction = Prediction("happy" if p >= 0.5 else "sad", p, (p >= 0.5) == bool(target))
            self.predictions.append(prediction)
        self.x.append(features)
        self.y.append(target)
        if len(self.x) >= 10 and len(set(self.y)) == 2:
            self.model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
            self.model.fit(np.vstack(self.x), np.asarray(self.y))
        return prediction

    @property
    def rolling_accuracy(self) -> float | None:
        last = self.predictions[-20:]
        return 100 * sum(p.correct for p in last) / len(last) if last else None

    def save(self, stem: Path, metadata: dict) -> tuple[Path, Path]:
        if self.model is None:
            raise ValueError("At least 10 usable EEG trials with both labels are needed to save a fitted model.")
        scaler, classifier = self.model.steps[0][1], self.model.steps[1][1]
        weights_path = stem.with_suffix(".npz")
        metadata_path = stem.with_suffix(".json")
        np.savez_compressed(weights_path, mean=scaler.mean_, scale=scaler.scale_,
                            coefficients=classifier.coef_[0], intercept=classifier.intercept_[0])
        record = {**metadata, "schema": 1, "fs": FS, "channels": CHANNELS,
                  "feature_names": FEATURE_NAMES, "n_eeg_trials": len(self.x),
                  "n_predictions": len(self.predictions), "rolling_accuracy": self.rolling_accuracy,
                  "model": "StandardScaler + L2 logistic regression", "class_1": "happy"}
        metadata_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return weights_path, metadata_path


def predict_saved(stem: Path, features: np.ndarray) -> float:
    """Safe inference helper: P(happy) using numeric weights, without pickle."""
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as data:
        z = (np.asarray(features) - data["mean"]) / data["scale"]
        logit = float(z @ data["coefficients"] + data["intercept"])
    return float(1 / (1 + np.exp(-np.clip(logit, -50, 50))))
