"""Trial allocation and prequential (test-before-train) classifiers.

One model is trained per (target, classifier) pair, e.g. valence/logreg,
valence/forest and arousal/svm, all on the same EEG feature vectors.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from . import classifiers
from .dataset import StimulusImage

MIN_TRIALS = 10
ROLLING = 20


def balanced_unique_capacity(groups: dict[str, list[StimulusImage]]) -> int:
    """Largest balanced finite run that never repeats an image."""
    return len(groups) * min((len(items) for items in groups.values()), default=0)


class TrialPlan:
    """Balanced across groups; each group cycles its images before repeating."""

    def __init__(self, groups: dict[str, list[StimulusImage]], total: int | None,
                 rng: random.Random | None = None, allow_repeats: bool = True):
        if not groups or not all(groups.values()):
            raise ValueError("Every trial group needs at least one image.")
        names = list(groups)
        if total is not None and (total < len(names) or total % len(names)):
            raise ValueError(f"The finite trial count must be a positive multiple of {len(names)} "
                             "(the number of trial groups) for exact balance.")
        if not allow_repeats and (total is None or total > balanced_unique_capacity(groups)):
            raise ValueError("A no-repeat run needs a finite trial count within the available images per group.")
        self.rng = rng or random.Random()
        self.groups = groups
        self.names = names
        self.total = total
        self.order = (names * (total // len(names))) if total else []
        self.rng.shuffle(self.order)
        self.queues: dict[str, list[StimulusImage]] = {k: [] for k in groups}
        self.previous: dict[str, StimulusImage | None] = {k: None for k in groups}
        self.index = 0

    def next(self) -> StimulusImage | None:
        if self.total is not None and self.index >= self.total:
            return None
        group = self.order[self.index] if self.total else self.rng.choice(self.names)
        self.index += 1
        queue = self.queues[group]
        if not queue:
            queue.extend(self.groups[group])
            self.rng.shuffle(queue)
            if len(queue) > 1 and queue[-1] == self.previous[group]:
                queue[0], queue[-1] = queue[-1], queue[0]
        item = queue.pop()
        self.previous[group] = item
        return item


@dataclass
class Prediction:
    label: int
    probability: float  # P(label 1); sigmoid-calibrated for SVMs, native for others
    correct: bool


class OnlineTrainer:
    """Refit one classifier after each held-out prediction."""

    def __init__(self, classifier: str, n_features: int):
        self.classifier = classifier
        self.n_features = n_features
        self.x: list[np.ndarray] = []
        self.y: list[int] = []
        self.model = None
        self.predictions: list[Prediction] = []

    def add(self, features: np.ndarray, label: int) -> Prediction | None:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.n_features,) or not np.isfinite(features).all():
            raise ValueError("Invalid EEG feature vector")
        prediction = None
        if self.model is not None:
            probability = float(self.model.predict_proba(features[None, :])[0, 1])
            predicted = int(probability >= .5)
            prediction = Prediction(predicted, probability, predicted == label)
            self.predictions.append(prediction)
        self.x.append(features)
        self.y.append(int(label))
        minimum_per_class = 3 if classifiers.needs_calibration(self.classifier) else 2
        if len(self.x) >= MIN_TRIALS and len(set(self.y)) == 2 \
                and min(self.y.count(0), self.y.count(1)) >= minimum_per_class:
            model = classifiers.build(self.classifier)
            model.fit(np.vstack(self.x), np.asarray(self.y))
            self.model = model
        return prediction

    @property
    def rolling_accuracy(self) -> float | None:
        last = self.predictions[-ROLLING:]
        return 100 * sum(p.correct for p in last) / len(last) if last else None


class MultiTrainer:
    """All (target, classifier) models trained side by side on the same epochs."""

    def __init__(self, targets: list[str], classifier_keys: list[str], n_features: int,
                 selected_pairs: list[tuple[str, str]] | None = None):
        self.targets = list(targets)
        self.classifier_keys = list(classifier_keys)
        pairs = ([(t, c) for t in self.targets for c in self.classifier_keys]
                 if selected_pairs is None else list(selected_pairs))
        if len(pairs) != len(set(pairs)) or any(t not in self.targets or c not in self.classifier_keys
                                                for t, c in pairs):
            raise ValueError("Selected target/model pairs must be unique and configured")
        self.models: dict[tuple[str, str], OnlineTrainer] = {
            (t, c): OnlineTrainer(c, n_features) for t, c in pairs}

    def add(self, features: np.ndarray, labels: dict[str, int]) -> dict[tuple[str, str], Prediction]:
        out = {}
        for (target, clf), trainer in self.models.items():
            prediction = trainer.add(features, labels[target])
            if prediction is not None:
                out[(target, clf)] = prediction
        return out

    @property
    def any_fitted(self) -> bool:
        return any(t.model is not None for t in self.models.values())

    def save(self, stem: Path, metadata: dict,
             selected_models: set[tuple[str, str]] | None = None) -> list[Path]:
        if not self.any_fitted:
            raise ValueError(f"At least {MIN_TRIALS} usable EEG trials with both labels are needed to save a fitted model.")
        selected = {key for key, trainer in self.models.items() if trainer.model is not None} \
            if selected_models is None else set(selected_models)
        if not selected:
            raise ValueError("Select at least one fitted model to save.")
        unknown = selected - self.models.keys()
        if unknown:
            raise ValueError(f"Unknown model selection: {sorted(unknown)}")
        unfitted = {key for key in selected if self.models[key].model is None}
        if unfitted:
            raise ValueError(f"Selected models are not fitted: {sorted(unfitted)}")
        written = []
        summary = {}
        for (target, clf), trainer in self.models.items():
            entry = {"classifier": classifiers.CLASSIFIERS[clf].label,
                     "probability_method": "3-fold cross-validated sigmoid calibration" if classifiers.needs_calibration(clf) else "native",
                     "n_eeg_trials": len(trainer.x),
                     "n_predictions": len(trainer.predictions), "rolling_accuracy": trainer.rolling_accuracy,
                     "selected_for_save": (target, clf) in selected,
                     "model_file": None}
            if (target, clf) in selected:
                path = stem.with_name(f"{stem.name}.{target}.{clf}.joblib")
                joblib.dump(trainer.model, path)
                entry["model_file"] = path.name
                written.append(path)
            summary[f"{target}/{clf}"] = entry
        metadata_path = stem.with_suffix(".json")
        record = {**metadata, "schema": 4, "saved_models": [f"{t}/{c}" for t, c in sorted(selected)],
                  "models": summary}
        metadata_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return [metadata_path, *written]


def predict_saved(model_path: Path, features: np.ndarray) -> float:
    """P(label 1) from a saved model, including calibrated SVMs.

    joblib uses pickle: only load files you created.
    """
    model = joblib.load(model_path)
    features = np.asarray(features, dtype=np.float64)[None, :]
    if not hasattr(model, "predict_proba"):
        raise ValueError("This saved model has no probability output; retrain it with the current app.")
    return float(model.predict_proba(features)[0, 1])
