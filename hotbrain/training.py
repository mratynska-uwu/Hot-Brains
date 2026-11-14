"""Trial allocation and prequential (test-before-train) classifiers.

One model is trained per (target, classifier) pair, e.g. valence/logreg,
valence/forest and arousal/svm. Each target has its own feature vector (its
configuration profile picks electrodes, bands and feature families).
Images whose label for a target is None (excluded middle) are skipped by that
target's classifiers; regression models learn the continuous value instead.
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
    correct: bool | None  # None when the image has no label for this target (excluded middle)
    truth: int | None = None
    value: float | None = None         # regression models: predicted value in [-1, 1]
    true_value: float | None = None    # regression models: target value in [-1, 1]


class OnlineTrainer:
    """Refit one classifier after each held-out prediction."""

    def __init__(self, classifier: str, n_features: int):
        self.classifier = classifier
        self.n_features = n_features
        self.regression = classifiers.is_regression(classifier)
        self.x: list[np.ndarray] = []
        self.y: list[int] = []            # labels (classifiers) or sign classes (regression, may be -1 = none)
        self.values: list[float] = []     # regression targets
        self.model = None
        self.predictions: list[Prediction] = []

    def _check(self, features) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (self.n_features,) or not np.isfinite(features).all():
            raise ValueError("Invalid EEG feature vector")
        return features

    def predict(self, features: np.ndarray, label: int | None, value: float | None = None) -> Prediction | None:
        """Test step only: predict with the current model (None before the first fit or without a target)."""
        if self.model is None:
            return None
        if self.regression and value is None:
            return None
        if not self.regression and label is None:
            return None
        x = features[None, :]
        probability = float(self.model.predict_proba(x)[0, 1])
        predicted = int(probability >= .5)
        predicted_value = classifiers.predict_value(self.model, features) if self.regression else None
        correct = None if label is None else predicted == label
        return Prediction(predicted, probability, correct, label, predicted_value, value)

    def learn(self, features: np.ndarray, label: int | None, value: float | None = None, refit: bool = True):
        """Train step: append the example and refit when enough data exists."""
        if self.regression:
            if value is None:
                return
            self.values.append(float(value))
            self.y.append(-1 if label is None else int(label))
        else:
            if label is None:
                return
            self.y.append(int(label))
        self.x.append(features)
        if refit:
            self.refit()

    def refit(self):
        if self.regression:
            if len(self.x) >= MIN_TRIALS and np.ptp(self.values) > 0:
                model = classifiers.build(self.classifier)
                model.fit(np.vstack(self.x), np.asarray(self.values))
                self.model = model
            return
        need = classifiers.min_per_class(self.classifier)
        if len(self.x) >= MIN_TRIALS and len(set(self.y)) == 2 and min(self.y.count(0), self.y.count(1)) >= need:
            model = classifiers.build(self.classifier)
            model.fit(np.vstack(self.x), np.asarray(self.y))
            self.model = model

    def add(self, features: np.ndarray, label: int | None, value: float | None = None) -> Prediction | None:
        features = self._check(features)
        prediction = self.predict(features, label, value)
        if prediction is not None:
            self.predictions.append(prediction)
        self.learn(features, label, value)
        return prediction

    def add_batch(self, rows: list[tuple[np.ndarray, int | None, float | None]]) -> list[Prediction | None]:
        """Predict every row with the current model first, then learn them all and refit once.

        Used for calibration blocks: windows of one block must not predict each other.
        """
        checked = [(self._check(f), label, value) for f, label, value in rows]
        predictions = []
        for features, label, value in checked:
            prediction = self.predict(features, label, value)
            if prediction is not None:
                self.predictions.append(prediction)
            predictions.append(prediction)
        for features, label, value in checked:
            self.learn(features, label, value, refit=False)
        self.refit()
        return predictions

    def _scored(self, last: int | None = ROLLING) -> list[Prediction]:
        scored = [p for p in self.predictions if p.correct is not None]
        return scored[-last:] if last else scored

    @property
    def rolling_accuracy(self) -> float | None:
        last = self._scored()
        return 100 * sum(p.correct for p in last) / len(last) if last else None

    def rolling_balanced_accuracy(self, last: int | None = ROLLING) -> float | None:
        """Mean recall of the classes present among the last predictions (None if fewer than 2 classes)."""
        rows = self._scored(last)
        recalls = []
        for cls in (0, 1):
            of_class = [p for p in rows if p.truth == cls]
            if of_class:
                recalls.append(sum(p.correct for p in of_class) / len(of_class))
        return 100 * float(np.mean(recalls)) if len(recalls) == 2 else None


class MultiTrainer:
    """All (target, classifier) models trained side by side on the same epochs."""

    def __init__(self, targets: list[str], classifier_keys: list[str], n_features: int | dict[str, int],
                 selected_pairs: list[tuple[str, str]] | None = None):
        self.targets = list(targets)
        self.classifier_keys = list(classifier_keys)
        pairs = ([(t, c) for t in self.targets for c in self.classifier_keys]
                 if selected_pairs is None else list(selected_pairs))
        if len(pairs) != len(set(pairs)) or any(t not in self.targets or c not in self.classifier_keys
                                                for t, c in pairs):
            raise ValueError("Selected target/model pairs must be unique and configured")
        sizes = n_features if isinstance(n_features, dict) else {t: n_features for t in self.targets}
        self.models: dict[tuple[str, str], OnlineTrainer] = {
            (t, c): OnlineTrainer(c, sizes[t]) for t, c in pairs}

    @staticmethod
    def _for(features, target: str):
        return features[target] if isinstance(features, dict) else features

    def add(self, features: np.ndarray | dict[str, np.ndarray], labels: dict[str, int | None],
            values: dict[str, float] | None = None) -> dict[tuple[str, str], Prediction]:
        """Test-before-train for every model. ``features`` may be one shared vector or one per target."""
        out = {}
        for (target, clf), trainer in self.models.items():
            prediction = trainer.add(self._for(features, target), labels.get(target),
                                     None if values is None else values.get(target))
            if prediction is not None:
                out[(target, clf)] = prediction
        return out

    def add_batch(self, rows: list[tuple[dict[str, np.ndarray], dict[str, int | None], dict[str, float] | None]]
                  ) -> dict[tuple[str, str], list[Prediction]]:
        """Block-wise test-before-train: all rows are predicted before any of them is learned."""
        out = {}
        for (target, clf), trainer in self.models.items():
            predictions = trainer.add_batch([(self._for(f, target), labels.get(target),
                                              None if values is None else values.get(target))
                                             for f, labels, values in rows])
            kept = [p for p in predictions if p is not None]
            if kept:
                out[(target, clf)] = kept
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
                     "kind": classifiers.CLASSIFIERS[clf].kind,
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
