"""Trial allocation, test-before-train classifiers and held-out checkpoints.

One model is trained per (target, classifier) pair, e.g. valence/logreg,
valence/forest and arousal/svm, all on the same EEG feature vectors.
"""
from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

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


class InterleavedPlan:
    """One session with balanced training and held-out image presentations."""

    def __init__(self, training: dict[str, list[StimulusImage]],
                 validation: dict[str, list[StimulusImage]], total: int | None,
                 validation_fraction: float, rng: random.Random | None = None):
        if not training or not 0 < validation_fraction < 1 or set(training) != set(validation):
            raise ValueError("Training and validation need the same nonempty trial groups")
        if any(not training[group] or not validation[group] for group in training):
            raise ValueError("Every class needs training and validation images")
        if {i.path for group in training.values() for i in group} & {
                i.path for group in validation.values() for i in group}:
            raise ValueError("Training and validation image identities overlap")
        n_groups = len(training)
        if total is not None:
            if total < 2 * n_groups or total % n_groups:
                raise ValueError(f"Split training needs at least {2 * n_groups} presentations "
                                 f"and a multiple of {n_groups}")
            validation_total = n_groups * round(total * validation_fraction / n_groups)
            validation_total = min(max(validation_total, n_groups), total - n_groups)
            training_total = total - validation_total
        else:
            validation_total = training_total = None
        self.rng = rng or random.Random()
        self.validation_fraction = validation_fraction
        self.training = TrialPlan(training, training_total, rng=self.rng)
        self.validation = TrialPlan(validation, validation_total, rng=self.rng)
        self.total = total
        self.index = 0
        self.last_role: str | None = None
        self.roles = (["training"] * training_total + ["validation"] * validation_total
                      if total is not None else [])
        self.rng.shuffle(self.roles)

    def next(self) -> StimulusImage | None:
        if self.total is not None and self.index >= self.total:
            return None
        role = (self.roles[self.index] if self.total is not None else
                "validation" if self.rng.random() < self.validation_fraction else "training")
        self.index += 1
        self.last_role = role
        return (self.validation if role == "validation" else self.training).next()


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
                 selected_pairs: list[tuple[str, str]] | None = None,
                 checkpoint_every: int | None = None):
        self.targets = list(targets)
        self.classifier_keys = list(classifier_keys)
        pairs = ([(t, c) for t in self.targets for c in self.classifier_keys]
                 if selected_pairs is None else list(selected_pairs))
        if len(pairs) != len(set(pairs)) or any(t not in self.targets or c not in self.classifier_keys
                                                for t, c in pairs):
            raise ValueError("Selected target/model pairs must be unique and configured")
        self.models: dict[tuple[str, str], OnlineTrainer] = {
            (t, c): OnlineTrainer(c, n_features) for t, c in pairs}
        self.checkpoint_every = checkpoint_every
        self.training_trials = 0
        self.checkpoints: dict[tuple[str, str], list[tuple[int, object]]] = {key: [] for key in pairs}

    def add(self, features: np.ndarray, labels: dict[str, int]) -> dict[tuple[str, str], Prediction]:
        out = {}
        for (target, clf), trainer in self.models.items():
            prediction = trainer.add(features, labels[target])
            if prediction is not None:
                out[(target, clf)] = prediction
        self.training_trials += 1
        if self.checkpoint_every and self.training_trials % self.checkpoint_every == 0:
            self.capture_checkpoint()
        return out

    def predict(self, features: np.ndarray, labels: dict[str, int]) -> dict[tuple[str, str], Prediction]:
        """Score held-out EEG without adding it to any model's training history."""
        x = np.asarray(features, dtype=np.float64)[None, :]
        out = {}
        for key, trainer in self.models.items():
            if trainer.model is None:
                continue
            p = float(trainer.model.predict_proba(x)[0, 1])
            label = int(p >= .5)
            out[key] = Prediction(label, p, label == labels[key[0]])
        return out

    def capture_checkpoint(self) -> None:
        for key, trainer in self.models.items():
            if trainer.model is None:
                continue
            snapshots = self.checkpoints[key]
            if not snapshots or snapshots[-1][0] != self.training_trials:
                snapshots.append((self.training_trials, deepcopy(trainer.model)))

    @property
    def any_fitted(self) -> bool:
        return any(t.model is not None for t in self.models.values())

    def save(self, stem: Path, metadata: dict,
             selected_models: set[tuple[str, str]] | None = None,
             model_overrides: dict[tuple[str, str], dict] | None = None) -> list[Path]:
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
        if model_overrides is not None and selected - model_overrides.keys():
            raise ValueError("Selected models have no validated checkpoint")
        written = []
        summary = {}
        for (target, clf), trainer in self.models.items():
            entry = {"classifier": classifiers.CLASSIFIERS[clf].label,
                     "probability_method": "3-fold cross-validated sigmoid calibration" if classifiers.needs_calibration(clf) else "native",
                     "n_eeg_trials": (model_overrides[(target, clf)]["training_trials"]
                                      if model_overrides is not None and (target, clf) in model_overrides
                                      else len(trainer.x)),
                     "n_predictions": len(trainer.predictions), "rolling_accuracy": trainer.rolling_accuracy,
                     "selected_for_save": (target, clf) in selected,
                     "model_file": None}
            if model_overrides is not None and (target, clf) in model_overrides:
                entry["checkpoint_selection"] = {field: value for field, value in model_overrides[(target, clf)].items()
                                                 if field != "model"}
            if (target, clf) in selected:
                path = stem.with_name(f"{stem.name}.{target}.{clf}.joblib")
                joblib.dump(model_overrides[(target, clf)]["model"] if model_overrides is not None
                            else trainer.model, path)
                entry["model_file"] = path.name
                written.append(path)
            summary[f"{target}/{clf}"] = entry
        metadata_path = stem.with_suffix(".json")
        record = {**metadata, "schema": 5, "saved_models": [f"{t}/{c}" for t, c in sorted(selected)],
                  "models": summary}
        metadata_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return [metadata_path, *written]


def select_best_checkpoints(
        checkpoints: dict[tuple[str, str], list[tuple[int, object]]],
        validation_features: list[np.ndarray], validation_labels: list[dict[str, int]],
        minimum_validation: int = 10) -> dict[tuple[str, str], dict]:
    """Pick each pair's lowest Brier-loss checkpoint on one fixed held-out set.

    Every candidate sees exactly the same validation epochs, including epochs
    recorded later than that checkpoint. No validation epoch enters fitting.
    """
    if len(validation_features) != len(validation_labels):
        raise ValueError("Validation features and labels have different lengths")
    if len(validation_features) < minimum_validation:
        return {}
    x = np.vstack(validation_features)
    selected = {}
    for key, snapshots in checkpoints.items():
        truth = np.asarray([labels[key[0]] for labels in validation_labels], dtype=np.int8)
        if min(np.count_nonzero(truth == 0), np.count_nonzero(truth == 1)) < 3:
            continue
        best = None
        for n_training, model in snapshots:
            probabilities = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
            if (probabilities.shape != truth.shape or not np.isfinite(probabilities).all()
                    or np.any((probabilities < 0) | (probabilities > 1))):
                raise ValueError(f"Invalid validation probabilities for {key}")
            brier = float(np.mean((probabilities - truth) ** 2))
            predicted = probabilities >= .5
            recalls = [float(np.mean(predicted[truth == cls] == cls)) for cls in (0, 1)]
            candidate = {"model": model, "training_trials": n_training,
                         "validation_brier": brier,
                         "validation_accuracy": float(np.mean(predicted == truth)),
                         "validation_balanced_accuracy": float(np.mean(recalls)),
                         "validation_auc": float(roc_auc_score(truth, probabilities)),
                         "coinflip_brier": .25, "beats_coinflip_brier": brier < .25,
                         "validation_n": len(truth), "candidate_checkpoints": len(snapshots)}
            if best is None or brier < best["validation_brier"]:
                best = candidate
        if best is not None:
            selected[key] = best
    return selected


def predict_saved(model_path: Path, features: np.ndarray) -> float:
    """P(label 1) from a saved model, including calibrated SVMs.

    joblib uses pickle: only load files you created.
    """
    model = joblib.load(model_path)
    features = np.asarray(features, dtype=np.float64)[None, :]
    if not hasattr(model, "predict_proba"):
        raise ValueError("This saved model has no probability output; retrain it with the current app.")
    return float(model.predict_proba(features)[0, 1])
