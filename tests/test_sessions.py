import json
import random
from pathlib import Path

import joblib
import numpy as np
import pytest

from hotbrain import classifiers
from hotbrain.config import CONFIG, AppConfig
from hotbrain.dataset import load_dataset
from hotbrain.sessions import (OncePlan, Smoother, ValidationScores, affect_point, load_models,
                               split_dataset)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(AppConfig.from_dict(CONFIG, ROOT))


def test_split_is_stratified_reproducible_and_disjoint(dataset):
    train, validation = split_dataset(dataset, 0.2, 42)
    assert {g: len(v) for g, v in validation.items()} == {"training/joy": 29, "training/sadness": 15}
    assert {g: len(v) for g, v in train.items()} == {"training/joy": 114, "training/sadness": 59}
    for group in dataset.groups:
        train_names = {i.path.name for i in train[group]}
        val_names = {i.path.name for i in validation[group]}
        assert not train_names & val_names
        assert train_names | val_names == {i.path.name for i in dataset.groups[group]}
    again = split_dataset(dataset, 0.2, 42)[1]
    assert {g: [i.path.name for i in v] for g, v in again.items()} == \
        {g: [i.path.name for i in v] for g, v in validation.items()}
    other = split_dataset(dataset, 0.2, 7)[1]
    assert [i.path.name for i in other["training/joy"]] != [i.path.name for i in validation["training/joy"]]


def test_once_plan_shows_each_image_once(dataset):
    _, validation = split_dataset(dataset, 0.2, 42)
    plan = OncePlan(validation, random.Random(1))
    shown = [plan.next() for _ in range(plan.total)]
    assert plan.next() is None
    assert len({i.path for i in shown}) == plan.total == 44


def test_normalized_targets_span_minus_one_to_one(dataset):
    values = [dataset.normalized("valence", i.values["valence"]) for i in dataset.images]
    assert min(values) == -1.0 or max(values) == 1.0
    assert all(-1 <= v <= 1 for v in values)
    joy = dataset.groups["training/joy"][0]
    assert dataset.normalized("valence", joy.values["valence"]) > 0


class _Fixed:
    """Stand-in model with a fixed decision."""

    def __init__(self, label: int, p: float):
        self.label, self.p = label, p

    def predict(self, x):
        return np.array([self.label])

    def predict_proba(self, x):
        return np.array([[1 - self.p, self.p]])


def test_validation_scores_balanced_accuracy_and_baseline():
    scores = ValidationScores({("valence", "logreg"): _Fixed(1, .8)})
    for label in (1, 1, 1, 0):
        scores.add(np.zeros(3), {"valence": label})
    accuracy, balanced, n = scores.accuracy(("valence", "logreg"))
    assert (accuracy, balanced, n) == (75.0, 50.0, 4)   # always "positive": high accuracy, chance balanced
    assert scores.majority_baseline(("valence", "logreg")) == 75.0


def test_affect_point_and_smoothing():
    models = {("valence", "m"): _Fixed(1, .9), ("arousal", "m"): _Fixed(0, .25)}
    point = affect_point(models, "m", np.zeros(3), ("valence", "arousal"))
    assert np.allclose(point, (0.8, -0.5))
    assert affect_point(models, "missing", np.zeros(3), ("valence", "arousal")) == (None, None)
    smoother = Smoother(0.5)
    assert smoother.update((1.0, 1.0)) == (1.0, 1.0)
    assert smoother.update((0.0, -1.0)) == (0.5, 0.0)


def test_svm_without_probabilities_still_gives_a_confidence():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4))
    y = (x[:, 0] > 0).astype(int)
    model = classifiers.build("svm").fit(x, y)
    high = classifiers.probability_class1(model, np.array([3.0, 0, 0, 0]))
    low = classifiers.probability_class1(model, np.array([-3.0, 0, 0, 0]))
    assert 0 <= low < .5 < high <= 1


def test_load_models_from_manifest(tmp_path):
    model = classifiers.build("logreg").fit(np.array([[0.0], [1.0], [0.1], [0.9]]), [0, 1, 0, 1])
    joblib.dump(model, tmp_path / "run.valence.logreg.joblib")
    manifest = {"feature_names": ["f"], "image_duration_s": 1.0,
                "models": {"valence/logreg": {"model_file": "run.valence.logreg.joblib"},
                           "arousal/logreg": {"model_file": None}}}
    (tmp_path / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    models, loaded = load_models(tmp_path / "run.json")
    assert list(models) == [("valence", "logreg")]
    assert loaded["image_duration_s"] == 1.0
