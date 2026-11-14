"""Registry of classifiers that can be trained on EEG feature vectors.

Add a new one with ``register_classifier("key", "Label", factory)`` where the
factory returns an unfitted scikit-learn estimator. Every estimator is wrapped
in a StandardScaler pipeline and must support ``predict``; ``predict_proba``
is used when available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ClassifierSpec:
    key: str
    label: str
    factory: Callable[[], BaseEstimator]


CLASSIFIERS: dict[str, ClassifierSpec] = {}


def register_classifier(key: str, label: str, factory: Callable[[], BaseEstimator]) -> None:
    CLASSIFIERS[key] = ClassifierSpec(key, label, factory)


def build(key: str) -> Pipeline:
    if key not in CLASSIFIERS:
        raise KeyError(f"Unknown classifier {key!r}; available: {sorted(CLASSIFIERS)}")
    return make_pipeline(StandardScaler(), CLASSIFIERS[key].factory())


def probability_class1(model: Pipeline, features) -> float:
    """P(label 1) for one feature vector.

    Uses predict_proba when the model has it; otherwise squashes the decision
    function with a logistic curve (an uncalibrated confidence, e.g. for SVMs).
    """
    x = np.asarray(features, dtype=np.float64).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x)[0, 1])
    if hasattr(model, "decision_function"):
        margin = float(np.ravel(model.decision_function(x))[0])
        return float(1 / (1 + np.exp(-np.clip(margin, -50, 50))))
    return float(model.predict(x)[0])


def coefficients(model: Pipeline | None):
    """Standardized weights for linear models, else None."""
    if model is None:
        return None
    coef = getattr(model.steps[-1][1], "coef_", None)
    return None if coef is None else coef[0]


register_classifier("logreg", "Logistic regression (L2)",
                    lambda: LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
register_classifier("svm", "SVM (RBF kernel)",
                    lambda: SVC(kernel="rbf", C=1.0, gamma="scale"))
register_classifier("svm_linear", "SVM (linear kernel)",
                    lambda: SVC(kernel="linear", C=1.0))
