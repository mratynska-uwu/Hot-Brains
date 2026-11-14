"""Registry of classifiers that can be trained on EEG feature vectors.

Add a new one with ``register_classifier("key", "Label", factory)`` where the
factory returns an unfitted scikit-learn estimator. Every estimator is wrapped
in a StandardScaler pipeline. Estimators without ``predict_proba`` are
calibrated with cross-validated sigmoid calibration around the whole pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


@dataclass(frozen=True)
class ClassifierSpec:
    key: str
    label: str
    factory: Callable[[], BaseEstimator]


CLASSIFIERS: dict[str, ClassifierSpec] = {}


def register_classifier(key: str, label: str, factory: Callable[[], BaseEstimator]) -> None:
    CLASSIFIERS[key] = ClassifierSpec(key, label, factory)


def needs_calibration(key: str) -> bool:
    return not hasattr(CLASSIFIERS[key].factory(), "predict_proba")


def build(key: str) -> BaseEstimator:
    if key not in CLASSIFIERS:
        raise KeyError(f"Unknown classifier {key!r}; available: {sorted(CLASSIFIERS)}")
    pipeline = make_pipeline(StandardScaler(), CLASSIFIERS[key].factory())
    if needs_calibration(key):
        # Keeping the scaler inside the estimator makes each calibration fold
        # fit its own scaler on training data only.
        return CalibratedClassifierCV(pipeline, method="sigmoid", cv=3, ensemble=False)
    return pipeline


def coefficients(model: BaseEstimator | None):
    """Standardized weights for linear models, else None."""
    if model is None:
        return None
    if isinstance(model, CalibratedClassifierCV):
        model = model.calibrated_classifiers_[0].estimator
    coef = getattr(model.steps[-1][1], "coef_", None)
    return None if coef is None else coef[0]


register_classifier("logreg", "Logistic regression (L2)",
                    lambda: LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
register_classifier("svm", "SVM (RBF kernel)",
                    lambda: SVC(kernel="rbf", C=1.0, gamma="scale"))
register_classifier("svm_linear", "SVM (linear kernel)",
                    lambda: SVC(kernel="linear", C=1.0))
register_classifier("forest", "Random forest (shallow)",
                    lambda: RandomForestClassifier(
                        n_estimators=48, max_depth=4, min_samples_leaf=4,
                        max_features="sqrt", class_weight="balanced_subsample",
                        random_state=41, n_jobs=1))
