"""Registry of models that can be trained on EEG feature vectors.

Add a new one with ``register_classifier("key", "Label", factory)`` where the
factory returns an unfitted scikit-learn estimator. Every estimator is wrapped
in a StandardScaler pipeline. Estimators without ``predict_proba`` are
calibrated with cross-validated sigmoid calibration around the whole pipeline.

``kind="regression"`` models learn the continuous target value (scaled to
[-1, 1] around the class boundary) instead of the binary label; they still
expose ``predict`` (sign -> class) and ``predict_proba`` so they plug into the
same accuracy tables, validation and live valence-arousal display.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class ClassifierSpec:
    key: str
    label: str
    factory: Callable[[], BaseEstimator]
    kind: str = "classifier"        # "classifier" (binary labels) or "regression" (continuous values)
    min_per_class: int = 2          # fewest examples per class before the first fit


CLASSIFIERS: dict[str, ClassifierSpec] = {}


def register_classifier(key: str, label: str, factory: Callable[[], BaseEstimator],
                        kind: str = "classifier", min_per_class: int = 2) -> None:
    CLASSIFIERS[key] = ClassifierSpec(key, label, factory, kind, min_per_class)


def needs_calibration(key: str) -> bool:
    return not hasattr(CLASSIFIERS[key].factory(), "predict_proba")


def is_regression(key: str) -> bool:
    return CLASSIFIERS[key].kind == "regression"


def min_per_class(key: str) -> int:
    return max(CLASSIFIERS[key].min_per_class, 3 if needs_calibration(key) else 2)


def build(key: str) -> BaseEstimator:
    if key not in CLASSIFIERS:
        raise KeyError(f"Unknown classifier {key!r}; available: {sorted(CLASSIFIERS)}")
    pipeline = make_pipeline(StandardScaler(), CLASSIFIERS[key].factory())
    if needs_calibration(key):
        # Keeping the scaler inside the estimator makes each calibration fold
        # fit its own scaler on training data only.
        return CalibratedClassifierCV(pipeline, method="sigmoid", cv=3, ensemble=False)
    return pipeline


def probability_class1(model: BaseEstimator, features) -> float:
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


def coefficients(model: BaseEstimator | None):
    """Standardized weights for linear models, else None."""
    if model is None:
        return None
    if isinstance(model, CalibratedClassifierCV):
        model = model.calibrated_classifiers_[0].estimator
    coef = getattr(model.steps[-1][1], "coef_", None)
    if coef is None:
        return None
    return np.ravel(coef) if np.ndim(coef) == 1 else coef[0]


class SelectKLogistic(ClassifierMixin, BaseEstimator):
    """L2 logistic regression with the number of features (SelectKBest, ANOVA F) and C
    chosen by an inner stratified cross-validation on the training data only."""

    def __init__(self, k_options=(2, 5, 10, 20), c_options=(0.01, 0.1, 1.0, 10.0), folds: int = 3, seed: int = 0):
        self.k_options = k_options
        self.c_options = c_options
        self.folds = folds
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        n_features = X.shape[1]
        ks = sorted({k for k in self.k_options if k < n_features}) + ["all"]
        folds = max(2, min(self.folds, int(np.bincount(y.astype(int)).min())))
        pipeline = Pipeline([("select", SelectKBest(f_classif)),
                             ("clf", LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"))])
        search = GridSearchCV(pipeline, {"select__k": ks, "clf__C": list(self.c_options)},
                              cv=StratifiedKFold(folds, shuffle=True, random_state=self.seed),
                              scoring="balanced_accuracy", error_score=0.0)
        search.fit(X, y)
        self.best_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.classes_ = self.best_.classes_
        mask = self.best_.named_steps["select"].get_support()
        coef = np.zeros(n_features)
        coef[mask] = self.best_.named_steps["clf"].coef_[0]
        self.coef_ = coef[None, :]
        return self

    def predict(self, X):
        return self.best_.predict(X)

    def predict_proba(self, X):
        return self.best_.predict_proba(X)


class ContinuousRidge(ClassifierMixin, BaseEstimator):
    """Ridge regression on the continuous target in [-1, 1]; the sign gives the class.

    ``predict_proba`` maps the clipped prediction v to P(class 1) = (v + 1) / 2,
    which is what the live plane shows: 2 * P - 1 = the predicted position.
    """

    def __init__(self, alphas=tuple(np.logspace(-2, 3, 12))):
        self.alphas = alphas

    def fit(self, X, y):
        self.ridge_ = RidgeCV(alphas=self.alphas).fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
        self.classes_ = np.array([0, 1])
        self.coef_ = self.ridge_.coef_[None, :]
        return self

    def predict_value(self, X):
        return np.clip(self.ridge_.predict(X), -1, 1)

    def predict(self, X):
        return (self.predict_value(X) > 0).astype(int)

    def predict_proba(self, X):
        p = (self.predict_value(X) + 1) / 2
        return np.column_stack([1 - p, p])


def predict_value(model: BaseEstimator, features) -> float | None:
    """Continuous prediction in [-1, 1] for regression models, else None."""
    x = np.asarray(features, dtype=np.float64).reshape(1, -1)
    final = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if isinstance(final, ContinuousRidge):
        if isinstance(model, Pipeline):
            x = model[:-1].transform(x)
        return float(final.predict_value(x)[0])
    return None


register_classifier("logreg", "Logistic regression (L2)",
                    lambda: LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", class_weight="balanced"))
register_classifier("svm", "SVM (RBF kernel)",
                    lambda: SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))
register_classifier("svm_linear", "SVM (linear kernel)",
                    lambda: SVC(kernel="linear", C=1.0, class_weight="balanced"))
register_classifier("forest", "Random forest (shallow)",
                    lambda: RandomForestClassifier(
                        n_estimators=48, max_depth=4, min_samples_leaf=4,
                        max_features="sqrt", class_weight="balanced_subsample",
                        random_state=41, n_jobs=1))
# Shrinkage LDA: the class covariance is shrunk towards a scaled identity with the
# analytic Ledoit-Wolf intensity, so it stays well conditioned when there are
# only a few trials per feature. It is the standard small-sample BCI classifier
# (Blankertz et al. 2011, NeuroImage 56:814). Equal priors aim at balanced accuracy.
register_classifier("lda_shrinkage", "Shrinkage LDA (Ledoit-Wolf, small samples)",
                    lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[.5, .5]))
register_classifier("logreg_select", "LogReg + feature selection (nested CV)",
                    lambda: SelectKLogistic(), min_per_class=3)
register_classifier("ridge", "Ridge regression (continuous V-A)",
                    lambda: ContinuousRidge(), kind="regression")
