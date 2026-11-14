"""Training-quality report written after every training or calibration run.

For each trained target/model pair it summarises:

- the prequential (test-before-train) predictions made during the run:
  accuracy, balanced accuracy, precision, recall, F1, specificity, ROC AUC,
  Brier score, confusion matrix and the majority-class baseline;
- a stratified k-fold cross-validation of the same model on all collected
  trials, with a label-permutation test (p-value of the balanced accuracy);
- band ablation: cross-validated balanced accuracy without each band, and with
  that band alone (is gamma helping, or is it muscle noise?);
- trials whose out-of-fold probability for their own label is very low
  (possible label issues; uses cleanlab when installed).

Regression models (continuous valence/arousal) report MAE, Pearson r and the
same classification metrics for the sign of the prediction.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, permutation_test_score

from . import classifiers

PERMUTATION_BUDGET_S = 45.0   # per model; fewer permutations are run for slow models


def classification_metrics(y_true, y_pred, p_class1=None) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if not len(y_true):
        return {"n": 0}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    out = {"n": int(len(y_true)), "n_class0": int(np.sum(y_true == 0)), "n_class1": int(np.sum(y_true == 1)),
           "accuracy": float(np.mean(y_true == y_pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) == 2 else None,
           "precision": float(precision_score(y_true, y_pred, zero_division=0)),
           "recall": float(recall_score(y_true, y_pred, zero_division=0)),
           "f1": float(f1_score(y_true, y_pred, zero_division=0)),
           "specificity": tn / (tn + fp) if tn + fp else None,
           "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
           "majority_baseline": float(max(np.mean(y_true), 1 - np.mean(y_true)))}
    if p_class1 is not None and len(set(y_true)) == 2:
        p = np.clip(np.asarray(p_class1, dtype=float), 0, 1)
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["brier"] = float(brier_score_loss(y_true, p))
    return out


def regression_metrics(true_values, predicted) -> dict:
    t, p = np.asarray(true_values, dtype=float), np.asarray(predicted, dtype=float)
    if len(t) < 3:
        return {"n": int(len(t))}
    r = float(np.corrcoef(t, p)[0, 1]) if np.std(t) > 0 and np.std(p) > 0 else None
    return {"n": int(len(t)), "mae": float(mean_absolute_error(t, p)), "pearson_r": r}


def _cv(y_class: np.ndarray, folds: int, seed: int):
    smallest = int(np.bincount(y_class).min()) if len(set(y_class)) == 2 else 0
    return StratifiedKFold(max(2, min(folds, smallest)), shuffle=True, random_state=seed)


def _oof(model_key: str, X: np.ndarray, y: np.ndarray, cv) -> np.ndarray:
    """Out-of-fold P(class 1) (classifiers) or predicted value (regression)."""
    estimator = classifiers.build(model_key)
    if classifiers.is_regression(model_key):
        return cross_val_predict(estimator, X, y, cv=cv)
    return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]


def _cv_score(model_key, X, y, y_class, folds, seed) -> float | None:
    if X.shape[1] == 0:
        return None
    try:
        if classifiers.is_regression(model_key):
            values = _oof(model_key, X, y, KFold(min(folds, len(y)), shuffle=True, random_state=seed))
            keep = y_class >= 0
            return float(balanced_accuracy_score(y_class[keep], (values[keep] > 0).astype(int)))
        p = _oof(model_key, X, y_class, _cv(y_class, folds, seed))
        return float(balanced_accuracy_score(y_class, (p >= .5).astype(int)))
    except ValueError:
        return None


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    return value if isinstance(value, (int, float, str, bool, type(None))) else repr(value)


def model_parameters(model, info: list, top: int = 10) -> dict:
    """Hyperparameters and learned weights of the final fitted model (the one that would be saved)."""
    if model is None:
        return {"fitted": False}
    from sklearn.calibration import CalibratedClassifierCV
    inner = model.calibrated_classifiers_[0].estimator if isinstance(model, CalibratedClassifierCV) else model
    final = inner.steps[-1][1] if hasattr(inner, "steps") else inner
    out: dict = {"fitted": True, "estimator": type(final).__name__,
                 "probability": "3-fold sigmoid calibration" if isinstance(model, CalibratedClassifierCV) else "native",
                 "hyperparameters": {k: _plain(v) for k, v in final.get_params(deep=False).items()
                                     if k not in ("alphas", "k_options", "c_options")}}
    if isinstance(final, classifiers.SelectKLogistic):
        out["selected_by_inner_cv"] = {k: _plain(v) for k, v in final.best_params_.items()}
    if isinstance(final, classifiers.ContinuousRidge):
        out["selected_by_inner_cv"] = {"alpha": float(final.ridge_.alpha_)}
    names = [f.name for f in info]
    weights = classifiers.coefficients(model)
    if weights is not None and len(weights) == len(names):
        order = np.argsort(-np.abs(weights))[:top]
        out["top_weights"] = [{"feature": names[i], "standardized_weight": round(float(weights[i]), 4)} for i in order]
        out["intercept"] = _plain(np.ravel(getattr(final, "intercept_", [np.nan]))[0]) if hasattr(final, "intercept_") else None
    importances = getattr(final, "feature_importances_", None)
    if importances is not None and len(importances) == len(names):
        order = np.argsort(-importances)[:top]
        out["top_importances"] = [{"feature": names[i], "importance": round(float(importances[i]), 4)} for i in order]
    return out


def _label_issues(y_class, p_class1, tags, threshold) -> list[dict]:
    """Trials whose out-of-fold probability for their own label is below the threshold."""
    p_own = np.where(y_class == 1, p_class1, 1 - p_class1)
    method = "threshold"
    try:  # optional: confident learning when cleanlab is installed
        from cleanlab.filter import find_label_issues
        flagged = find_label_issues(y_class, np.column_stack([1 - p_class1, p_class1]))
        method = "cleanlab"
    except Exception:
        flagged = p_own < threshold
    rows = [{"trial": tags[i] if i < len(tags) else i, "label": int(y_class[i]),
             "p_own_label": round(float(p_own[i]), 3), "method": method}
            for i in np.flatnonzero(flagged)]
    return sorted(rows, key=lambda r: r["p_own_label"])


def evaluate_model(model_key: str, trainer, info: list, settings: dict) -> dict:
    """Prequential + cross-validated metrics, permutation test, band ablation and label issues."""
    regression = classifiers.is_regression(model_key)
    folds, seed = int(settings.get("cv_folds", 5)), int(settings.get("seed", 0))
    out: dict = {"model": classifiers.CLASSIFIERS[model_key].label, "kind": classifiers.CLASSIFIERS[model_key].kind,
                 "n_training_examples": len(trainer.x), "n_features": len(info),
                 "parameters": model_parameters(trainer.model, info)}
    scored = [p for p in trainer.predictions if p.correct is not None]
    out["prequential"] = classification_metrics([p.truth for p in scored], [p.label for p in scored],
                                                 [p.probability for p in scored])
    if regression:
        with_values = [p for p in trainer.predictions if p.value is not None and p.true_value is not None]
        out["prequential"]["regression"] = regression_metrics([p.true_value for p in with_values],
                                                              [p.value for p in with_values])
    if len(trainer.x) < 10:
        out["cross_validation"] = {"skipped": "fewer than 10 training examples"}
        return out
    X = np.vstack(trainer.x)
    y_class = np.asarray(trainer.y, dtype=int)
    y = np.asarray(trainer.values, dtype=float) if regression else y_class
    tags = list(getattr(trainer, "tags", []))
    try:
        if regression:
            cv = KFold(min(folds, len(y)), shuffle=True, random_state=seed)
            values = _oof(model_key, X, y, cv)
            keep = y_class >= 0
            cv_metrics = classification_metrics(y_class[keep], (values[keep] > 0).astype(int), (values[keep] + 1) / 2)
            cv_metrics["regression"] = regression_metrics(y, values)
            p_class1 = (np.clip(values, -1, 1) + 1) / 2
        else:
            cv = _cv(y_class, folds, seed)
            p_class1 = _oof(model_key, X, y_class, cv)
            cv_metrics = classification_metrics(y_class, (p_class1 >= .5).astype(int), p_class1)
        cv_metrics["folds"] = cv.get_n_splits()
        out["cross_validation"] = cv_metrics
    except ValueError as exc:
        out["cross_validation"] = {"skipped": str(exc)}
        return out

    # Permutation test of the cross-validated score (#1), within a time budget.
    requested = int(settings.get("permutations", 100))
    if requested > 0 and not regression and len(set(y_class)) == 2:
        started = time.perf_counter()
        _cv_score(model_key, X, y, y_class, folds, seed)
        one = max(time.perf_counter() - started, 1e-3)
        n_perm = int(min(requested, max(10, PERMUTATION_BUDGET_S / one)))
        score, permuted, p_value = permutation_test_score(
            classifiers.build(model_key), X, y_class, cv=cv, scoring="balanced_accuracy",
            n_permutations=n_perm, random_state=seed)
        out["permutation_test"] = {"balanced_accuracy": float(score), "p_value": float(p_value),
                                   "n_permutations": n_perm, "chance_mean": float(np.mean(permuted)),
                                   "chance_95th": float(np.percentile(permuted, 95))}
    elif regression and requested > 0:
        out["permutation_test"] = {"skipped": "regression model (see cross-validated MAE / r)"}

    # Band ablation (#6): drop each band, and each band alone.
    bands = sorted({b for f in info for b in ([f.band] if f.band and "/" not in f.band else
                                                (f.band.split("/") if f.band else []))})
    ablation = {}
    if len(bands) > 1:
        baseline = out["cross_validation"].get("balanced_accuracy")
        for band in bands:
            uses = np.array([bool(f.band) and band in f.band.split("/") for f in info])
            ablation[band] = {"without": _cv_score(model_key, X[:, ~uses], y, y_class, folds, seed),
                              "only": _cv_score(model_key, X[:, uses], y, y_class, folds, seed),
                              "with_all": baseline, "n_features": int(uses.sum())}
    out["band_ablation"] = ablation or {"skipped": "one band only"}

    # Possible label issues (#21).
    labelled = y_class >= 0
    out["label_issues"] = _label_issues(y_class[labelled], p_class1[labelled],
                                        [t for t, keep in zip(tags, labelled) if keep] if tags else [],
                                        float(settings.get("label_issue_threshold", .2)))
    return out


def build_report(models: dict, info_by_target: dict, metadata: dict, settings: dict) -> dict:
    """Full report: one entry per target/model plus the run's data-quality summary."""
    started = time.perf_counter()
    log = metadata.get("trial_log", [])
    statuses: dict[str, int] = {}
    for row in log:
        statuses[row.get("eeg", "unknown")] = statuses.get(row.get("eeg", "unknown"), 0) + 1
    rejected = sum(n for s, n in statuses.items() if s not in ("usable", "continuous"))
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "session": metadata.get("session"), "participant_id": metadata.get("participant_id"),
        "configuration": metadata.get("configuration"), "evaluation_mode": metadata.get("evaluation_mode"),
        "image_duration_s": metadata.get("image_duration_s"),
        "epoch_window_ms": [metadata.get("epoch_start_ms"), metadata.get("epoch_end_ms")],
        "target_features": metadata.get("target_features"),
        "data_quality": {"presented": len(log), "eeg_status_counts": statuses,
                         "rejected_or_missing": rejected,
                         "rejection_rate": rejected / len(log) if log else None,
                         "artifact_reasons": metadata.get("artifact_reasons", {}),
                         "calibration_windows": metadata.get("calibration_windows")},
        "stop_rule": metadata.get("stop_rule_result"),
        "resting_reference": metadata.get("resting_reference_file"),
        "models": {}}
    for (target, key), trainer in models.items():
        report["models"][f"{target}/{key}"] = evaluate_model(key, trainer, list(info_by_target.get(target, ())), settings)
    report["computation_s"] = round(time.perf_counter() - started, 2)
    return report


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _num(value, digits=3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def to_markdown(report: dict) -> str:
    lines = [f"# Training quality report · {report.get('participant_id')} · {report.get('created_at', '')[:19]}", "",
             f"- Session: **{report.get('session')}** · configuration **{report.get('configuration')}** · "
             f"evaluation mode {report.get('evaluation_mode')}",
             f"- Image {report.get('image_duration_s')} s · epoch {report.get('epoch_window_ms')} ms around onset"]
    dq = report["data_quality"]
    lines += [f"- Presented trials: {dq['presented']} · rejected/missing EEG: {dq['rejected_or_missing']} "
              f"({_pct(dq['rejection_rate'])}) · statuses: {dq['eeg_status_counts']}"]
    if dq.get("artifact_reasons"):
        lines.append(f"- Artifact reasons: {dq['artifact_reasons']}")
    if dq.get("calibration_windows"):
        lines.append(f"- Calibration windows: {dq['calibration_windows']}")
    if report.get("stop_rule"):
        lines.append(f"- Stopping rule: {report['stop_rule']}")
    lines += ["", "Valence and arousal are separate classifiers: each target has its own feature vector "
              "(electrodes, bands, families) and its own fitted model(s), evaluated and reported separately."]
    lines += ["", "## Summary", "",
              "| Target / model | Prequential bal. acc | CV bal. acc | CV precision | CV recall | CV F1 | CV AUC | "
              "Majority baseline | Permutation p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, entry in report["models"].items():
        pq, cv = entry.get("prequential", {}), entry.get("cross_validation", {})
        perm = entry.get("permutation_test", {})
        lines.append(f"| {key} | {_pct(pq.get('balanced_accuracy'))} | {_pct(cv.get('balanced_accuracy'))} | "
                     f"{_pct(cv.get('precision'))} | {_pct(cv.get('recall'))} | {_pct(cv.get('f1'))} | "
                     f"{_num(cv.get('roc_auc'))} | {_pct(cv.get('majority_baseline'))} | {_num(perm.get('p_value'))} |")
    targets = list(dict.fromkeys(key.split("/")[0] for key in report["models"]))
    entries = []
    for target in targets:
        spec = (report.get("target_features") or {}).get(target, {})
        entries.append(("target", target, spec))
        entries += [("model", key, entry) for key, entry in report["models"].items() if key.split("/")[0] == target]
    for kind, key, entry in entries:
        if kind == "target":
            lines += ["", f"# {key.title()}", "",
                      f"Features: {entry.get('n_features', '—')} · electrodes {', '.join(entry.get('electrodes', []))} "
                      f"· bands {', '.join(entry.get('bands', []))} · families {', '.join(entry.get('features', []))}"]
            continue
        lines += ["", f"## {key} · {entry['model']}", "",
                  f"Training examples: {entry['n_training_examples']} · features: {entry.get('n_features')}", ""]
        params = entry.get("parameters", {})
        lines.append("**Model parameters (final fitted model)**")
        if not params.get("fitted"):
            lines += ["- not fitted (too few trials or one class only)", ""]
        else:
            lines.append(f"- {params['estimator']} · probabilities: {params['probability']}")
            lines.append("- hyperparameters: " + ", ".join(f"{k}={v}" for k, v in params["hyperparameters"].items()))
            if params.get("selected_by_inner_cv"):
                lines.append("- chosen by inner cross-validation: "
                             + ", ".join(f"{k}={v}" for k, v in params["selected_by_inner_cv"].items()))
            if params.get("top_weights"):
                lines.append("- largest standardized weights: " + "; ".join(
                    f"{w['feature']} {w['standardized_weight']:+.3f}" for w in params["top_weights"]))
            if params.get("top_importances"):
                lines.append("- largest importances: " + "; ".join(
                    f"{w['feature']} {w['importance']:.3f}" for w in params["top_importances"]))
            lines.append("")
        for title, block in (("Prequential (test-before-train during the run)", entry.get("prequential", {})),
                             ("Cross-validation on all collected trials", entry.get("cross_validation", {}))):
            lines.append(f"**{title}**")
            if "skipped" in block:
                lines += [f"- skipped: {block['skipped']}", ""]
                continue
            cm = block.get("confusion_matrix", {})
            lines += [f"- n = {block.get('n')} (class 0: {block.get('n_class0')}, class 1: {block.get('n_class1')})",
                      f"- accuracy {_pct(block.get('accuracy'))} · balanced accuracy {_pct(block.get('balanced_accuracy'))} "
                      f"· majority baseline {_pct(block.get('majority_baseline'))}",
                      f"- precision {_pct(block.get('precision'))} · recall {_pct(block.get('recall'))} "
                      f"· specificity {_pct(block.get('specificity'))} · F1 {_pct(block.get('f1'))}",
                      f"- ROC AUC {_num(block.get('roc_auc'))} · Brier {_num(block.get('brier'))}",
                      f"- confusion matrix: TN {cm.get('tn')} · FP {cm.get('fp')} · FN {cm.get('fn')} · TP {cm.get('tp')}"]
            if block.get("regression"):
                reg = block["regression"]
                lines.append(f"- regression: MAE {_num(reg.get('mae'))} · Pearson r {_num(reg.get('pearson_r'))}")
            lines.append("")
        perm = entry.get("permutation_test")
        if perm:
            lines += ["**Permutation test**", ("- skipped: " + perm["skipped"]) if "skipped" in perm else
                      f"- balanced accuracy {_pct(perm['balanced_accuracy'])} · p = {perm['p_value']:.3f} "
                      f"({perm['n_permutations']} permutations; chance mean {_pct(perm['chance_mean'])}, "
                      f"95th percentile {_pct(perm['chance_95th'])})", ""]
        ablation = entry.get("band_ablation", {})
        if ablation and "skipped" not in ablation:
            lines += ["**Band ablation (cross-validated balanced accuracy)**", "",
                      "| Band | Without it | It alone | Features |", "|---|---|---|---|"]
            for band, row in ablation.items():
                lines.append(f"| {band} | {_pct(row['without'])} | {_pct(row['only'])} | {row['n_features']} |")
            lines.append("")
        issues = entry.get("label_issues", [])
        lines.append(f"**Possible label issues:** {len(issues)}"
                     + (" — lowest: " + ", ".join(f"trial {r['trial']} (p={r['p_own_label']})" for r in issues[:8])
                        if issues else ""))
    lines += ["", f"_Computed in {report.get('computation_s')} s._", ""]
    return "\n".join(lines)


def write_report(report: dict, directory: Path, stem: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    md_path = directory / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return md_path, json_path


def clone_trainers(models: dict) -> dict:
    """Snapshot trainers so the report can run in a worker while the GUI moves on."""
    import copy
    out = {}
    for key, trainer in models.items():
        snap = copy.copy(trainer)
        snap.x, snap.y = list(trainer.x), list(trainer.y)
        snap.values = list(getattr(trainer, "values", []))
        snap.tags = list(getattr(trainer, "tags", []))
        snap.predictions = list(trainer.predictions)
        snap.model = None if trainer.model is None else copy.deepcopy(trainer.model)
        out[key] = snap
    return out
