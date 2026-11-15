"""Reproducible exclusion of distracted scene trials and clean offline refit.

The original continuous EEG and saved manifest are intentionally immutable.
Run from the repo with .venv/bin/python scripts/reanalyze_trimmed_run.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotbrain.classifiers import build, CLASSIFIERS
from hotbrain.config import EEGSettings
from hotbrain.features import EpochSpec, FeatureExtractor


SOURCE = ROOT / "saved_models/scene_debug_20261115_024214_587640.json"
STEM = ROOT / "saved_models/scene_debug_20261115_024214_587640_trimmed"


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    hits = int(np.sum((p >= .5) == y))
    return {"n": len(y), "class_0": int(np.sum(y == 0)), "class_1": int(np.sum(y == 1)),
            "correct": hits, "accuracy": float(accuracy_score(y, p >= .5)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5)),
            "auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "coinflip_brier": .25,
            "binomial_p_one_sided": float(binomtest(hits, len(y), .5, alternative="greater").pvalue)}


def live_metrics(trials: list[dict], role: str) -> dict:
    rows = [(int(row["labels"]["valence"] == "positive"),
             float(row["predictions"]["valence/logreg"]["p_class1"]))
            for row in trials if row["trial_role"] == role
            and "valence/logreg" in row.get("predictions", {})]
    return metrics(np.array([y for y, _ in rows]), np.array([p for _, p in rows]))


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    last = int(source["presented_trials"])
    excluded = set(range(1, 11)) | set(range(last - 9, last + 1))
    original_epochs = np.load(ROOT / source["epochs_file"])
    keep = ~np.isin(original_epochs["trial_number"], list(excluded))
    cleaned = {key: original_epochs[key][keep] if len(original_epochs[key].shape) > 0
               and original_epochs[key].shape[0] == len(keep) else original_epochs[key]
               for key in original_epochs.files}
    epochs_path = STEM.with_suffix(".epochs.npz")
    np.savez_compressed(epochs_path, **cleaned)

    eeg = EEGSettings.from_dict(source["config"]["eeg"])
    model_eeg = eeg.select_electrodes(tuple(eeg.device_channels.index(ch)
                                            for ch in source["model_device_channels"]))
    extractor = FeatureExtractor(model_eeg)
    assert extractor.names == tuple(source["feature_names"])
    spec = EpochSpec(round(float(source["image_duration_s"]) * 1000), eeg.fs)
    channel_positions = [eeg.device_channels.index(ch) for ch in model_eeg.device_channels]
    X = np.stack([extractor.extract(epoch[:, channel_positions], spec)
                  for epoch in cleaned["eeg_uV"]])
    train = cleaned["trial_role"] == "training"
    val = cleaned["trial_role"] == "validation"
    assert len(X) == len(cleaned["label_valence"]) and np.all(train | val)
    comparison = {}
    model_entries = {}
    probability_archive = {"trial_number": cleaned["trial_number"][val],
                           "label_valence": cleaned["label_valence"][val],
                           "label_arousal": cleaned["label_arousal"][val]}
    for target in ("valence", "arousal"):
        y = cleaned[f"label_{target}"]
        for key in CLASSIFIERS:
            model = build(key)
            model.fit(X[train], y[train])
            train_p = model.predict_proba(X[train])[:, 1]
            val_p = model.predict_proba(X[val])[:, 1]
            label = f"{target}/{key}"
            comparison[label] = {"training": metrics(y[train], train_p),
                                 "held_out": metrics(y[val], val_p)}
            probability_archive[f"p_{target}_{key}"] = val_p
            model_path = STEM.with_name(f"{STEM.name}.{target}.{key}.joblib")
            joblib.dump(model, model_path)
            model_entries[label] = {"classifier": CLASSIFIERS[key].label,
                                    "n_eeg_trials": int(np.sum(train)),
                                    "selected_for_save": True,
                                    "model_file": model_path.name,
                                    "source": "clean offline refit on retained training epochs"}
    probabilities_path = STEM.with_suffix(".predictions.npz")
    np.savez_compressed(probabilities_path, **probability_archive)
    refit_result = dict(comparison["valence/logreg"]["held_out"])
    refit_result["training_eeg_trials"] = int(np.sum(train))
    y = cleaned["label_valence"]
    old_file = SOURCE.with_name(source["models"]["valence/logreg"]["model_file"])
    historic_model = joblib.load(old_file)
    historic_result = metrics(y[val], historic_model.predict_proba(X[val])[:, 1])

    retained_trials = [row for row in source["trial_log"] if row["number"] not in excluded]
    assert len(retained_trials) == last - len(excluded)
    record = dict(source)
    record.update({"schema": 5, "derived_from": str(SOURCE.relative_to(ROOT)),
                   "curation": {"excluded_scene_trial_numbers": sorted(excluded),
                                "reason": "Participant reported distraction during first and last ten scene presentations",
                                "original_continuous_recording_preserved": True,
                                "original_recording_contains_excluded_trials": True,
                                "selection_warning": "Validation data had been used for earlier checkpoint selection; scores are exploratory."},
                   "presented_trials": len(retained_trials), "requested_trials": len(retained_trials),
                   "trial_log": retained_trials,
                   "usable_training_trials": int(np.sum(train)),
                   "usable_validation_trials": int(np.sum(val)),
                   "skipped_eeg_trials": int(sum(row.get("eeg") != "usable" for row in retained_trials)),
                   "epochs_file": str(epochs_path.relative_to(ROOT)),
                   "offline_predictions_file": str(probabilities_path.relative_to(ROOT)),
                   "offline_comparison": comparison,
                   "checkpoint_selection": {},
                   "offline_refit": refit_result,
                   "historic_saved_checkpoint_on_retained_validation": historic_result,
                   "trimmed_live_training": live_metrics(retained_trials, "training"),
                   "trimmed_live_validation": live_metrics(retained_trials, "validation"),
                   "models": model_entries,
                   "saved_models": list(model_entries)})
    manifest_path = STEM.with_suffix(".json")
    manifest_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "epochs": str(epochs_path),
                      "excluded": sorted(excluded), "refit": refit_result,
                      "historic": historic_result,
                      "offline_comparison": comparison,
                      "live_training": record["trimmed_live_training"],
                      "live_validation": record["trimmed_live_validation"]}, indent=2))


if __name__ == "__main__":
    main()
