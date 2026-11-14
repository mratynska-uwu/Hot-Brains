"""Training-quality report and the calibration/resting helpers."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from hotbrain.config import CONFIG, AppConfig
from hotbrain.dataset import StimulusImage
from hotbrain.features import TargetFeatures
from hotbrain.gui import calibrate_block, resting_reference, train_epoch
from hotbrain.report import build_report, clone_trainers, write_report
from hotbrain.sessions import ValidationScores, affect_point, calibration_images, sliding_windows
from hotbrain.training import MultiTrainer

ROOT = Path(__file__).resolve().parents[1]
APP = AppConfig.from_dict(CONFIG, ROOT)
SETTINGS = {**APP.report, "permutations": 10, "cv_folds": 4}


def _trained(seed=0, n=60):
    features = TargetFeatures(APP.eeg, APP.profile(2))
    sizes = features.n_features()
    rng = np.random.default_rng(seed)
    pairs = [("valence", "logreg"), ("valence", "ridge"), ("arousal", "logreg"), ("arousal", "forest")]
    trainer = MultiTrainer(["valence", "arousal"], ["logreg", "ridge", "forest"], sizes, pairs)
    for i in range(n):
        v, a = i % 2, (i // 3) % 2
        valence = rng.normal(0, 1, sizes["valence"])
        valence[[1, 3]] += 1.5 * v                     # gamma features carry valence
        trainer.add({"valence": valence, "arousal": rng.normal(0, 1, sizes["arousal"]) + .8 * a},
                    {"valence": v, "arousal": a}, {"valence": 2 * v - 1, "arousal": 2 * a - 1}, tag=i + 1)
    return features, trainer


def test_report_has_separate_valence_and_arousal_models_with_metrics_and_parameters(tmp_path):
    features, trainer = _trained()
    meta = {"session": "training", "participant_id": "p1", "configuration": 2,
            "target_features": features.describe(),
            "trial_log": [{"eeg": "usable"}] * 60 + [{"eeg": "rejected_artifact"}] * 3,
            "artifact_reasons": {"amplitude": 3}}
    report = build_report(clone_trainers(trainer.models), {t: e.info for t, e in features.extractors.items()},
                          meta, SETTINGS)
    assert set(report["models"]) == {"valence/logreg", "valence/ridge", "arousal/logreg", "arousal/forest"}
    valence, arousal = report["models"]["valence/logreg"], report["models"]["arousal/logreg"]
    assert (valence["n_features"], arousal["n_features"]) == (8, 3)        # separate feature vectors
    for entry in (valence, arousal):
        for block in ("prequential", "cross_validation"):
            assert {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "confusion_matrix"} <= set(entry[block])
        assert 0 < entry["permutation_test"]["p_value"] <= 1
        assert entry["parameters"]["fitted"] and entry["parameters"]["top_weights"]
    assert valence["parameters"]["top_weights"][0]["feature"].split()[1] == "gamma"
    assert report["models"]["arousal/forest"]["parameters"]["top_importances"]
    assert "alpha" in report["models"]["valence/ridge"]["parameters"]["selected_by_inner_cv"]
    assert "regression" in report["models"]["valence/ridge"]["cross_validation"]
    ablation = valence["band_ablation"]
    assert set(ablation) == {"beta", "gamma"} and ablation["gamma"]["only"] > ablation["beta"]["only"]
    assert report["data_quality"]["rejection_rate"] == 3 / 63
    md_path, json_path = write_report(report, tmp_path, "run")
    text = md_path.read_text(encoding="utf-8")
    assert "# Valence" in text and "# Arousal" in text and "Model parameters" in text and json_path.exists()


def test_validation_permutation_p_and_dict_vectors():
    features, trainer = _trained()
    models = {k: t.model for k, t in trainer.models.items()}
    scores = ValidationScores(models)
    rng = np.random.default_rng(5)
    for i in range(30):
        v = i % 2
        vec = rng.normal(0, 1, 8)
        vec[[1, 3]] += 1.5 * v
        scores.add({"valence": vec, "arousal": rng.normal(0, 1, 3)}, {"valence": v, "arousal": None})
    assert scores.accuracy(("valence", "logreg"))[2] == 30
    assert scores.accuracy(("arousal", "logreg"))[2] == 0            # excluded labels are not scored
    assert scores.permutation_p(("valence", "logreg")) < .05
    point = affect_point(models, "logreg", {"valence": np.zeros(8), "arousal": np.zeros(3)}, ("valence", "arousal"))
    assert all(-1 <= v <= 1 for v in point)


def test_calibration_windows_block_learning_and_resting_reference():
    assert len(sliding_windows(np.zeros((20 * 250, 8)), 250, 2.0, 2.0, 1.0)) == 17
    images = {"g": [StimulusImage(Path(f"{i}.png"), "g", {}, {}) for i in range(5)]}
    picked = calibration_images(images, {f"{i}.png": float(i) for i in range(5)}, 2)
    assert [i.path.name for i in picked["g"]] == ["4.png", "3.png"]
    features = TargetFeatures(APP.eeg, APP.profile(2))
    trainer = MultiTrainer(["valence", "arousal"], ["logreg"], features.n_features())
    rng = np.random.default_rng(1)
    for block in range(6):
        segment = rng.normal(0, 5, (8 * 250, 8))
        vectors, predictions, _, reasons, total = calibrate_block(
            features, trainer, segment, {"valence": block % 2, "arousal": 0}, {"valence": 0., "arousal": 0.},
            dict(APP.calibration, skip_s=1.0), tag=block)
        assert total == len(vectors) + sum(reasons.values()) == 6
    assert len(trainer.models[("valence", "logreg")].x) == 36
    reference = resting_reference(APP.eeg, rng.normal(0, 5, (10 * 250, 8)))
    assert reference["usable_windows"] == 5 and len(reference["log_band_power"]) == 8 * 4


def test_train_epoch_reports_rejection_reason():
    features = TargetFeatures(APP.eeg, APP.profile(2))
    spec = features.epoch_spec(1000)
    epoch = np.random.default_rng(2).normal(0, 5, (spec.n_samples, 8))
    vectors, predictions, _ = train_epoch(features, spec, None, epoch, {"valence": 1})
    assert set(vectors) == {"valence", "arousal"} and predictions is None
    epoch[100:160, 5] += 300 * np.hanning(60)
    vectors, reason, _ = train_epoch(features, spec, None, epoch, {"valence": 1})
    assert vectors is None and "amplitude" in reason
