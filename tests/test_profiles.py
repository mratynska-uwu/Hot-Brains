"""Per-target feature profiles, new feature families, label exclusion and the added models."""
import copy
from pathlib import Path

import numpy as np
import pytest

from hotbrain import classifiers
from hotbrain.config import CONFIG, AppConfig, EEGSettings
from hotbrain.dataset import load_dataset, rethreshold
from hotbrain.features import ArtifactError, FeatureExtractor, TargetFeatures
from hotbrain.sessions import split_dataset
from hotbrain.training import MultiTrainer

ROOT = Path(__file__).resolve().parents[1]
APP = AppConfig.from_dict(CONFIG, ROOT)
RNG = np.random.default_rng(7)


def test_configuration_2_window_and_per_target_features():
    features = TargetFeatures(APP.eeg, APP.profile(2))
    spec = features.epoch_spec(3000)
    assert (spec.start_ms, spec.end_ms, spec.baseline_end_ms) == (-500, 450, -200)
    assert spec.n_samples == round(950 * APP.eeg.fs / 1000)
    names = features.names
    assert names["arousal"] == ("PO7 alpha 8-13 Hz log power", "Oz alpha 8-13 Hz log power",
                                "PO8 alpha 8-13 Hz log power")
    assert {n.split()[0] for n in names["valence"]} == {"Fz", "C3", "Cz", "C4"}
    assert {n.split()[1] for n in names["valence"]} == {"beta", "gamma"}
    vectors = features.extract(RNG.normal(0, 5, (spec.n_samples, 8)), spec)
    assert {t: len(v) for t, v in vectors.items()} == {"valence": 8, "arousal": 3}


def test_configuration_1_uses_all_electrodes_and_target_specific_families():
    features = TargetFeatures(APP.eeg, APP.profile(1))
    assert features.describe()["valence"]["features"] == ["band_power", "fractal_dimension", "asymmetry"]
    assert features.describe()["arousal"]["features"] == ["band_power", "fractal_dimension", "beta_alpha_ratio"]
    spec = features.epoch_spec(3000)
    assert (spec.start_ms, spec.end_ms) == (-500, 2700)
    vectors = features.extract(RNG.normal(0, 5, (spec.n_samples, 8)), spec)
    assert len(vectors["valence"]) == len(features.names["valence"]) == 8 * 4 + 8 + 2 * 4
    assert len(vectors["arousal"]) == 8 * 4 + 8 + 8


def test_gui_electrode_selection_limits_each_target():
    # Fz, C3, Cz only: valence keeps three electrodes, arousal (occipital) has none left.
    features = TargetFeatures(APP.eeg, APP.profile(2), positions=(0, 1, 2))
    assert features.targets == ("valence",)
    assert "arousal" in features.problems
    assert features.channel_indices["valence"] == (0, 1, 2)


def test_state_windows_are_demeaned_without_a_prestimulus_baseline():
    features = TargetFeatures(APP.eeg, APP.profile(2))
    spec = features.state_spec(2.0)
    assert (spec.start_ms, spec.end_ms, spec.baseline_samples, spec.onset_index) == (0, 2000, 0, 0)
    assert len(features.extract(RNG.normal(0, 5, (spec.n_samples, 8)), spec)["arousal"]) == 3


def test_new_feature_families_and_psd_methods():
    eeg = copy.deepcopy(CONFIG["eeg"])
    eeg["features"] = ["beta_alpha_ratio", "hjorth", "katz_fd", "petrosian_fd", "variance"]
    for method in ("welch", "fft"):
        eeg["psd_method"] = method
        extractor = FeatureExtractor(EEGSettings.from_dict(eeg))
        spec = extractor.epoch_spec(1000)
        values = extractor.extract(RNG.normal(0, 5, (spec.n_samples, 8)), spec)
        assert values.shape == (8 + 16 + 8 + 8 + 8,) and np.isfinite(values).all()


def test_artifact_threshold_and_common_average_reference():
    features = TargetFeatures(APP.eeg, APP.profile(2))
    spec = features.epoch_spec(1000)
    clean = RNG.normal(0, 5, (spec.n_samples, 8))
    features.extract(clean, spec)
    spiky = clean.copy()
    blink = 300 * np.hanning(50)                 # ~200 ms, 300 µV eye-blink-like deflection
    spiky[130:180, 6] += blink
    with pytest.raises(ArtifactError, match="amplitude"):
        features.extract(spiky, spec)
    flat = clean.copy()
    flat[:, 0] = 1.0
    with pytest.raises(ArtifactError, match="flat"):
        features.extract(flat, spec)
    eeg = copy.deepcopy(CONFIG["eeg"])
    eeg["reference"] = "car"
    car = TargetFeatures(EEGSettings.from_dict(eeg), APP.profile(2))
    common = clean + 30 * np.sin(np.linspace(0, 20, spec.n_samples))[:, None]
    assert np.allclose(car.reference(common), car.reference(clean))


def test_middle_tercile_is_excluded_and_threshold_comes_from_training_split():
    dataset = load_dataset(APP)
    train, _ = split_dataset(dataset, APP.validation_fraction, APP.split_seed)
    train_images = [i for items in train.values() for i in items]
    relabelled = rethreshold(APP, dataset, train_images)
    assert relabelled.threshold_source == "training split"
    values = sorted(i.values["arousal"] for i in train_images)
    assert relabelled.thresholds["arousal"] == pytest.approx(float(np.median(values)))
    margin = relabelled.margins["arousal"]
    for image in relabelled.images:
        distance = abs(image.values["arousal"] - relabelled.thresholds["arousal"])
        assert (image.labels["arousal"] is None) == (distance < margin)
    assert all(image.labels["valence"] is not None for image in relabelled.images)


def test_feature_selection_model_and_balanced_class_weights():
    x = RNG.normal(size=(60, 12))
    y = (x[:, 0] + .3 * RNG.normal(size=60) > .8).astype(int)   # imbalanced, one informative feature
    model = classifiers.build("logreg_select").fit(x, y)
    selector = model.steps[-1][1]
    assert selector.best_params_["select__k"] in (2, 5, 10, "all")
    assert abs(classifiers.coefficients(model)[0]) == max(abs(classifiers.coefficients(model)))
    assert classifiers.CLASSIFIERS["logreg"].factory().class_weight == "balanced"


def test_ridge_regression_learns_the_continuous_value():
    x = RNG.normal(size=(40, 4))
    target = np.clip(x[:, 1] / 2, -1, 1)
    model = classifiers.build("ridge").fit(x, target)
    probe = np.array([0, 1.5, 0, 0])
    assert classifiers.predict_value(model, probe) == pytest.approx(.75, abs=.15)
    assert classifiers.probability_class1(model, probe) > .8


def test_trainer_skips_excluded_labels_and_batches_do_not_predict_each_other():
    trainer = MultiTrainer(["valence", "arousal"], ["logreg", "ridge"], {"valence": 4, "arousal": 2},
                           [("valence", "logreg"), ("arousal", "logreg"), ("arousal", "ridge")])
    for i in range(30):
        v, a = i % 2, (i // 2) % 2
        trainer.add({"valence": RNG.normal(0, .3, 4) + v, "arousal": RNG.normal(0, .3, 2) + a},
                    {"valence": v, "arousal": None if i % 3 == 0 else a},
                    {"valence": 2 * v - 1, "arousal": a - .5})
    assert len(trainer.models[("valence", "logreg")].x) == 30
    assert len(trainer.models[("arousal", "logreg")].x) == 20      # every third arousal label excluded
    assert len(trainer.models[("arousal", "ridge")].x) == 30       # regression keeps the continuous value
    model_before = trainer.models[("valence", "logreg")].model
    rows = [({"valence": RNG.normal(0, .3, 4) + 1, "arousal": RNG.normal(0, .3, 2)},
             {"valence": 1, "arousal": 0}, {"valence": 1.0, "arousal": -.5}) for _ in range(3)]
    batch = trainer.add_batch(rows)
    assert len(batch[("valence", "logreg")]) == 3
    assert trainer.models[("valence", "logreg")].model is not model_before   # refit once after the block
    assert trainer.models[("valence", "logreg")].rolling_balanced_accuracy() is not None
