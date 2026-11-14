import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest

from PySide6.QtCore import QCoreApplication

from hotbrain.acquisition import DeviceManager, SampleRing
from hotbrain.config import CONFIG, AppConfig, EEGSettings
from hotbrain.dataset import StimulusImage, load_dataset
from hotbrain.features import EpochSpec, FeatureExtractor, higuchi_fd
from hotbrain.training import MultiTrainer, TrialPlan, predict_saved

ROOT = Path(__file__).resolve().parents[1]
EEG = EEGSettings.from_dict(CONFIG["eeg"])
N_CH = len(EEG.electrodes)


def test_stream_ready_requires_recent_sdk_frames(monkeypatch):
    QCoreApplication.instance() or QCoreApplication([])
    monkeypatch.setattr(DeviceManager, "_tick", lambda self: None)
    device = DeviceManager(EEG)
    try:
        assert not device.stream_ready()
        device.connected = True  # SDK opened, but no frames yet.
        device.pipeline = type("PipelineStub", (), {"get_condition": lambda self: "Healthy"})()
        assert not device.stream_ready()
        device.ring.append(np.full((EEG.fs, N_CH), np.nan))
        assert device.ring.received_samples == EEG.fs
        assert not device.stream_ready()
        device.ring.append(np.ones((EEG.fs // 2 - 1, N_CH)))
        assert not device.stream_ready()
        device.ring.append(np.ones((1, N_CH)))
        assert device.stream_ready()
        with device.ring._lock:
            device.ring._last_arrival -= 2
        assert not device.stream_ready()
        device.disconnect()
        assert not device.stream_ready()
        assert device.ring.received_samples == 0
    finally:
        device.close()


def test_config_matches_basic_info():
    assert EEG.fs == 250
    assert EEG.electrodes == ("Fz", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz")
    assert EEG.bandpass_hz == (1, 48)
    assert EEG.notch_hz == 50
    assert [name for name, _ in EEG.bands_hz] == ["alpha", "beta", "gamma"]
    assert EEG.features == ("band_power", "fractal_dimension")
    assert EEG.warnings() == []  # gamma 30-45 Hz fits inside the 1-48 Hz bandpass
    narrow = copy.deepcopy(CONFIG["eeg"])
    narrow["bandpass_hz"] = [1, 30]
    assert any("gamma" in w for w in EEGSettings.from_dict(narrow).warnings())


def test_target_formula_from_weighted_columns():
    from hotbrain.config import TargetSpec
    spec = TargetSpec.from_dict("valence", {"columns": {"positive_mean": 1, "negative_mean": -1}})
    assert spec.formula == "positive_mean - negative_mean"
    assert spec.value({"positive_mean": "6", "negative_mean": "1.5"}) == 4.5
    with pytest.raises(ValueError, match="exactly one"):
        TargetSpec.from_dict("x", {"column": "a", "columns": {"b": 1}})


def test_config_rejects_unknown_electrode_in_region():
    bad = copy.deepcopy(CONFIG["eeg"])
    bad["regions"]["frontal"] = ["F2"]
    with pytest.raises(ValueError, match="F2"):
        EEGSettings.from_dict(bad)


def test_real_dataset_loads_with_valence_and_arousal_labels():
    """data/training/images + data/training/records.csv, trials balanced by valence class."""
    dataset = load_dataset(AppConfig.from_dict(CONFIG, ROOT))
    assert set(dataset.groups) == {"valence=negative", "valence=positive"}
    negative, positive = dataset.groups["valence=negative"], dataset.groups["valence=positive"]
    assert len(negative) + len(positive) == len(list((ROOT / "data/training/images").iterdir()))
    assert all(i.labels["valence"] == 0 for i in negative) and all(i.labels["valence"] == 1 for i in positive)
    assert all(i.path.parent.name == "images" for i in negative + positive)
    rows = {r["image_name"]: r for r in csv.DictReader(open(ROOT / "data/training/records.csv", encoding="utf-8"))}
    for r in rows.values():
        assert np.isclose(float(r["valence"]), float(r["positive_mean"]) - float(r["negative_mean"]))
        assert r["arousal"] == r["aroused_mean"]
    first = positive[0]
    assert np.isclose(first.values["valence"], float(rows[first.path.name]["valence"]))
    assert np.isclose(first.values["arousal"], float(rows[first.path.name]["arousal"]))
    # CSV rows whose image file is absent are skipped and reported, not fatal.
    assert dataset.rows_without_image == sorted(set(rows) - {i.path.name for i in negative + positive})
    arousal = [i.labels["arousal"] for i in negative + positive]
    assert 0 < sum(arousal) < len(arousal)
    assert dataset.thresholds["arousal"] is not None


def test_dataset_with_explicitly_named_metadata_csv(tmp_path):
    folder = tmp_path / "data" / "set" / "images"
    folder.mkdir(parents=True)
    lines = ["image_name,valence,arousal"]
    for name, valence, arousal in (("a.png", 2, 1.0), ("b.png", -2, 3.0), ("c.png", 1, 2.0)):
        (folder / name).write_bytes(b"x")
        lines.append(f"{name},{valence},{arousal}")
    lines.append("missing.png,1,1.0")
    (tmp_path / "data" / "set" / "records.csv").write_text("\n".join(lines), encoding="utf-8")
    config = copy.deepcopy(CONFIG)
    config["datasets"] = [{"images": "set/images", "metadata": "set/records.csv"}]
    app_config = AppConfig.from_dict(config, tmp_path)
    assert app_config.n_trial_groups == 2
    dataset = load_dataset(app_config)
    assert {g: sorted(i.path.name for i in v) for g, v in dataset.groups.items()} == \
        {"valence=negative": ["b.png"], "valence=positive": ["a.png", "c.png"]}
    assert dataset.rows_without_image == ["missing.png"]
    config["balance_by"] = None
    dataset = load_dataset(AppConfig.from_dict(config, tmp_path))
    assert list(dataset.groups) == ["set/images"]
    config["balance_by"] = "nonsense"
    with pytest.raises(ValueError, match="balance_by"):
        AppConfig.from_dict(config, tmp_path)


def _write_dataset(root: Path, rows: dict[str, list[tuple[str, int, float]]]):
    for folder, items in rows.items():
        path = root / "data" / folder
        path.mkdir(parents=True)
        lines = ["image_name,is_positive,valence,arousal"]
        for name, positive, aroused in items:
            (path / name).write_bytes(b"x")
            lines.append(f"{name},{positive},{3 if positive else -3},{aroused}")
        path.with_suffix(".csv").write_text("\n".join(lines), encoding="utf-8")


def test_dataset_folders_are_a_parameter_and_constant_targets_are_rejected(tmp_path):
    _write_dataset(tmp_path, {"a/pos": [("p1.png", 1, 2.0), ("p2.png", 1, 5.0)],
                              "a/neg": [("n1.png", 0, 1.0)]})
    config = copy.deepcopy(CONFIG)
    config["datasets"] = ["a/pos", "a/neg"]
    dataset = load_dataset(AppConfig.from_dict(config, tmp_path))
    assert {i.path.name: i.labels["arousal"] for g in dataset.groups.values() for i in g} == \
        {"p1.png": 0, "p2.png": 1, "n1.png": 0}
    assert {i.path.name: i.values["valence"] for g in dataset.groups.values() for i in g} == \
        {"p1.png": 3, "p2.png": 3, "n1.png": -3}
    config["targets"] = {"arousal": {"column": "is_positive", "threshold": None}}
    config["balance_by"] = None
    config["datasets"] = ["a/pos"]
    with pytest.raises(ValueError, match="same label"):
        load_dataset(AppConfig.from_dict(config, tmp_path))


def _images(tmp_path, n=3):
    return {group: [StimulusImage(tmp_path / f"{group}{i}.png", group, {"valence": int(group == "joy")})
                    for i in range(n)] for group in ("joy", "sadness")}


def test_finite_plan_balances_and_uses_images_before_repeating(tmp_path):
    groups = _images(tmp_path)
    plan = TrialPlan(groups, 20, random.Random(42))
    trials = [plan.next() for _ in range(20)]
    assert plan.next() is None
    assert sum(t.group == "joy" for t in trials) == 10
    for group in groups:
        first_three = [t.path for t in trials if t.group == group][:3]
        assert len(set(first_three)) == 3
    with pytest.raises(ValueError, match="multiple of 2"):
        TrialPlan(groups, 21)


def test_features_follow_config():
    extractor = FeatureExtractor(EEG)
    spec = extractor.epoch_spec(1000)
    features = extractor.extract(np.random.default_rng(3).normal(0, 4, (spec.n_samples, N_CH)), spec)
    # 8 electrodes x 3 bands of log power + 8 per-electrode fractal dimensions
    assert features.shape == (len(extractor.names),) == (32,)
    assert len(extractor.indices("band_power", band="gamma")) == 8
    assert [extractor.info[i].band for i in extractor.indices("band_power", location="Oz")] == ["alpha", "beta", "gamma"]
    assert [extractor.info[i].location for i in extractor.indices("fractal_dimension")] == list(EEG.electrodes)
    with pytest.raises(ValueError):
        extractor.epoch_spec(600)


def test_optional_feature_families():
    eeg = copy.deepcopy(CONFIG["eeg"])
    eeg["features"] = ["band_power", "fractal_dimension", "region_band_power", "asymmetry", "erp"]
    extractor = FeatureExtractor(EEGSettings.from_dict(eeg))
    spec = extractor.epoch_spec(1000)
    features = extractor.extract(np.random.default_rng(5).normal(0, 4, (spec.n_samples, N_CH)), spec)
    # + 4 regions x 3 bands + 2 pairs x 3 bands of asymmetry + 3 ERP windows x 4 regions
    assert features.shape == (32 + 12 + 6 + 12,)
    # A single-electrode region averages the same spectrum as that electrode.
    fz = extractor.indices("band_power", location="Fz")
    frontal = extractor.indices("region_band_power", location="frontal")
    assert np.allclose(features[fz], features[frontal])
    eeg["features"] = ["wavelets"]
    with pytest.raises(ValueError, match="wavelets"):
        EEGSettings.from_dict(eeg)


def test_higuchi_fractal_dimension_separates_smooth_and_noisy_signals():
    t = np.arange(500) / 250
    assert higuchi_fd(np.sin(2 * np.pi * 2 * t)) < 1.2
    assert higuchi_fd(np.random.default_rng(0).normal(size=500)) > 1.8


@pytest.mark.parametrize("classifiers", [["logreg"], ["svm"], ["logreg", "svm", "svm_linear"]])
def test_separate_models_per_target_and_classifier(tmp_path, classifiers):
    rng = np.random.default_rng(3)
    n = 24
    trainer = MultiTrainer(["valence", "arousal"], classifiers, n)
    for number in range(40):
        labels = {"valence": number % 2, "arousal": (number // 2) % 2}
        x = rng.normal(0, .3, n)
        x[0] += 3 * labels["valence"]
        x[1] += 3 * labels["arousal"]
        trainer.add(x, labels)
    assert set(trainer.models) == {(t, c) for t in ("valence", "arousal") for c in classifiers}
    for (target, clf), model in trainer.models.items():
        assert model.model is not None
        # Sigmoid calibration optimizes probabilities and can move the 0.5
        # boundary on this small synthetic sample.
        assert model.rolling_accuracy >= (70 if clf == "svm" else 80), (target, clf, model.rolling_accuracy)
        assert all(p.probability is not None and 0 < p.probability < 1 and
                   p.label == int(p.probability >= .5) for p in model.predictions)
    written = trainer.save(tmp_path / "run", {"participant": "test"})
    assert written[0].suffix == ".json" and len(written) == 1 + 2 * len(classifiers)
    probe = np.zeros(n)
    probe[0] = 3
    model_path = tmp_path / f"run.valence.{classifiers[0]}.joblib"
    saved_probability = predict_saved(model_path, probe)
    assert .5 < saved_probability < 1
    assert np.isclose(saved_probability, trainer.models[("valence", classifiers[0])].model.predict_proba(probe[None, :])[0, 1])


def test_forest_prequential_probabilities_and_selective_save(tmp_path):
    rng = np.random.default_rng(7)
    trainer = MultiTrainer(["valence"], ["logreg", "svm", "forest"], 20)
    for number in range(36):
        label = number % 2
        features = rng.normal(0, .4, 20)
        features[0] += 2 * label
        predictions = trainer.add(features, {"valence": label})
        for key, prediction in predictions.items():
            assert len(trainer.models[key].x) == number + 1
            assert 0 <= prediction.probability <= 1
            assert prediction.label == int(prediction.probability >= .5)
    assert trainer.models[("valence", "forest")].model is not None
    selection = {("valence", "forest")}
    written = trainer.save(tmp_path / "run", {"participant": "test"}, selection)
    assert {path.name for path in written} == {"run.json", "run.valence.forest.joblib"}
    manifest = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert manifest["saved_models"] == ["valence/forest"]
    assert manifest["models"]["valence/forest"]["selected_for_save"] is True
    assert manifest["models"]["valence/logreg"]["model_file"] is None
    assert manifest["models"]["valence/svm"]["model_file"] is None
    probe = np.zeros(20)
    probability = predict_saved(tmp_path / "run.valence.forest.joblib", probe)
    assert 0 <= probability <= 1
    assert np.isclose(probability, trainer.models[("valence", "forest")].model.predict_proba(probe[None, :])[0, 1])
    with pytest.raises(ValueError, match="Select at least one"):
        trainer.save(tmp_path / "empty", {}, set())
    with pytest.raises(ValueError, match="Unknown model"):
        trainer.save(tmp_path / "unknown", {}, {("valence", "missing")})
    assert not (tmp_path / "empty.json").exists()
    assert not (tmp_path / "unknown.json").exists()


def test_bursty_frame_arrivals_preserve_a_complete_epoch(monkeypatch):
    ring = SampleRing(channels=EEG.device_channels)
    callback_times = iter(100.012 + .016 * i + (.03 if i % 2 else -.02) for i in range(65))
    monkeypatch.setattr("hotbrain.acquisition.time.perf_counter", lambda: next(callback_times))
    signal = np.tile(np.arange(260, dtype=float)[:, None], (1, 8))
    marker_code = 101
    for first in range(0, 260, 4):
        frame = np.column_stack((signal[first:first + 4], np.zeros(4)))
        if first <= 75 < first + len(frame):
            frame[75 - first, -1] = marker_code
        ring.append(frame, marker_channel=8)
    spec = EpochSpec()
    epoch = ring.epoch(marker_code, spec)
    assert epoch is not None
    assert epoch.shape == (spec.n_samples, N_CH)
    assert ring.marker_sample(marker_code) == 75
    assert ring.eeg_onset_sample(marker_code) == 85
    assert np.array_equal(epoch[:, 0], np.arange(35, 260))
    assert ring.eeg_onset_sample(marker_code, 8) == 83
    corrected = ring.epoch(marker_code, spec, paint_to_marker_ms=8)
    assert np.array_equal(corrected[:, 0], np.arange(33, 258))
    assert np.allclose(np.diff(ring.recent(2)[0]), 1 / 250)


def test_ring_keeps_only_configured_device_channels(monkeypatch):
    ring = SampleRing(channels=(7, 0))
    monkeypatch.setattr("hotbrain.acquisition.time.perf_counter", lambda: 104.0)
    frame = np.tile(np.arange(8, dtype=float), (1000, 1))
    frame = np.column_stack((frame, np.zeros(len(frame))))
    frame[300, -1] = 17
    ring.append(frame, marker_channel=8)
    spec = EpochSpec(2000)
    assert spec.end_ms == 1700 and spec.n_samples == 475
    epoch = ring.epoch(17, spec)
    assert epoch is not None
    assert epoch.shape == (475, 2)
    assert np.all(epoch[:, 0] == 7) and np.all(epoch[:, 1] == 0)


def test_epoch_requires_its_own_marker_and_complete_sample_window(monkeypatch):
    ring = SampleRing(channels=EEG.device_channels)
    monkeypatch.setattr("hotbrain.acquisition.time.perf_counter", lambda: 100.0)
    frame = np.column_stack((np.tile(np.arange(8, dtype=float), (100, 1)), np.zeros(100)))
    frame[60, -1] = 41
    ring.append(frame, marker_channel=8)
    assert ring.epoch(42, EpochSpec()) is None
    assert ring.epoch(41, EpochSpec()) is None
    more = np.column_stack((np.tile(np.arange(8, dtype=float), (200, 1)), np.zeros(200)))
    ring.append(more, marker_channel=8)
    assert ring.epoch(41, EpochSpec()) is not None


def test_marker_codes_do_not_reuse_old_trial_when_a_new_marker_is_pending(monkeypatch):
    ring = SampleRing(channels=EEG.device_channels)
    monkeypatch.setattr("hotbrain.acquisition.time.perf_counter", lambda: 100.0)
    frame = np.zeros((260, 9))
    frame[70, -1] = 1
    ring.append(frame, marker_channel=8)
    assert ring.epoch(2, EpochSpec()) is None
    assert ring.epoch(1, EpochSpec()) is not None


def test_fabricated_unicorn_scans_are_never_used(monkeypatch):
    ring = SampleRing(channels=EEG.device_channels)
    monkeypatch.setattr("hotbrain.acquisition.time.perf_counter", lambda: 100.0)
    frame = np.zeros((300, 12))
    frame[:, 10] = 1  # Unicorn validation flag: 1 = real scan.
    frame[75, 11] = 23
    frame[230, 10] = 0  # The driver filled one lost scan.
    ring.append(frame, marker_channel=11, validation_channel=10)
    assert ring.marker_sample(23) == 75
    assert ring.epoch(23, EpochSpec()) is None
    assert np.isnan(ring.recent(1)[1][180]).all()
    assert not ring.valid_recent(.5)
