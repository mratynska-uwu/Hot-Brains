"""Single place to configure datasets, EEG montage/filters, targets and classifiers.

Edit ``CONFIG`` (or pass your own dict to ``hotbrain.gui.launch``). The EEG
part mirrors ``basic_info.md``. ``AppConfig.from_dict`` validates everything
up front so a typo fails at startup rather than mid-session.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

CONFIG: dict[str, Any] = {
    # Root folder for stimuli, relative to the project directory.
    "data_dir": "data",
    # Image folders under data_dir. Each entry is either
    #   "sub/folder"  -> metadata CSV next to it with the same name (sub/folder.csv), or
    #   {"images": "sub/folder", "metadata": "sub/records.csv"} for any CSV name.
    # The CSV needs one row per image; extra rows without an image file are reported.
    "datasets": [{"images": "training/images", "metadata": "training/records.csv"}],
    # Training trials are balanced across these groups and the 80/20 split is
    # stratified by them: a target name (e.g. "valence" = its two classes) or
    # None to balance across the dataset folders.
    "balance_by": "valence",
    # Column in the metadata CSV that holds the image file name.
    "image_column": "image_name",
    # Training/validation split of the datasets above, stratified per folder
    # and reproducible from the seed. Training sessions use only the training
    # part; validation sessions show each validation image once.
    "split": {"validation_fraction": 0.2, "seed": 42},
    # Subfolders of data_dir for the test session (same CSV convention; the CSV
    # needs the target columns, e.g. valence and arousal). Not available yet.
    "test_datasets": ["test"],
    "test_session": {
        "trial_s": 15.0,     # how long each test image (and its target) is shown
        "rest_s": 5.0,       # fixation between test images
    },
    # Live valence/arousal decoding shown during validation and test sessions.
    "feedback": {
        "update_s": 0.25,    # continuous decoding interval in the test session
        "smoothing": 0.1,    # exponential smoothing weight of each new estimate (1 = none)
        "on_target_distance": 0.35,  # plane distance counted as "on target" (beep + check mark)
        "practice_images": 3,        # images in a practice test run (not a measurement)
    },
    # Calibration session for goal 2: the participant holds the image's emotion
    # for a whole block; models learn from sliding windows of that sustained
    # state (instead of the short evoked response) and drive the test session.
    "calibration": {
        "blocks": 8,          # images (blocks), balanced across the balance_by groups
        "block_s": 20.0,      # how long the participant holds each state
        "rest_s": 5.0,        # fixation between blocks
        "window_s": 2.0,      # sliding analysis window
        "step_s": 1.0,        # window step
        "skip_s": 2.0,        # ignore the first seconds of each block (orienting response)
        "candidates_per_group": 10,  # most extreme images per group to choose blocks from
    },
    # Eyes-open resting recording, saved as the participant's individual reference.
    "resting": {"duration_s": 90.0},
    # Optional automatic stop of a training run once a model is good enough.
    "stop_rule": {"enabled": False, "target": "valence", "balanced_accuracy": 65.0, "min_predictions": 40},
    # Training-quality report written after every training / calibration run.
    "report": {"cv_folds": 5, "permutations": 100, "label_issue_threshold": 0.2, "seed": 0},
    # Two selectable feature configurations (GUI "Configuration"). Each target
    # has its own electrodes, bands and feature families; electrodes may be
    # "all" and are further limited by the GUI's model-electrode checkboxes.
    #   window_ms   analysis window after image onset; None = until 300 ms
    #               before the image disappears
    #   baseline_ms pre-onset interval subtracted per channel
    "profiles": {
        1: {"name": "Full spectrum · all electrodes",
            "window_ms": None,
            "baseline_ms": [-500, -200],
            "targets": {
                "valence": {"electrodes": "all", "bands": ["theta", "alpha", "beta", "gamma"],
                            "features": ["band_power", "fractal_dimension", "asymmetry"]},
                "arousal": {"electrodes": "all", "bands": ["theta", "alpha", "beta", "gamma"],
                            "features": ["band_power", "fractal_dimension", "beta_alpha_ratio"]},
            }},
        2: {"name": "450 ms · arousal: occipital alpha · valence: fronto-central beta/gamma",
            "window_ms": 450,
            "baseline_ms": [-500, -200],
            "targets": {
                "arousal": {"electrodes": ["PO7", "Oz", "PO8"], "bands": ["alpha"], "features": ["band_power"]},
                "valence": {"electrodes": ["Fz", "C3", "Cz", "C4"], "bands": ["beta", "gamma"],
                            "features": ["band_power"]},
            }},
    },
    "default_profile": 1,
    "eeg": {
        "sampling_rate": 250,
        # Confirmed physical Unicorn Hybrid Black cap: clips 1-8 in this order.
        "electrodes": ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"],
        # Unicorn EEG channel (0-7) wired to each electrode above, in order.
        # None = the first len(electrodes) channels in order.
        "device_channels": None,
        "bandpass_hz": [1, 48],
        "notch_hz": 50,  # None disables the notch filter
        "bands_hz": {"theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 45]},
        # Spectral estimate: "welch" (averaged, steadier for short windows) or "fft".
        "psd_method": "welch",
        # Re-reference before feature extraction: None (device reference) or "car"
        # (common average of the eight channels).
        "reference": None,
        # Reject an epoch when any model channel exceeds this absolute amplitude
        # after filtering and baseline correction (dry electrodes may need more).
        "artifact_max_abs_uv": 150.0,
        # Feature families (used by the live feature plot; models use the profiles below):
        #   "band_power"        log power per electrode x band
        #   "fractal_dimension" Higuchi fractal dimension per electrode
        #   "region_band_power" log power averaged over each region x band
        #   "asymmetry"         right-minus-left log power per pair x band
        #   "beta_alpha_ratio"  log(beta) - log(alpha) per electrode (arousal index)
        #   "hjorth"            Hjorth mobility and complexity per electrode
        #   "katz_fd"           Katz fractal dimension per electrode
        #   "petrosian_fd"      Petrosian fractal dimension per electrode
        #   "variance"          log variance per electrode
        #   "erp"               mean ERP amplitude per region x window
        "features": ["band_power", "fractal_dimension"],
        "fractal_kmax": 10,
        # Regions used by "region_band_power" and "erp".
        "regions": {"frontal": ["Fz"], "central": ["C3", "Cz", "C4"],
                    "parietal": ["Pz"], "posterior": ["PO7", "Oz", "PO8"]},
        # (right, left) pairs used by "asymmetry".
        "asymmetry_pairs": [["C4", "C3"], ["PO8", "PO7"]],
        # Windows after image onset used by "erp"; None = end of the epoch.
        "erp_windows_ms": [[100, 200], [200, 350], [350, None]],
    },
    # What to classify. Each target gets one value per image, either a single
    # CSV column ("column") or a weighted sum of columns ("columns": {name:
    # weight}), and a binary label from it: threshold None means the value is
    # already 0/1; a number or "median" means label 1 = value above the
    # threshold (median of the loaded dataset). Swap these for any other
    # columns (e.g. approach_mean).
    # The CSVs carry ready-made columns: valence = positive_mean - negative_mean
    # (> 0 means rated more positive than negative) and arousal = aroused_mean.
    # A dataset without them could use {"columns": {"positive_mean": 1,
    # "negative_mean": -1}} instead.
    "targets": {
        "valence": {"column": "valence", "threshold": 0, "classes": ["negative", "positive"]},
        # exclude_middle: drop the middle share of images (by the training images'
        # quantiles) from this target's training, keeping only clearly low/high ones.
        "arousal": {"column": "arousal", "threshold": "median", "classes": ["low", "high"],
                    "exclude_middle": 1 / 3},
    },
    # Keys from hotbrain.classifiers.CLASSIFIERS selected by default in the GUI.
    "classifiers": ["logreg"],
}


@dataclass(frozen=True)
class EEGSettings:
    fs: int
    electrodes: tuple[str, ...]
    device_channels: tuple[int, ...]
    bandpass_hz: tuple[float, float]
    notch_hz: float | None
    bands_hz: tuple[tuple[str, tuple[float, float]], ...]
    regions: tuple[tuple[str, tuple[str, ...]], ...]
    asymmetry_pairs: tuple[tuple[str, str], ...]
    erp_windows_ms: tuple[tuple[int, int | None], ...]
    features: tuple[str, ...] = ("band_power", "fractal_dimension")
    fractal_kmax: int = 10
    psd_method: str = "welch"
    reference: str | None = None
    artifact_max_abs_uv: float | None = 150.0

    UNICORN_CHANNELS = 8
    FEATURE_FAMILIES = ("band_power", "fractal_dimension", "region_band_power", "asymmetry", "erp",
                        "beta_alpha_ratio", "hjorth", "katz_fd", "petrosian_fd", "variance")

    def for_target(self, electrodes: tuple[str, ...], bands: tuple[str, ...],
                   features: tuple[str, ...]) -> "EEGSettings":
        """Settings for one target: a subset of electrodes, bands and feature families."""
        positions = tuple(i for i, name in enumerate(self.electrodes) if name in set(electrodes))
        if not positions:
            raise ValueError("No electrodes left for this target")
        selected = self.select_electrodes(positions)
        band_set = set(bands)
        kept_bands = tuple((name, edges) for name, edges in self.bands_hz if name in band_set)
        if not kept_bands:
            raise ValueError("No bands left for this target")
        families = tuple(f for f in features if f != "asymmetry" or selected.asymmetry_pairs)
        if "beta_alpha_ratio" in families and not {"alpha", "beta"} <= band_set:
            families = tuple(f for f in families if f != "beta_alpha_ratio")
        if not families:
            raise ValueError("No feature families left for this target")
        return replace(selected, bands_hz=kept_bands, features=families)

    def select_electrodes(self, positions: tuple[int, ...]) -> "EEGSettings":
        """Build model settings for a subset while preserving physical channel order."""
        if not positions or len(set(positions)) != len(positions) or any(
                index < 0 or index >= len(self.electrodes) for index in positions):
            raise ValueError("Select at least one distinct configured electrode")
        positions = tuple(sorted(positions))
        electrodes = tuple(self.electrodes[index] for index in positions)
        selected = set(electrodes)
        regions = tuple((name, tuple(member for member in members if member in selected))
                        for name, members in self.regions)
        regions = tuple((name, members) for name, members in regions if members)
        pairs = tuple((right, left) for right, left in self.asymmetry_pairs
                      if right in selected and left in selected)
        features = tuple(family for family in self.features if family != "asymmetry" or pairs)
        return replace(self, electrodes=electrodes,
                       device_channels=tuple(self.device_channels[index] for index in positions),
                       regions=regions, asymmetry_pairs=pairs, features=features)

    @classmethod
    def from_dict(cls, d: dict) -> "EEGSettings":
        electrodes = tuple(str(e) for e in d["electrodes"])
        if not electrodes or len(set(electrodes)) != len(electrodes):
            raise ValueError("eeg.electrodes must be a non-empty list of unique names")
        channels = d.get("device_channels")
        channels = tuple(range(len(electrodes))) if channels is None else tuple(int(c) for c in channels)
        if len(channels) != len(electrodes) or len(set(channels)) != len(channels) \
                or not all(0 <= c < cls.UNICORN_CHANNELS for c in channels):
            raise ValueError(f"eeg.device_channels needs one unique index 0-{cls.UNICORN_CHANNELS - 1} per electrode")
        fs = int(d["sampling_rate"])
        low, high = (float(v) for v in d["bandpass_hz"])
        if not 0 < low < high < fs / 2:
            raise ValueError("eeg.bandpass_hz must satisfy 0 < low < high < sampling_rate/2")
        notch = d.get("notch_hz")
        if notch is not None and not 0 < float(notch) < fs / 2:
            raise ValueError("eeg.notch_hz must be below sampling_rate/2")
        bands = tuple((str(name), (float(lo), float(hi))) for name, (lo, hi) in d["bands_hz"].items())
        for name, (lo, hi) in bands:
            if not 0 <= lo < hi <= fs / 2:
                raise ValueError(f"eeg.bands_hz[{name!r}] must satisfy 0 <= low < high <= sampling_rate/2")
        regions = tuple((str(name), tuple(members)) for name, members in d["regions"].items())
        pairs = tuple((str(r), str(l)) for r, l in d.get("asymmetry_pairs", ()))
        for name in [m for _, members in regions for m in members] + [e for p in pairs for e in p]:
            if name not in electrodes:
                raise ValueError(f"Electrode {name!r} is used in regions/asymmetry_pairs but not listed in eeg.electrodes")
        if not bands or not regions:
            raise ValueError("eeg.bands_hz and eeg.regions must not be empty")
        windows = tuple((int(lo), None if hi is None else int(hi)) for lo, hi in d.get("erp_windows_ms", ()))
        for lo, hi in windows:
            if lo < 0 or (hi is not None and hi <= lo):
                raise ValueError("eeg.erp_windows_ms entries must be [start >= 0, end > start or None]")
        features = tuple(str(f) for f in d.get("features", cls.FEATURE_FAMILIES[:2]))
        unknown = [f for f in features if f not in cls.FEATURE_FAMILIES]
        if not features or unknown or len(set(features)) != len(features):
            raise ValueError(f"eeg.features must be unique names from {cls.FEATURE_FAMILIES}; got {list(features)}")
        if "asymmetry" in features and not pairs:
            raise ValueError("eeg.features includes 'asymmetry' but eeg.asymmetry_pairs is empty")
        if "erp" in features and not windows:
            raise ValueError("eeg.features includes 'erp' but eeg.erp_windows_ms is empty")
        kmax = int(d.get("fractal_kmax", 10))
        if kmax < 2:
            raise ValueError("eeg.fractal_kmax must be at least 2")
        psd = str(d.get("psd_method", "welch"))
        if psd not in ("welch", "fft"):
            raise ValueError("eeg.psd_method must be 'welch' or 'fft'")
        reference = d.get("reference")
        if reference not in (None, "car"):
            raise ValueError("eeg.reference must be None or 'car'")
        threshold = d.get("artifact_max_abs_uv", 150.0)
        threshold = None if threshold is None else float(threshold)
        return cls(fs, electrodes, channels, (low, high), None if notch is None else float(notch),
                   bands, regions, pairs, windows, features, kmax, psd, reference, threshold)

    def warnings(self) -> list[str]:
        low, high = self.bandpass_hz
        out = []
        for name, (lo, hi) in self.bands_hz:
            if hi > high or lo < low:
                out.append(f"Band {name} {lo:g}-{hi:g} Hz lies partly outside the {low:g}-{high:g} Hz bandpass, "
                           "so its power is mostly filter roll-off. Widen bandpass_hz or drop the band.")
        if self.fs != 250:
            out.append(f"sampling_rate {self.fs} Hz: the Unicorn streams at 250 Hz; timing assumes the configured rate.")
        return out


@dataclass(frozen=True)
class TargetSpec:
    name: str
    columns: tuple[tuple[str, float], ...]  # value = sum(weight * column)
    threshold: float | str | None
    classes: tuple[str, str]
    exclude_middle: float = 0.0   # share of images around the threshold left out of training

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "TargetSpec":
        if ("column" in d) == ("columns" in d):
            raise ValueError(f"targets.{name} needs exactly one of 'column' or 'columns'")
        columns = ((str(d["column"]), 1.0),) if "column" in d else \
            tuple((str(c), float(w)) for c, w in d["columns"].items())
        if not columns:
            raise ValueError(f"targets.{name}.columns must not be empty")
        threshold = d.get("threshold")
        if threshold is not None and threshold != "median":
            threshold = float(threshold)
        classes = tuple(str(c) for c in d.get("classes", ("0", "1")))
        if len(classes) != 2:
            raise ValueError(f"targets.{name}.classes must have exactly two names (label 0, label 1)")
        exclude = float(d.get("exclude_middle", 0.0) or 0.0)
        if not 0 <= exclude < 1 or (exclude and threshold is None):
            raise ValueError(f"targets.{name}.exclude_middle must be in [0, 1) and needs a threshold")
        return cls(str(name), columns, threshold, classes, exclude)

    @property
    def formula(self) -> str:
        """Readable definition, e.g. 'positive_mean - negative_mean'."""
        out = ""
        for i, (column, weight) in enumerate(self.columns):
            sign = "-" if weight < 0 else "+"
            factor = "" if abs(weight) == 1 else f"{abs(weight):g}*"
            out += (("-" if sign == "-" else "") if i == 0 else f" {sign} ") + factor + column
        return out

    def value(self, row: dict[str, str]) -> float:
        return sum(weight * float(row[column]) for column, weight in self.columns)


@dataclass(frozen=True)
class TargetFeatureSpec:
    electrodes: tuple[str, ...] | None   # None = all configured electrodes
    bands: tuple[str, ...]
    features: tuple[str, ...]


@dataclass(frozen=True)
class Profile:
    """One selectable feature configuration ("Configuration 1/2" in the GUI)."""
    number: int
    name: str
    window_ms: int | None
    baseline_ms: tuple[int, int]
    targets: tuple[tuple[str, TargetFeatureSpec], ...]

    def target(self, name: str) -> TargetFeatureSpec:
        return dict(self.targets)[name]

    @classmethod
    def from_dict(cls, number: int, d: dict, eeg: EEGSettings, target_names: list[str]) -> "Profile":
        window = d.get("window_ms")
        window = None if window is None else int(window)
        if window is not None and window < 100:
            raise ValueError(f"profiles.{number}.window_ms must be at least 100 ms")
        b0, b1 = (int(v) for v in d.get("baseline_ms", (-200, 0)))
        if not b0 < b1 <= 0:
            raise ValueError(f"profiles.{number}.baseline_ms must be [start < end <= 0]")
        band_names = {name for name, _ in eeg.bands_hz}
        specs = []
        for target in target_names:
            t = d.get("targets", {}).get(target)
            if t is None:
                raise ValueError(f"profiles.{number} has no feature spec for target {target!r}")
            electrodes = t.get("electrodes", "all")
            electrodes = None if electrodes == "all" else tuple(str(e) for e in electrodes)
            if electrodes is not None and (not electrodes or set(electrodes) - set(eeg.electrodes)):
                raise ValueError(f"profiles.{number}.{target}.electrodes must be 'all' or configured electrodes")
            bands = tuple(str(b) for b in t.get("bands", band_names))
            if not bands or set(bands) - band_names:
                raise ValueError(f"profiles.{number}.{target}.bands must name configured bands {sorted(band_names)}")
            features = tuple(str(f) for f in t.get("features", ("band_power",)))
            if not features or set(features) - set(EEGSettings.FEATURE_FAMILIES):
                raise ValueError(f"profiles.{number}.{target}.features must come from {EEGSettings.FEATURE_FAMILIES}")
            specs.append((target, TargetFeatureSpec(electrodes, bands, features)))
        return cls(int(number), str(d.get("name", f"Configuration {number}")), window, (b0, b1), tuple(specs))


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    datasets: tuple[str, ...]
    image_column: str
    eeg: EEGSettings
    targets: tuple[TargetSpec, ...]
    classifiers: tuple[str, ...]
    raw: dict
    validation_fraction: float = 0.2
    split_seed: int = 42
    test_datasets: tuple[str, ...] = ("test",)
    test_trial_s: float = 15.0
    test_rest_s: float = 5.0
    feedback_update_s: float = 0.25
    feedback_smoothing: float = 0.3
    # Explicit metadata CSV per image folder (folders not listed use <folder>.csv).
    dataset_metadata: tuple[tuple[str, str], ...] = ()
    balance_by: str | None = None
    profiles: tuple[Profile, ...] = ()
    default_profile: int = 1
    calibration: dict | None = None
    resting_s: float = 90.0
    stop_rule: dict | None = None
    report: dict | None = None
    on_target_distance: float = 0.35
    practice_images: int = 3

    def profile(self, number: int) -> Profile:
        for profile in self.profiles:
            if profile.number == number:
                return profile
        raise KeyError(f"No configuration {number}")

    @property
    def n_trial_groups(self) -> int:
        """Groups a finite training run is balanced across (the trial count must be a multiple)."""
        return 2 if self.balance_by else len(self.datasets)

    def metadata_path(self, folder: str) -> Path:
        override = dict(self.dataset_metadata).get(folder)
        return self.data_dir / override if override else (self.data_dir / folder).with_suffix(".csv")

    @classmethod
    def from_dict(cls, d: dict, root: Path) -> "AppConfig":
        from .classifiers import CLASSIFIERS

        def parse_sources(entries) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
            folders, metadata = [], []
            for entry in entries:
                if isinstance(entry, dict):
                    if "images" not in entry:
                        raise ValueError("dataset entries given as dicts need an 'images' folder")
                    folders.append(str(entry["images"]))
                    if entry.get("metadata"):
                        metadata.append((str(entry["images"]), str(entry["metadata"])))
                else:
                    folders.append(str(entry))
            return tuple(folders), tuple(metadata)

        datasets, dataset_metadata = parse_sources(d["datasets"])
        if not datasets:
            raise ValueError("datasets must list at least one subfolder of data_dir")
        test_datasets, test_metadata = parse_sources(d.get("test_datasets", ("test",)))
        targets = tuple(TargetSpec.from_dict(name, t) for name, t in d["targets"].items())
        if not targets:
            raise ValueError("targets must define at least one classification target")
        balance_by = d.get("balance_by")
        if balance_by is not None and balance_by not in {t.name for t in targets}:
            raise ValueError(f"balance_by must be a target name or None, got {balance_by!r}")
        classifiers = tuple(d.get("classifiers", ("logreg",)))
        unknown = [c for c in classifiers if c not in CLASSIFIERS]
        if unknown:
            raise ValueError(f"Unknown classifiers {unknown}; available: {sorted(CLASSIFIERS)}")
        if len(classifiers) != len(set(classifiers)) or len(classifiers) > 3:
            raise ValueError("Select at most three distinct default classifiers")
        split = d.get("split", {})
        fraction = float(split.get("validation_fraction", 0.2))
        if not 0 <= fraction < 1:
            raise ValueError("split.validation_fraction must be in [0, 1)")
        test = d.get("test_session", {})
        feedback = d.get("feedback", {})
        trial_s, rest_s = float(test.get("trial_s", 15.0)), float(test.get("rest_s", 5.0))
        update_s, smoothing = float(feedback.get("update_s", 0.25)), float(feedback.get("smoothing", 0.3))
        if trial_s <= 0 or rest_s < 0 or update_s <= 0 or not 0 < smoothing <= 1:
            raise ValueError("test_session/feedback values must be positive (smoothing in (0, 1])")
        data_dir = Path(d.get("data_dir", "data"))
        eeg = EEGSettings.from_dict(d["eeg"])
        target_names = [t.name for t in targets]
        default_profile_dict = {"name": "Default", "window_ms": None, "baseline_ms": [-200, 0],
                                "targets": {n: {"electrodes": "all", "bands": [b for b, _ in eeg.bands_hz],
                                                "features": list(eeg.features)} for n in target_names}}
        raw_profiles = d.get("profiles") or {1: default_profile_dict}
        profiles = tuple(Profile.from_dict(int(k), v, eeg, target_names) for k, v in sorted(raw_profiles.items(),
                                                                                         key=lambda kv: int(kv[0])))
        default_profile = int(d.get("default_profile", profiles[0].number))
        if default_profile not in {p.number for p in profiles}:
            raise ValueError("default_profile must be one of the configured profiles")
        calibration = {"blocks": 8, "block_s": 20.0, "rest_s": 5.0, "window_s": 2.0, "step_s": 1.0,
                       "skip_s": 2.0, "candidates_per_group": 10, **d.get("calibration", {})}
        if calibration["window_s"] + calibration["skip_s"] > calibration["block_s"] or calibration["step_s"] <= 0:
            raise ValueError("calibration needs window_s + skip_s <= block_s and step_s > 0")
        stop_rule = {"enabled": False, "target": target_names[0], "balanced_accuracy": 65.0,
                     "min_predictions": 40, **d.get("stop_rule", {})}
        report = {"cv_folds": 5, "permutations": 100, "label_issue_threshold": 0.2, "seed": 0, **d.get("report", {})}
        return cls(data_dir if data_dir.is_absolute() else root / data_dir, datasets,
                   str(d.get("image_column", "image_name")), eeg,
                   targets, classifiers, d, fraction, int(split.get("seed", 42)),
                   test_datasets, trial_s, rest_s, update_s, smoothing,
                   dataset_metadata + test_metadata, balance_by, profiles, default_profile, calibration,
                   float(d.get("resting", {}).get("duration_s", 90.0)), stop_rule, report,
                   float(feedback.get("on_target_distance", 0.35)), int(feedback.get("practice_images", 3)))
