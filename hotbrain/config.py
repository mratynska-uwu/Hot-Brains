"""Single place to configure datasets, EEG montage/filters, targets and classifiers.

Edit ``CONFIG`` (or pass your own dict to ``hotbrain.gui.launch``). The EEG
part mirrors ``basic_info.md``. ``AppConfig.from_dict`` validates everything
up front so a typo fails at startup rather than mid-session.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG: dict[str, Any] = {
    # Root folder for stimuli, relative to the project directory.
    "data_dir": "data",
    # Subfolders of data_dir with images. Each folder needs a metadata CSV next
    # to it with the same name, e.g. data/training/joy -> data/training/joy.csv.
    # Trials are balanced across these folders.
    "datasets": ["training/joy", "training/sadness"],
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
        "smoothing": 0.3,    # exponential smoothing weight of each new estimate (1 = none)
    },
    "eeg": {
        "sampling_rate": 250,
        "electrodes": ["Fz", "C3", "Cz", "C4", "P3", "Pz", "P4", "Oz"],
        # Unicorn EEG channel (0-7) wired to each electrode above, in order.
        # None = the first len(electrodes) channels in order.
        "device_channels": None,
        "bandpass_hz": [1, 48],
        "notch_hz": 50,  # None disables the notch filter
        "bands_hz": {"alpha": [8, 13], "beta": [13, 30], "gamma": [30, 45]},
        # Feature families, computed in this order:
        #   "band_power"        log power per electrode x band
        #   "fractal_dimension" Higuchi fractal dimension per electrode
        #   "region_band_power" log power averaged over each region x band
        #   "asymmetry"         right-minus-left log power per pair x band
        #   "erp"               mean ERP amplitude per region x window
        "features": ["band_power", "fractal_dimension"],
        "fractal_kmax": 10,
        # Regions used by "region_band_power" and "erp".
        "regions": {"frontal": ["Fz"], "central": ["C3", "Cz", "C4"],
                    "parietal": ["P3", "Pz", "P4"], "occipital": ["Oz"]},
        # (right, left) pairs used by "asymmetry".
        "asymmetry_pairs": [["C4", "C3"], ["P4", "P3"]],
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
        "arousal": {"column": "arousal", "threshold": "median", "classes": ["low", "high"]},
    },
    # Keys from hotbrain.classifiers.CLASSIFIERS selected by default in the GUI.
    "classifiers": ["logreg", "svm", "forest"],
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

    UNICORN_CHANNELS = 8
    FEATURE_FAMILIES = ("band_power", "fractal_dimension", "region_band_power", "asymmetry", "erp")

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
        return cls(fs, electrodes, channels, (low, high), None if notch is None else float(notch),
                   bands, regions, pairs, windows, features, kmax)

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
        return cls(str(name), columns, threshold, classes)

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

    @classmethod
    def from_dict(cls, d: dict, root: Path) -> "AppConfig":
        from .classifiers import CLASSIFIERS

        datasets = tuple(str(p) for p in d["datasets"])
        if not datasets:
            raise ValueError("datasets must list at least one subfolder of data_dir")
        targets = tuple(TargetSpec.from_dict(name, t) for name, t in d["targets"].items())
        if not targets:
            raise ValueError("targets must define at least one classification target")
        classifiers = tuple(d.get("classifiers", ("logreg",)))
        unknown = [c for c in classifiers if c not in CLASSIFIERS]
        if unknown:
            raise ValueError(f"Unknown classifiers {unknown}; available: {sorted(CLASSIFIERS)}")
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
        return cls(data_dir if data_dir.is_absolute() else root / data_dir, datasets,
                   str(d.get("image_column", "image_name")), EEGSettings.from_dict(d["eeg"]),
                   targets, classifiers, d, fraction, int(split.get("seed", 42)),
                   tuple(str(p) for p in d.get("test_datasets", ("test",))), trial_s, rest_s, update_s, smoothing)
