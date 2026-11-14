"""Stimulus datasets: image folders under data/ with per-image metadata CSVs.

Each configured folder (e.g. ``training/joy``) needs a sibling CSV with the
same name (``training/joy.csv``) containing one row per image. Target labels
(valence, arousal, ...) are derived from CSV columns as set in the config.
"""
from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig, TargetSpec

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}


@dataclass(frozen=True)
class StimulusImage:
    path: Path
    group: str
    # 0/1 per target, or None when the image is left out of that target's training
    # (exclude_middle: too close to the class boundary).
    labels: dict[str, int | None] = field(hash=False, compare=False)
    values: dict[str, float] = field(default_factory=dict, hash=False, compare=False)


@dataclass
class Dataset:
    groups: dict[str, list[StimulusImage]]
    targets: tuple[TargetSpec, ...]
    thresholds: dict[str, float | None]
    # Half-range of each target around its threshold, used to map values to [-1, 1].
    scales: dict[str, float] = field(default_factory=dict)
    # CSV rows whose image file is missing from the folder (they are skipped).
    rows_without_image: list[str] = field(default_factory=list)
    # Per target: images with |value - threshold| below this are excluded (label None).
    margins: dict[str, float] = field(default_factory=dict)
    # Which images the thresholds/margins were computed from ("all images" or "training split").
    threshold_source: str = "all images"

    @property
    def images(self) -> list[StimulusImage]:
        return [i for items in self.groups.values() for i in items]

    def normalized(self, target: str, value: float) -> float:
        """Map a target value to [-1, 1]: 0 = class boundary, +-1 = most extreme training image."""
        threshold = self.thresholds[target]
        if threshold is None:
            return float(max(-1.0, min(1.0, 2 * value - 1)))
        return float(max(-1.0, min(1.0, (value - threshold) / self.scales.get(target, 1.0))))

    def class_name(self, target: str, label: int | None) -> str:
        if label is None:
            return "excluded"
        return next(t for t in self.targets if t.name == target).classes[label]

    def summary(self) -> str:
        images = [i for items in self.groups.values() for i in items]
        parts = [", ".join(f"{g}: {len(v)}" for g, v in self.groups.items())]
        for t in self.targets:
            labelled = [i.labels[t.name] for i in images if i.labels[t.name] is not None]
            ones = sum(labelled)
            rule = t.formula if self.thresholds[t.name] is None else f"{t.formula} > {self.thresholds[t.name]:.3g}"
            excluded = len(images) - len(labelled)
            parts.append(f"{t.name} ({rule}): {t.classes[0]} {len(labelled) - ones} / {t.classes[1]} {ones}"
                         + (f" / {excluded} excluded (middle {t.exclude_middle:.0%})" if excluded else ""))
        if self.rows_without_image:
            parts.append(f"{len(self.rows_without_image)} CSV row(s) without an image file skipped "
                         f"(e.g. {', '.join(self.rows_without_image[:3])})")
        return " · ".join(parts)

    def manifest(self) -> dict:
        return {"groups": {g: [i.path.name for i in v] for g, v in self.groups.items()},
                "targets": {t.name: {"value": t.formula, "columns": dict(t.columns),
                                     "threshold": self.thresholds[t.name], "scale": self.scales.get(t.name),
                                     "exclusion_margin": self.margins.get(t.name),
                                     "threshold_source": self.threshold_source,
                                     "classes": list(t.classes)} for t in self.targets}}


def _read_metadata(csv_path: Path, image_column: str) -> dict[str, dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if image_column not in (reader.fieldnames or ()):
            raise ValueError(f"{csv_path.name} has no {image_column!r} column")
        return {row[image_column]: row for row in reader}


def load_dataset(config: AppConfig, folders: tuple[str, ...] | list[str] | None = None,
                 reference: Dataset | None = None) -> Dataset:
    """Load image folders (default: ``config.datasets``).

    With ``reference`` (e.g. the training dataset when loading test images),
    thresholds and scales are taken from it instead of being recomputed, and a
    target may have a single class.
    """
    rows: dict[Path, tuple[str, dict[str, str]]] = {}
    groups: dict[str, list[Path]] = {}
    rows_without_image: list[str] = []
    for folder_name in (config.datasets if folders is None else folders):
        folder = config.data_dir / folder_name
        csv_path = config.metadata_path(folder_name)
        if not folder.is_dir():
            raise ValueError(f"Dataset folder not found: {folder}")
        if not csv_path.is_file():
            raise ValueError(f"Missing metadata CSV for {folder_name}: {csv_path}")
        metadata = _read_metadata(csv_path, config.image_column)
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise ValueError(f"No images in {folder}")
        missing = [p.name for p in paths if p.name not in metadata]
        if missing:
            raise ValueError(f"{len(missing)} image(s) in {folder_name} have no row in {csv_path.name}, e.g. {missing[:3]}")
        on_disk = {p.name for p in paths}
        rows_without_image += sorted(name for name in metadata if name not in on_disk)
        groups[folder_name] = paths
        for p in paths:
            rows[p] = (folder_name, metadata[p.name])

    scores: dict[Path, dict[str, float]] = {p: {} for p in rows}
    for target in config.targets:
        for p, (_, row) in rows.items():
            try:
                scores[p][target.name] = target.value(row)
            except KeyError as exc:
                raise ValueError(f"Target {target.name!r}: column {exc} missing from a metadata CSV") from None
            except ValueError:
                raise ValueError(f"Target {target.name!r}: {target.formula} has non-numeric values") from None
    if reference is not None:
        thresholds, margins, scales = dict(reference.thresholds), dict(reference.margins), dict(reference.scales)
    else:
        thresholds, margins, scales = _fit_thresholds(config.targets, list(scores.values()))
    labels = {p: _labels(config.targets, scores[p], thresholds, margins) for p in rows}
    if reference is None:
        for target in config.targets:
            if len({labels[p][target.name] for p in rows} - {None}) < 2:
                raise ValueError(f"Target {target.name!r}: every image has the same label from {target.formula}, "
                                 "so it cannot be classified. Pick another column or threshold.")
    images = [StimulusImage(p, group, labels[p], scores[p]) for group, paths in groups.items() for p in paths]
    return Dataset(_group(config, images, groups), config.targets, thresholds, scales, rows_without_image, margins,
                   reference.threshold_source if reference is not None else "all images")


def _fit_thresholds(targets, values: list[dict[str, float]]):
    """Thresholds ("median" -> median of the given images), exclusion margins and display scales."""
    thresholds: dict[str, float | None] = {}
    margins: dict[str, float] = {}
    scales: dict[str, float] = {}
    for target in targets:
        column = [v[target.name] for v in values]
        threshold = statistics.median(column) if target.threshold == "median" else target.threshold
        thresholds[target.name] = threshold
        if threshold is None:
            continue
        distances = sorted(abs(v - threshold) for v in column)
        scales[target.name] = (distances[-1] if distances else 1.0) or 1.0
        if target.exclude_middle and distances:
            # Leave out the share of images closest to the boundary (the middle tercile for 1/3).
            margins[target.name] = float(distances[min(len(distances) - 1, int(len(distances) * target.exclude_middle))])
    return thresholds, margins, scales


def _labels(targets, values: dict[str, float], thresholds, margins) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for target in targets:
        v, threshold = values[target.name], thresholds[target.name]
        if threshold is None:
            if v not in (0, 1):
                raise ValueError(f"Target {target.name!r}: {target.formula} is not 0/1; set a threshold")
            out[target.name] = int(v)
        elif abs(v - threshold) < margins.get(target.name, 0.0):
            out[target.name] = None
        else:
            out[target.name] = int(v > threshold)
    return out


def _group(config: AppConfig, images: list[StimulusImage], folders: dict[str, list[Path]]):
    """Folder groups, or classes of the balance_by target (e.g. "valence=negative")."""
    if not config.balance_by:
        by_path = {i.path: i for i in images}
        return {g: [by_path[p] for p in paths] for g, paths in folders.items()}
    target = next(t for t in config.targets if t.name == config.balance_by)
    regrouped: dict[str, list[StimulusImage]] = {}
    for image in images:
        label = image.labels[target.name]
        name = f"{target.name}={'excluded' if label is None else target.classes[label]}"
        regrouped.setdefault(name, []).append(
            StimulusImage(image.path, name, image.labels, image.values))
    return {g: sorted(v, key=lambda i: i.path) for g, v in sorted(regrouped.items())}


def rethreshold(config: AppConfig, dataset: Dataset, reference_images: list[StimulusImage]) -> Dataset:
    """Recompute data-driven thresholds/margins from ``reference_images`` (the training split) and relabel.

    Avoids leaking the held-out images' ratings into the class boundary. Fixed
    thresholds (e.g. valence > 0) are unaffected.
    """
    thresholds, margins, scales = _fit_thresholds(config.targets, [i.values for i in reference_images])
    images = [StimulusImage(i.path, i.group, _labels(config.targets, i.values, thresholds, margins), i.values)
              for items in dataset.groups.values() for i in items]
    folders: dict[str, list[Path]] = {}
    for items in dataset.groups.values():
        for i in items:
            folders.setdefault(i.group, []).append(i.path)
    return Dataset(_group(config, images, folders), dataset.targets, thresholds, scales,
                   dataset.rows_without_image, margins, "training split")
