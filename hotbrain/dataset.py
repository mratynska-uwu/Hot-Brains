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
    labels: dict[str, int] = field(hash=False, compare=False)
    values: dict[str, float] = field(default_factory=dict, hash=False, compare=False)


@dataclass
class Dataset:
    groups: dict[str, list[StimulusImage]]
    targets: tuple[TargetSpec, ...]
    thresholds: dict[str, float | None]

    def class_name(self, target: str, label: int) -> str:
        return next(t for t in self.targets if t.name == target).classes[label]

    def summary(self) -> str:
        images = [i for items in self.groups.values() for i in items]
        parts = [", ".join(f"{g}: {len(v)}" for g, v in self.groups.items())]
        for t in self.targets:
            ones = sum(i.labels[t.name] for i in images)
            rule = t.formula if self.thresholds[t.name] is None else f"{t.formula} > {self.thresholds[t.name]:.3g}"
            parts.append(f"{t.name} ({rule}): {t.classes[0]} {len(images) - ones} / {t.classes[1]} {ones}")
        return " · ".join(parts)

    def manifest(self) -> dict:
        return {"groups": {g: [i.path.name for i in v] for g, v in self.groups.items()},
                "targets": {t.name: {"value": t.formula, "columns": dict(t.columns),
                                     "threshold": self.thresholds[t.name],
                                     "classes": list(t.classes)} for t in self.targets}}


def _read_metadata(csv_path: Path, image_column: str) -> dict[str, dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if image_column not in (reader.fieldnames or ()):
            raise ValueError(f"{csv_path.name} has no {image_column!r} column")
        return {row[image_column]: row for row in reader}


def load_dataset(config: AppConfig) -> Dataset:
    rows: dict[Path, tuple[str, dict[str, str]]] = {}
    groups: dict[str, list[Path]] = {}
    for folder_name in config.datasets:
        folder = config.data_dir / folder_name
        csv_path = folder.with_suffix(".csv")
        if not folder.is_dir():
            raise ValueError(f"Dataset folder not found: {folder}")
        if not csv_path.is_file():
            raise ValueError(f"Missing metadata CSV next to the folder: {csv_path}")
        metadata = _read_metadata(csv_path, config.image_column)
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise ValueError(f"No images in {folder}")
        missing = [p.name for p in paths if p.name not in metadata]
        if missing:
            raise ValueError(f"{len(missing)} image(s) in {folder_name} have no row in {csv_path.name}, e.g. {missing[:3]}")
        groups[folder_name] = paths
        for p in paths:
            rows[p] = (folder_name, metadata[p.name])

    thresholds: dict[str, float | None] = {}
    labels: dict[Path, dict[str, int]] = {p: {} for p in rows}
    scores: dict[Path, dict[str, float]] = {p: {} for p in rows}
    for target in config.targets:
        try:
            values = {p: target.value(row) for p, (_, row) in rows.items()}
        except KeyError as exc:
            raise ValueError(f"Target {target.name!r}: column {exc} missing from a metadata CSV") from None
        except ValueError:
            raise ValueError(f"Target {target.name!r}: {target.formula} has non-numeric values") from None
        threshold = statistics.median(values.values()) if target.threshold == "median" else target.threshold
        thresholds[target.name] = threshold
        for p, v in values.items():
            scores[p][target.name] = v
            if threshold is None:
                if v not in (0, 1):
                    raise ValueError(f"Target {target.name!r}: {target.formula} is not 0/1; set a threshold")
                labels[p][target.name] = int(v)
            else:
                labels[p][target.name] = int(v > threshold)
        if len({labels[p][target.name] for p in rows}) < 2:
            raise ValueError(f"Target {target.name!r}: every image has the same label from {target.formula}, "
                             "so it cannot be classified. Pick another column or threshold.")
    return Dataset({g: [StimulusImage(p, g, labels[p], scores[p]) for p in paths] for g, paths in groups.items()},
                   config.targets, thresholds)
