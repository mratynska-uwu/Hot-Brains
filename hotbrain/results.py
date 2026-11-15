"""Replay saved run metrics without a connected EEG device."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QScrollArea, QVBoxLayout, QWidget


def replay_curves(manifest: dict, window: int = 20) -> dict:
    """Rebuild the displayed prequential curves from logged trial predictions."""
    curves = defaultdict(lambda: {"trial": [], "accuracy": [], "brier": [], "n": 0})
    for trial in manifest.get("trial_log", []):
        role = trial.get("trial_role")
        for model, prediction in trial.get("predictions", {}).items():
            target = model.split("/", 1)[0]
            class1 = manifest.get("config", {}).get("targets", {}).get(target, {}).get("classes", ["0", "1"])[1]
            label = int(trial.get("labels", {}).get(target) == class1)
            p = float(prediction["p_class1"])
            key = (model, role)
            row = curves[key]
            row.setdefault("hits", []).append(float(bool(prediction["correct"])))
            row.setdefault("squared_errors", []).append((p - label) ** 2)
            row["trial"].append(int(trial["number"]))
            row["accuracy"].append(100 * float(np.mean(row["hits"][-window:])))
            row["brier"].append(float(np.mean(row["squared_errors"][-window:])))
            row["n"] += 1
    return dict(curves)


class RunResultsWindow(QMainWindow):
    """Large, scrollable review window; all plots remain visible after saving."""

    COLORS = ("#4dd6b0", "#89aaff", "#ffbb65", "#fb7780", "#c9a3ff", "#f5dc68")

    def __init__(self, path: Path):
        super().__init__()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest.get("trial_log"), list):
            raise ValueError("This file has no saved trial log")
        self.setWindowTitle(f"Hot Brain · saved results · {path.name}")
        self.resize(1200, 900)
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(14)
        title = QLabel(f"Saved run · {path.name}")
        title.setStyleSheet("font-size:20px;font-weight:700")
        layout.addWidget(title)
        summary = QLabel(self._summary(manifest))
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary)
        curves = replay_curves(manifest)
        self.accuracy_plot = self._plot("Rolling correct predictions", "Correct", "%")
        self.brier_plot = self._plot("Rolling Brier loss (lower is better)", "Brier loss", "")
        self.accuracy_plot.setYRange(0, 100)
        self.brier_plot.addItem(pg.InfiniteLine(pos=.25, angle=0,
                                 pen=pg.mkPen("#d8d8d8", width=1, style=Qt.PenStyle.DashLine)))
        self.brier_plot.setYRange(0, .7)
        for index, ((model, role), values) in enumerate(sorted(curves.items())):
            color = self.COLORS[index % len(self.COLORS)]
            label = f"{model} · {'held-out stream' if role == 'validation' else role}"
            self.accuracy_plot.plot(values["trial"], values["accuracy"],
                                    pen=pg.mkPen(color, width=2), name=label)
            self.brier_plot.plot(values["trial"], values["brier"],
                                 pen=pg.mkPen(color, width=2), name=label)
        layout.addWidget(self.accuracy_plot)
        layout.addWidget(self.brier_plot)
        caption = QLabel("Curves show each trial's prediction before any fit on that trial. "
                         "The saved checkpoint was selected afterward on the full held-out set; "
                         "its score is in the summary above, not on these live curves. "
                         "The dotted Brier reference is constant P(+) = 0.5 (loss 0.25).")
        caption.setWordWrap(True)
        layout.addWidget(caption)
        scroller.setWidget(content)
        self.setCentralWidget(scroller)
        self.setStyleSheet("QMainWindow,QWidget{background:#182335;color:#e8effc;font-size:14px} "
                           "QScrollArea{border:0}")

    @staticmethod
    def _plot(title: str, axis: str, units: str) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setFixedHeight(310)
        plot.setBackground("#101827")
        plot.setTitle(title)
        plot.setLabel("left", axis, units=units)
        plot.getAxis("left").enableAutoSIPrefix(False)
        plot.setLabel("bottom", "Scene presentation")
        plot.showGrid(x=True, y=True, alpha=.2)
        plot.addLegend(offset=(-10, 10))
        return plot

    @staticmethod
    def _summary(data: dict) -> str:
        gabor = data.get("gabor_check", {})
        lines = [f"Scene trials: {data.get('presented_trials', '?')} / {data.get('requested_trials', '?')} requested"
                 f" · usable training EEG: {data.get('usable_training_trials', '?')}"
                 f" · usable held-out EEG: {data.get('usable_validation_trials', '?')}"
                 f" · skipped: {data.get('skipped_eeg_trials', '?')}"]
        if gabor.get("status") == "scored":
            lines.append(f"Gabor check: {gabor['balanced_accuracy']:.1%} held-out balanced accuracy "
                         f"({gabor['usable_held_out']} EEG trials), AUC {gabor['auc']:.3f}, "
                         f"Brier {gabor['brier']:.3f}; "
                         + ("visual response detected" if gabor.get("visual_response_detected") else
                            "no clear visual response detected"))
        for model, entry in data.get("checkpoint_selection", {}).items():
            lines.append(f"Selected {model} checkpoint after {entry['training_trials']} training EEG trials: "
                         f"held-out accuracy {entry['validation_accuracy']:.1%}, "
                         f"balanced accuracy {entry['validation_balanced_accuracy']:.1%}, "
                         f"AUC {entry['validation_auc']:.3f}, Brier {entry['validation_brier']:.3f} "
                         f"on {entry['validation_n']} EEG trials "
                         f"({'beats' if entry.get('beats_coinflip_brier') else 'does not beat'} "
                         "constant 50/50 Brier 0.25).")
        if "offline_refit" in data:
            entry = data["offline_refit"]
            lines.append(f"Trimmed clean refit valence/logreg: held-out accuracy {entry['accuracy']:.1%}, "
                         f"balanced accuracy {entry['balanced_accuracy']:.1%}, "
                         f"AUC {entry['auc']:.3f}, Brier {entry['brier']:.3f} on {entry['n']} EEG trials.")
            lines.append("Scene presentations 1-10 and 701-710 were excluded from this analysis and refit.")
        lines.append("Checkpoint selection used these held-out trials; this is not an independent test-set estimate.")
        return "\n".join(lines)
