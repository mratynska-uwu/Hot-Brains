"""Taste experiment: is the participant tasting something sweet or salty?

"Gorzki smak porażki i słodko-słony sukces (?)" is an alternative experiment
selected from the GUI's Configuration list. It reuses the EEG pipeline (filters,
feature families, classifiers) of the scene-emotion trainer but replaces the
images with taste cues, and it classifies a sustained state instead of an
image-evoked response. This module is Qt-free so it can be tested directly.

Trial timeline (seconds from cue onset, defaults in ``TASTE_CONFIG``)::

    -6..0   fixation "+" (mouth still; -3.0..-0.5 s = the trial's own baseline)
     0..3   cue word "SŁODKIE" / "SŁONE": the participant puts the tastant on the tongue
     3..20  fixation dot, identical for both tastes: hold the tastant, do not swallow
     5..15  analysed EEG: 2 s windows every 1 s, each minus the pre-cue baseline
    20..40  "swallow and rinse with water" (removes the taste before the next trial)

Why it is laid out like this (details in docs/smak_procedura.md):

* Placing the tastant moves the hand, jaw and tongue. Those artifacts and the
  visual response to the cue word are over well before +5 s, so the analysis
  window starts there.
* The cue word differs between the classes (different letters = different
  visual input). It is replaced by the same fixation dot for both tastes at
  +3 s, 2 s before the analysis window, so a classifier cannot learn to "read"
  the word from occipital activity. ``cue_visible_s = trial_s`` keeps the word
  on screen for the whole trial instead.
* Features are changes relative to the trial's own pre-cue baseline
  (log power differences = ERD/ERS). This removes slow drifts of electrode
  contact over a 30-minute session.
* Windows of one trial are strongly correlated, so every evaluation is
  trial-level: prequential predictions average the windows of an unseen trial,
  and cross-validation never splits one trial's windows across folds.
* Gamma and high beta are dominated by jaw/tongue EMG during tasting; they are
  available but not in the default preset. A separate EMG guard (30-45 Hz RMS)
  rejects windows with muscle activity.
"""
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from . import classifiers
from .config import CONFIG, EEGSettings, Profile
from .features import EpochSpec, TargetFeatures

TASTE_EXPERIMENT = "taste"
TASTE_NAME = "Gorzki smak porażki i słodko-słony sukces (?)"
TASTE_TARGET = "taste"

SALTY, SWEET = 0, 1   # label 0 / label 1; models output P(sweet)

TASTE_CONFIG: dict = {
    "name": TASTE_NAME,
    # Label 0 and label 1. Every probability in this experiment is P(label 1) = P(sweet).
    "classes": ["słone", "słodkie"],
    # Participant-facing texts (Polish).
    "cue_text": ["SŁONE", "SŁODKIE"],
    "blind_cue_text": "SMAKUJ",                      # blind mode: same word for both tastes
    "rinse_text": "Przełknij i przepłucz usta wodą",
    "timing": {
        "trial_s": 20.0,             # cue onset -> end of the tasting period
        "cue_visible_s": 3.0,        # then the identical fixation dot (set = trial_s to keep the word)
        "analysis_s": [5.0, 15.0],   # analysed EEG after cue onset
        "baseline_s": [-3.0, -0.5],  # the trial's reference interval before the cue
        "rinse_s": 20.0,             # swallow and rinse with water
        "fixation_s": [5.0, 7.0],    # jittered fixation before the cue (contains the baseline)
        "warmup_s": 15.0,            # fixation before the first trial (OSCAR settling, calm start)
    },
    # Sliding windows inside the analysis interval (and the baseline interval).
    "windows": {"window_s": 2.0, "step_s": 1.0},
    "trials": {"per_taste": 20, "max_run": 2},   # 40 trials ≈ 31 min; never more than 2 equal tastes in a row
    # Subtract the trial's pre-cue features from every analysis window (ERD/ERS).
    "baseline_correction": True,
    "artifacts": {
        "emg_band_hz": [30.0, 45.0],   # muscle (jaw, tongue, swallowing) dominates this band
        "emg_max_rms_uv": 10.0,        # reject a window when any model channel exceeds this RMS there
        "min_windows": 3,              # fewest clean analysis windows for a usable trial
    },
    # Live P(sweet) gauge in test sessions.
    "feedback": {"update_s": 0.5, "smoothing": 0.3},
    # Trial-level grouped cross-validation and permutation test after each training run.
    "evaluation": {"cv_folds": 5, "permutations": 200, "seed": 0, "budget_s": 40.0},
    # Shrinkage LDA is the default: few trials, correlated windows, linear and well regularised.
    "classifiers": ["lda_shrinkage", "logreg"],
    # Same cap and wiring as the scene experiment (one headset), taste-specific bands and limits.
    "eeg": {
        **copy.deepcopy(CONFIG["eeg"]),
        "bands_hz": {"theta": [4, 8], "alpha": [8, 13], "beta_low": [13, 20], "beta_high": [20, 30],
                     "gamma": [30, 45]},
        "features": ["band_power"],
        "reference": None,
        "artifact_max_abs_uv": 150.0,
    },
    # Feature presets for the single target "taste" (GUI "Preset"). The window_ms
    # field is unused here: the analysis interval comes from "timing".
    "profiles": {
        1: {"name": "Fronto-central θ/α/low β + C4−C3 asymmetry (18 features)",
            "window_ms": None, "baseline_ms": [-3000, -500],
            "targets": {TASTE_TARGET: {"electrodes": ["Fz", "C3", "Cz", "C4", "Pz"],
                                       "bands": ["theta", "alpha", "beta_low"],
                                       "features": ["band_power", "asymmetry"]}}},
        2: {"name": "Regions · all electrodes θ/α/low β + asymmetry (18 features, fewest trials)",
            "window_ms": None, "baseline_ms": [-3000, -500],
            "targets": {TASTE_TARGET: {"electrodes": "all", "bands": ["theta", "alpha", "beta_low"],
                                       "features": ["region_band_power", "asymmetry"]}}},
        3: {"name": "Exploratory · all electrodes θ/α/β + Hjorth (56 features)",
            "window_ms": None, "baseline_ms": [-3000, -500],
            "targets": {TASTE_TARGET: {"electrodes": "all",
                                       "bands": ["theta", "alpha", "beta_low", "beta_high"],
                                       "features": ["band_power", "asymmetry", "hjorth"]}}},
    },
    "default_profile": 1,
}

# Feature families that need an image-locked epoch, and models that need a continuous target.
UNSUPPORTED_FEATURES = ("erp",)
UNSUPPORTED_CLASSIFIERS = ("ridge",)


# ---------------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class TasteTiming:
    trial_s: float = 20.0
    cue_visible_s: float = 3.0
    analysis_start_s: float = 5.0
    analysis_end_s: float = 15.0
    baseline_start_s: float = -3.0
    baseline_end_s: float = -0.5
    rinse_s: float = 20.0
    fixation_min_s: float = 5.0
    fixation_max_s: float = 7.0
    window_s: float = 2.0
    step_s: float = 1.0

    def problems(self) -> list[str]:
        """Human-readable reasons why this timing cannot be used (empty = valid)."""
        out = []
        if self.trial_s <= 0 or self.rinse_s < 0 or self.window_s <= 0 or self.step_s <= 0:
            out.append("durations must be positive")
        if not 0 < self.analysis_start_s < self.analysis_end_s <= self.trial_s:
            out.append("the analysis window must lie inside the trial (0 < start < end ≤ trial)")
        elif self.analysis_end_s - self.analysis_start_s < self.window_s:
            out.append(f"the analysis window must be at least one {self.window_s:g} s window long")
        if not 0 < self.cue_visible_s <= self.trial_s:
            out.append("the cue must be visible for more than 0 s and at most the trial")
        elif self.cue_visible_s < self.trial_s and self.cue_visible_s > self.analysis_start_s - 1.0:
            # The cue's disappearance evokes a visual response for about a second.
            out.append("hide the cue at least 1 s before the analysis window starts "
                       "(or keep it for the whole trial)")
        if not self.baseline_start_s < self.baseline_end_s <= 0:
            out.append("the baseline must end at or before cue onset")
        elif self.baseline_end_s - self.baseline_start_s < self.window_s:
            out.append(f"the baseline must be at least one {self.window_s:g} s window long")
        if not 0 < self.fixation_min_s <= self.fixation_max_s:
            out.append("fixation needs 0 < minimum ≤ maximum")
        elif self.fixation_min_s < -self.baseline_start_s:
            out.append("the shortest fixation must contain the whole baseline")
        return out

    @property
    def cue_kept(self) -> bool:
        return self.cue_visible_s >= self.trial_s

    def epoch_spec(self, fs: int) -> EpochSpec:
        """Marker-aligned epoch from the baseline start to the analysis end."""
        return EpochSpec(round(self.trial_s * 1000), fs, round(self.analysis_end_s * 1000),
                         round(self.baseline_start_s * 1000), None)

    def n_windows(self) -> int:
        return max(0, int(np.floor((self.analysis_end_s - self.analysis_start_s - self.window_s)
                                   / self.step_s + 1e-9)) + 1)

    def mean_trial_s(self) -> float:
        return (self.fixation_min_s + self.fixation_max_s) / 2 + self.trial_s + self.rinse_s

    @classmethod
    def from_config(cls, d: dict) -> "TasteTiming":
        t, w = d["timing"], d["windows"]
        return cls(float(t["trial_s"]), float(t["cue_visible_s"]), float(t["analysis_s"][0]),
                   float(t["analysis_s"][1]), float(t["baseline_s"][0]), float(t["baseline_s"][1]),
                   float(t["rinse_s"]), float(t["fixation_s"][0]), float(t["fixation_s"][1]),
                   float(w["window_s"]), float(w["step_s"]))


@dataclass(frozen=True)
class TasteConfig:
    raw: dict
    eeg: EEGSettings
    profiles: tuple[Profile, ...]
    default_profile: int
    timing: TasteTiming
    classes: tuple[str, str]
    cue_text: tuple[str, str]
    blind_cue_text: str
    rinse_text: str
    per_taste: int
    max_run: int
    baseline_correction: bool
    emg_band_hz: tuple[float, float] | None
    emg_max_rms_uv: float | None
    min_windows: int
    warmup_s: float
    feedback_update_s: float
    feedback_smoothing: float
    evaluation: dict
    classifiers: tuple[str, ...]

    def profile(self, number: int) -> Profile:
        for profile in self.profiles:
            if profile.number == number:
                return profile
        raise KeyError(f"No taste preset {number}")

    @classmethod
    def from_dict(cls, d: dict) -> "TasteConfig":
        eeg = EEGSettings.from_dict(d["eeg"])
        profiles = tuple(Profile.from_dict(int(k), v, eeg, [TASTE_TARGET])
                         for k, v in sorted(d["profiles"].items(), key=lambda kv: int(kv[0])))
        for profile in profiles:
            unsupported = set(profile.target(TASTE_TARGET).features) & set(UNSUPPORTED_FEATURES)
            if unsupported:
                raise ValueError(f"taste preset {profile.number} uses {sorted(unsupported)}, "
                                 "which needs an image-locked epoch")
        default = int(d.get("default_profile", profiles[0].number))
        if default not in {p.number for p in profiles}:
            raise ValueError("taste default_profile must be one of the presets")
        timing = TasteTiming.from_config(d)
        problems = timing.problems()
        if problems:
            raise ValueError("taste timing: " + "; ".join(problems))
        classes = tuple(str(c) for c in d["classes"])
        cues = tuple(str(c) for c in d["cue_text"])
        if len(classes) != 2 or len(cues) != 2:
            raise ValueError("taste classes and cue_text need exactly two entries (salty, sweet)")
        trials = d["trials"]
        per_taste, max_run = int(trials["per_taste"]), int(trials["max_run"])
        if per_taste < 2 or max_run < 1:
            raise ValueError("taste trials need per_taste ≥ 2 and max_run ≥ 1")
        artifacts = d.get("artifacts", {})
        band = artifacts.get("emg_band_hz")
        band = None if band is None else (float(band[0]), float(band[1]))
        if band is not None and not 0 < band[0] < band[1] < eeg.fs / 2:
            raise ValueError("taste artifacts.emg_band_hz must lie below the Nyquist frequency")
        emg = artifacts.get("emg_max_rms_uv")
        keys = tuple(d.get("classifiers", ("lda_shrinkage",)))
        unknown = [k for k in keys if k not in classifiers.CLASSIFIERS or k in UNSUPPORTED_CLASSIFIERS]
        if unknown or not keys or len(keys) > 3:
            raise ValueError(f"taste classifiers must be 1-3 binary classifiers; got {list(keys)}")
        feedback = d.get("feedback", {})
        return cls(d, eeg, profiles, default, timing, classes, cues, str(d.get("blind_cue_text", "SMAKUJ")),
                   str(d.get("rinse_text", "")), per_taste, max_run, bool(d.get("baseline_correction", True)),
                   band, None if emg is None else float(emg), int(artifacts.get("min_windows", 3)),
                   float(d["timing"].get("warmup_s", 15.0)), float(feedback.get("update_s", 0.5)),
                   float(feedback.get("smoothing", 0.3)),
                   {"cv_folds": 5, "permutations": 200, "seed": 0, "budget_s": 40.0, **d.get("evaluation", {})},
                   keys)


# ---------------------------------------------------------------------------------------------- trial order
def taste_sequence(per_taste: int, max_run: int, rng: random.Random | None = None) -> list[int]:
    """Exactly ``per_taste`` salty (0) and sweet (1) trials in random order, no run longer than ``max_run``.

    Short runs keep expectation and taste carry-over from lining up with the label,
    and slow drifts over the session from correlating with it.
    """
    rng = rng or random.Random()
    remaining = [per_taste, per_taste]
    out: list[int] = []

    def feasible(left: list[int], last: int, run: int) -> bool:
        other = 1 - last
        return (left[last] <= (max_run - run) + max_run * left[other]
                and left[other] <= max_run * (left[last] + 1))

    while sum(remaining):
        options = []
        for label in (0, 1):
            if not remaining[label]:
                continue
            run = 0
            for previous in reversed(out):
                if previous != label:
                    break
                run += 1
            if run >= max_run:
                continue
            left = remaining.copy()
            left[label] -= 1
            if feasible(left, label, run + 1):
                options.append(label)
        if not options:
            raise ValueError(f"Cannot order {per_taste}+{per_taste} trials with runs of at most {max_run}")
        label = rng.choices(options, weights=[remaining[o] for o in options])[0]
        out.append(label)
        remaining[label] -= 1
    return out


# ---------------------------------------------------------------------------------------------- features
@dataclass
class TrialFeatures:
    """Feature rows of one trial (one dict per clean analysis window, baseline-corrected)."""
    rows: list[dict[str, np.ndarray]]
    n_windows: int
    reasons: dict[str, int]
    baseline_source: str               # "trial", "session median", "none" (no correction) or "missing"
    rejected: str | None = None        # why the whole trial is unusable


def _reason(text: str) -> str:
    return text.split(" ")[0]


class TasteFeatures:
    """Sliding-window features of taste trials for every target of a preset."""

    HISTORY = 20   # clean trial baselines kept for the session-median fallback

    def __init__(self, config: TasteConfig, profile: Profile, timing: TasteTiming | None = None,
                 baseline_correction: bool | None = None, emg_max_rms_uv: float | None | str = "config"):
        self.config = config
        self.eeg = config.eeg
        self.timing = timing or config.timing
        self.baseline_correction = config.baseline_correction if baseline_correction is None else baseline_correction
        self.emg_max_rms_uv = config.emg_max_rms_uv if emg_max_rms_uv == "config" else emg_max_rms_uv
        self.target_features = TargetFeatures(self.eeg, profile)
        self.window_spec = self.target_features.state_spec(self.timing.window_s)
        self.emg_sos = None
        if config.emg_band_hz is not None and self.emg_max_rms_uv is not None:
            sos = butter(4, config.emg_band_hz, btype="bandpass", fs=self.eeg.fs, output="sos")
            if self.eeg.notch_hz is not None:
                b, a = iirnotch(self.eeg.notch_hz, Q=30, fs=self.eeg.fs)
                sos = np.vstack([tf2sos(b, a), sos])
            self.emg_sos = sos
        self.history: list[dict[str, np.ndarray]] = []

    # --- description ----------------------------------------------------------------------------
    @property
    def targets(self) -> tuple[str, ...]:
        return self.target_features.targets

    @property
    def names(self) -> dict[str, tuple[str, ...]]:
        return self.target_features.names

    @property
    def problems(self) -> dict[str, str]:
        return self.target_features.problems

    def n_features(self) -> dict[str, int]:
        return self.target_features.n_features()

    def describe(self) -> dict:
        out = self.target_features.describe()
        for spec in out.values():
            spec["window_ms"] = None
        return out

    def epoch_spec(self) -> EpochSpec:
        return self.timing.epoch_spec(self.eeg.fs)

    # --- windows ----------------------------------------------------------------------------------
    def _cut(self, data: np.ndarray, onset: int, start_s: float, end_s: float, step_s: float) -> list[np.ndarray]:
        fs = self.eeg.fs
        width, step = round(self.timing.window_s * fs), max(1, round(step_s * fs))
        first, last = onset + round(start_s * fs), onset + round(end_s * fs)
        return [data[i:i + width] for i in range(max(0, first), min(last, len(data)) - width + 1, step)]

    def split(self, epoch: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """(baseline windows, analysis windows) of a marker-aligned trial epoch."""
        t = self.timing
        onset = round(-t.baseline_start_s * self.eeg.fs)
        baseline = self._cut(epoch, onset, t.baseline_start_s, t.baseline_end_s, min(t.step_s, .5))
        analysis = self._cut(epoch, onset, t.analysis_start_s, t.analysis_end_s, t.step_s)
        return baseline, analysis

    def emg_rms(self, window: np.ndarray) -> float:
        """Largest 30-45 Hz RMS (µV) over the model channels of one raw window."""
        data = np.asarray(window, dtype=np.float64)[:, self.target_features.union_channels]
        return float(np.sqrt(np.mean(sosfiltfilt(self.emg_sos, data, axis=0) ** 2, axis=0)).max())

    def window_vector(self, window: np.ndarray) -> dict[str, np.ndarray]:
        """Features of one raw window per target; ValueError (artifact) when it must be rejected."""
        vectors = self.target_features.extract(window, self.window_spec)
        if self.emg_sos is not None:
            rms = self.emg_rms(self.target_features.reference(window))
            if rms > self.emg_max_rms_uv:
                raise ValueError(f"emg {rms:.1f} µV RMS in {self.config.emg_band_hz[0]:g}-"
                                 f"{self.config.emg_band_hz[1]:g} Hz > {self.emg_max_rms_uv:g} µV")
        return vectors

    def _vectors(self, windows: list[np.ndarray], reasons: dict[str, int], prefix: str = ""):
        rows = []
        for window in windows:
            try:
                rows.append(self.window_vector(window))
            except ValueError as exc:
                key = prefix + _reason(str(exc))
                reasons[key] = reasons.get(key, 0) + 1
        return rows

    def baseline(self, windows: list[np.ndarray], reasons: dict[str, int] | None = None
                 ) -> dict[str, np.ndarray] | None:
        rows = self._vectors(windows, reasons if reasons is not None else {}, "baseline ")
        if not rows:
            return None
        return {t: np.mean([r[t] for r in rows], axis=0) for t in rows[0]}

    def fallback_baseline(self) -> dict[str, np.ndarray] | None:
        """Median of the previous clean trial baselines (only past trials: no look-ahead)."""
        if not self.history:
            return None
        return {t: np.median([h[t] for h in self.history], axis=0) for t in self.history[0]}

    def trial(self, epoch: np.ndarray) -> TrialFeatures:
        """Baseline-corrected feature rows of one marker-aligned epoch."""
        baseline_windows, analysis_windows = self.split(epoch)
        reasons: dict[str, int] = {}
        base, source = None, "none"
        if self.baseline_correction:
            base = self.baseline(baseline_windows, reasons)
            if base is not None:
                source = "trial"
                self.history = (self.history + [base])[-self.HISTORY:]
            else:
                base = self.fallback_baseline()
                source = "session median" if base is not None else "missing"
        rows = self._vectors(analysis_windows, reasons)
        if base is not None:
            rows = [{t: r[t] - base[t] for t in r} for r in rows]
        rejected = None
        if source == "missing":
            rejected = "no clean baseline (pre-cue) window yet"
        elif len(rows) < self.config.min_windows:
            rejected = f"only {len(rows)} clean analysis windows (< {self.config.min_windows})"
        return TrialFeatures(rows, len(analysis_windows), reasons, source, rejected)


# ---------------------------------------------------------------------------------------------- trial-level decisions
def trial_probability(model, rows: list[dict[str, np.ndarray]], target: str = TASTE_TARGET) -> float:
    """Mean P(sweet) over the windows of one trial (soft vote)."""
    x = np.vstack([r[target] for r in rows])
    return float(np.mean(model.predict_proba(x)[:, 1]))


@dataclass
class TrialOutcome:
    number: int
    label: int | None
    features: TrialFeatures
    probabilities: dict[tuple[str, str], float] = field(default_factory=dict)  # trial-level P(sweet)
    elapsed_ms: float = 0.0


def learn_trial(taste: TasteFeatures, trainer, epoch: np.ndarray, label: int, number: int) -> TrialOutcome:
    """Training step (worker thread): predict the unseen trial with every model, then learn its windows.

    ``probabilities`` is empty for models not fitted yet (fewer than 10 windows of both tastes).
    """
    started = time.perf_counter()
    trial = taste.trial(epoch)
    outcome = TrialOutcome(number, label, trial)
    if trial.rejected is None:
        predictions = trainer.add_batch([(r, {TASTE_TARGET: label}, None) for r in trial.rows], tag=number)
        outcome.probabilities = {key: float(np.mean([p.probability for p in rows]))
                                 for key, rows in predictions.items()}
    outcome.elapsed_ms = (time.perf_counter() - started) * 1000
    return outcome


def decode_trial(taste: TasteFeatures, models: dict, epoch: np.ndarray, label: int | None,
                 number: int) -> TrialOutcome:
    """Test step (worker thread): frozen models, trial-level P(sweet)."""
    started = time.perf_counter()
    trial = taste.trial(epoch)
    outcome = TrialOutcome(number, label, trial)
    if trial.rejected is None:
        outcome.probabilities = {key: trial_probability(model, trial.rows, key[0]) for key, model in models.items()}
    outcome.elapsed_ms = (time.perf_counter() - started) * 1000
    return outcome


# ---------------------------------------------------------------------------------------------- evaluation
def _metrics(truth, p) -> dict:
    from .report import classification_metrics
    p = np.asarray(p, dtype=float)
    return classification_metrics(truth, (p >= .5).astype(int), p)


def _balanced(truth: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean([np.mean(predicted[truth == c] == c) for c in (0, 1)]))


def grouped_oof(model_key: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int, seed: int
                ) -> np.ndarray:
    """Out-of-fold window P(sweet); all windows of a trial stay in the same fold."""
    from sklearn.model_selection import StratifiedGroupKFold
    oof = np.full(len(y), np.nan)
    for train, test in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups):
        if len(set(y[train])) < 2:
            continue
        model = classifiers.build(model_key)
        model.fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
    return oof


def per_trial(values: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(trial ids, trial labels, mean value per trial)."""
    ids = np.array(sorted(set(groups.tolist())))
    truth = np.array([y[groups == g][0] for g in ids])
    mean = np.array([np.nanmean(values[groups == g]) for g in ids])
    return ids, truth, mean


def evaluate_model(model_key: str, trainer, settings: dict,
                   prequential: list[tuple[int, float]] | None = None) -> dict:
    """Trial-level grouped cross-validation and a trial-label permutation test for one model.

    ``prequential`` holds the (taste, P(sweet)) of every trial predicted during the run
    before it was learned.
    """
    X = np.vstack(trainer.x) if trainer.x else np.empty((0, trainer.n_features))
    y = np.asarray(trainer.y, dtype=int)
    groups = np.asarray(trainer.tags)
    out: dict = {"windows": int(len(y)), "trials": int(len(set(groups.tolist())))}
    scored = list(prequential or [])
    if scored:
        truth = np.array([t for t, _ in scored])
        out["prequential"] = _metrics(truth, [p for _, p in scored])
    if not len(y):
        return out
    trial_labels = {g: y[groups == g][0] for g in set(groups.tolist())}
    smallest = min(sum(1 for v in trial_labels.values() if v == c) for c in (0, 1))
    folds = min(int(settings.get("cv_folds", 5)), smallest)
    if folds < 2:
        out["cv_note"] = "fewer than 2 trials of one taste: no cross-validation"
        return out
    seed = int(settings.get("seed", 0))
    try:
        oof = grouped_oof(model_key, X, y, groups, folds, seed)
    except ValueError as exc:
        out["cv_note"] = f"cross-validation failed: {exc}"
        return out
    keep = np.isfinite(oof)
    ids, truth, p_trial = per_trial(oof[keep], y[keep], groups[keep])
    out["cv"] = {"folds": folds, "trial_level": _metrics(truth, p_trial),
                 "window_level": _metrics(y[keep], oof[keep])}
    out["cv"]["trials"] = [{"trial": int(i), "label": int(t), "p_sweet": round(float(p), 4)}
                           for i, t, p in zip(ids, truth, p_trial)]
    # Permutation test: shuffle the taste of whole trials (never of single windows).
    observed = _balanced(truth, (p_trial >= .5).astype(int))
    rng = np.random.default_rng(seed)
    budget = float(settings.get("budget_s", 40.0))
    started, hits, done = time.perf_counter(), 0, 0
    trial_ids = np.array(sorted(trial_labels))
    for _ in range(int(settings.get("permutations", 200))):
        if time.perf_counter() - started > budget:
            break
        shuffled = dict(zip(trial_ids, rng.permutation([trial_labels[g] for g in trial_ids])))
        y_perm = np.array([shuffled[g] for g in groups])
        try:
            perm = grouped_oof(model_key, X, y_perm, groups, folds, seed)
        except ValueError:
            continue
        k = np.isfinite(perm)
        _, t_perm, p_perm = per_trial(perm[k], y_perm[k], groups[k])
        hits += _balanced(t_perm, (p_perm >= .5).astype(int)) >= observed
        done += 1
    out["permutation"] = {"n": done, "observed_balanced_accuracy": observed,
                          "p_value": (hits + 1) / (done + 1) if done else None}
    return out


def build_taste_report(models: dict, metadata: dict, settings: dict,
                       prequential: dict[tuple[str, str], list[tuple[int, float]]] | None = None) -> dict:
    started = time.perf_counter()
    report = {"experiment": TASTE_NAME, "created_at": datetime.now().astimezone().isoformat(),
              "participant_id": metadata.get("participant_id"), "session": metadata.get("session"),
              "data_quality": {k: metadata.get(k) for k in ("presented_trials", "usable_trials", "rejected_trials",
                                                           "marked_bad_trials", "window_counts", "artifact_reasons",
                                                           "baseline_sources")},
              "timing": metadata.get("timing"), "features": metadata.get("target_features"),
              "models": {}}
    for (target, key), trainer in models.items():
        entry = {"classifier": classifiers.CLASSIFIERS[key].label}
        entry.update(evaluate_model(key, trainer, settings, (prequential or {}).get((target, key))))
        report["models"][f"{target}/{key}"] = entry
    report["computation_s"] = round(time.perf_counter() - started, 2)
    return report


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def taste_markdown(report: dict) -> str:
    lines = [f"# {report['experiment']}", "",
             f"Participant **{report.get('participant_id')}** · {report.get('created_at')}", ""]
    quality = report.get("data_quality") or {}
    lines += ["## Data quality", ""] + [f"- {k.replace('_', ' ')}: {v}" for k, v in quality.items() if v is not None]
    timing = report.get("timing") or {}
    if timing:
        lines += ["", "## Timing", ""] + [f"- {k}: {v}" for k, v in timing.items()]
    lines += ["", "## Models (trial level; P(sweet) = mean over the trial's windows)", "",
              "| Model | Trials | Prequential bal. acc. | CV bal. acc. | CV ROC AUC | Permutation p | Majority |",
              "|---|---|---|---|---|---|---|"]
    for name, m in report.get("models", {}).items():
        pre = m.get("prequential") or {}
        cv = (m.get("cv") or {}).get("trial_level") or {}
        perm = m.get("permutation") or {}
        auc = "—" if cv.get("roc_auc") is None else f"{cv['roc_auc']:.3f}"
        p_value = "—" if perm.get("p_value") is None else f"{perm['p_value']:.3f} ({perm['n']} perm.)"
        lines.append(f"| {name} | {m.get('trials')} | {_pct(pre.get('balanced_accuracy'))} | "
                     f"{_pct(cv.get('balanced_accuracy'))} | {auc} | {p_value} | "
                     f"{_pct(cv.get('majority_baseline'))} |")
        if m.get("cv_note"):
            lines.append(f"| {name} | | {m['cv_note']} | | | | |")
    lines += ["", "Cross-validation is stratified and grouped by trial (StratifiedGroupKFold): windows of one "
              "trial never appear in both training and test folds. The permutation test shuffles the taste "
              "labels of whole trials. Chance level is 50% balanced accuracy.", "",
              f"_Computed in {report.get('computation_s')} s._", ""]
    return "\n".join(lines)


def write_taste_report(report: dict, directory: Path, stem: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path, md_path = directory / f"{stem}.json", directory / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(taste_markdown(report), encoding="utf-8")
    return md_path, json_path
