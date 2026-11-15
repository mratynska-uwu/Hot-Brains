"""Config-driven, inspectable EEG features.

Band power and fractal dimension (the methods in ``basic_info.md``), plus
optional asymmetry, beta/alpha ratio, Hjorth parameters, Katz/Petrosian fractal
dimensions, variance and evoked amplitudes, are common affective EEG
descriptors. These features are hypotheses, not validated biomarkers.

``FeatureExtractor`` computes one feature vector for one set of electrodes,
bands and families. ``TargetFeatures`` holds one extractor per target
(valence, arousal, ...), as defined by the selected configuration profile, and
applies the shared re-referencing and artifact rejection.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

from .config import EEGSettings, Profile

START_MS = -200          # default epoch start relative to image onset (baseline)
END_BEFORE_OFFSET_MS = 300  # default epoch end: this long before the image disappears


@dataclass(frozen=True)
class EpochSpec:
    """Sample layout of one epoch.

    The epoch runs from ``start_ms`` (the baseline start) to ``end_ms`` after
    image onset. ``end_ms`` is ``window_ms`` when set, otherwise the image end
    minus 300 ms. The baseline is ``start_ms``..``baseline_end_ms``; with
    ``baseline_end_ms=None`` the whole epoch is de-meaned instead (sustained-state
    windows have no pre-stimulus period).
    """
    image_duration_ms: int = 1000
    fs: int = 250
    window_ms: int | None = None
    start_ms: int = START_MS
    baseline_end_ms: int | None = 0

    @property
    def end_ms(self) -> int:
        return self.window_ms if self.window_ms is not None else self.image_duration_ms - END_BEFORE_OFFSET_MS

    @property
    def n_samples(self) -> int:
        return round((self.end_ms - self.start_ms) * self.fs / 1000)

    @property
    def onset_index(self) -> int:
        return max(0, round(-self.start_ms * self.fs / 1000))

    @property
    def baseline_samples(self) -> int:
        """Samples from the epoch start to the end of the baseline (0 = whole-epoch de-mean)."""
        if self.baseline_end_ms is None:
            return 0
        return round((self.baseline_end_ms - self.start_ms) * self.fs / 1000)


def make_filter(eeg: EEGSettings) -> np.ndarray:
    """Bandpass (order 3 Butterworth) plus optional notch, as one SOS cascade."""
    sos = butter(3, eeg.bandpass_hz, btype="bandpass", fs=eeg.fs, output="sos")
    if eeg.notch_hz is not None:
        b, a = iirnotch(eeg.notch_hz, Q=30, fs=eeg.fs)
        sos = np.vstack([tf2sos(b, a), sos])
    return sos


def clean_epoch(sos: np.ndarray, epoch: np.ndarray, spec: EpochSpec) -> np.ndarray:
    """Notch + bandpass, then subtract the baseline mean per channel (µV)."""
    data = sosfiltfilt(sos, np.asarray(epoch, dtype=np.float64), axis=0)
    baseline = spec.baseline_samples
    return data - (data[:baseline] if baseline > 0 else data).mean(axis=0)


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    """Higuchi fractal dimension of a 1-D signal (about 1 for smooth, 2 for white noise)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    ks = np.arange(1, kmax + 1)
    lengths = []
    for k in ks:
        per_offset = []
        for m in range(k):
            steps = (n - 1 - m) // k
            if steps < 1:
                continue
            curve = np.abs(np.diff(x[m:m + steps * k + 1:k])).sum()
            per_offset.append(curve * (n - 1) / (steps * k) / k)
        lengths.append(np.mean(per_offset) + 1e-12)
    slope, _ = np.polyfit(np.log(1 / ks), np.log(lengths), 1)
    return float(slope)


def katz_fd(x: np.ndarray) -> float:
    """Katz fractal dimension (curve length vs. planar extent)."""
    x = np.asarray(x, dtype=np.float64)
    steps = np.abs(np.diff(x))
    length = steps.sum() + 1e-12
    mean_step = steps.mean() + 1e-12
    extent = np.max(np.abs(x[1:] - x[0])) + 1e-12
    ln = np.log10(length / mean_step)
    return float(ln / (ln + np.log10(extent / length)))


def petrosian_fd(x: np.ndarray) -> float:
    """Petrosian fractal dimension (from sign changes of the derivative)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    derivative = np.diff(x)
    changes = int(np.sum(derivative[1:] * derivative[:-1] < 0))
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * changes))))


def hjorth(x: np.ndarray) -> tuple[float, float]:
    """Hjorth mobility and complexity."""
    x = np.asarray(x, dtype=np.float64)
    dx, ddx = np.diff(x), np.diff(x, n=2)
    var_x, var_dx, var_ddx = np.var(x) + 1e-12, np.var(dx) + 1e-12, np.var(ddx) + 1e-12
    mobility = np.sqrt(var_dx / var_x)
    return float(mobility), float(np.sqrt(var_ddx / var_dx) / mobility)


@dataclass(frozen=True)
class FeatureInfo:
    """What one feature measures, so the GUI and reports can group features."""
    family: str           # band_power, region_band_power, fractal_dimension, asymmetry, ...
    location: str         # electrode, region or "right-left" pair
    band: str | None      # band name for spectral features, else None
    name: str


class FeatureExtractor:
    def __init__(self, eeg: EEGSettings):
        self.eeg = eeg
        self.sos = make_filter(eeg)
        index = {name: i for i, name in enumerate(eeg.electrodes)}
        self._regions = [(name, tuple(index[m] for m in members)) for name, members in eeg.regions]
        self._pairs = [(r, l, index[r], index[l]) for r, l in eeg.asymmetry_pairs]
        bands = dict(eeg.bands_hz)
        info: list[FeatureInfo] = []
        for family in eeg.features:
            if family == "band_power":
                for electrode in eeg.electrodes:
                    for band, (lo, hi) in eeg.bands_hz:
                        info.append(FeatureInfo(family, electrode, band, f"{electrode} {band} {lo:g}-{hi:g} Hz log power"))
            elif family == "region_band_power":
                for region, _ in self._regions:
                    for band, (lo, hi) in eeg.bands_hz:
                        info.append(FeatureInfo(family, region, band, f"{region} {band} {lo:g}-{hi:g} Hz log power"))
            elif family == "fractal_dimension":
                for electrode in eeg.electrodes:
                    info.append(FeatureInfo(family, electrode, None, f"{electrode} Higuchi fractal dimension"))
            elif family == "asymmetry":
                for right, left, _, _ in self._pairs:
                    for band, _ in eeg.bands_hz:
                        info.append(FeatureInfo(family, f"{right}-{left}", band, f"{right}-{left} {band} log-power asymmetry"))
            elif family == "beta_alpha_ratio":
                if "alpha" in bands and "beta" in bands:
                    for electrode in eeg.electrodes:
                        info.append(FeatureInfo(family, electrode, "beta/alpha", f"{electrode} log beta/alpha ratio"))
            elif family == "hjorth":
                for electrode in eeg.electrodes:
                    info.append(FeatureInfo(family, electrode, None, f"{electrode} Hjorth mobility"))
                    info.append(FeatureInfo(family, electrode, None, f"{electrode} Hjorth complexity"))
            elif family == "katz_fd":
                for electrode in eeg.electrodes:
                    info.append(FeatureInfo(family, electrode, None, f"{electrode} Katz fractal dimension"))
            elif family == "petrosian_fd":
                for electrode in eeg.electrodes:
                    info.append(FeatureInfo(family, electrode, None, f"{electrode} Petrosian fractal dimension"))
            elif family == "variance":
                for electrode in eeg.electrodes:
                    info.append(FeatureInfo(family, electrode, None, f"{electrode} log variance"))
            elif family == "erp":
                for lo, hi in eeg.erp_windows_ms:
                    for region, _ in self._regions:
                        info.append(FeatureInfo(family, region, None, f"{region} ERP {lo}-{'end' if hi is None else hi} ms"))
        self.info: tuple[FeatureInfo, ...] = tuple(info)
        self.names: tuple[str, ...] = tuple(f.name for f in info)

    def indices(self, family: str, location: str | None = None, band: str | None = None) -> list[int]:
        """Feature positions matching a family and optionally a location and/or band."""
        return [i for i, f in enumerate(self.info) if f.family == family
                and (location is None or f.location == location) and (band is None or f.band == band)]

    @property
    def channels(self) -> tuple[str, ...]:
        return self.eeg.electrodes

    @property
    def min_image_duration_ms(self) -> int:
        """Shortest image giving 400 ms of post-onset EEG and samples in every ERP window."""
        need = 400
        if "erp" in self.eeg.features:
            need = max([need, *(hi if hi is not None else lo + 50 for lo, hi in self.eeg.erp_windows_ms)])
        return need + END_BEFORE_OFFSET_MS

    def epoch_spec(self, image_duration_ms: int, window_ms: int | None = None,
                   baseline_ms: tuple[int, int] = (START_MS, 0)) -> EpochSpec:
        if window_ms is None and image_duration_ms < self.min_image_duration_ms:
            raise ValueError(f"Image duration must be at least {self.min_image_duration_ms / 1000:g} s "
                             "for the configured ERP windows.")
        if window_ms is not None and image_duration_ms < window_ms:
            raise ValueError(f"Image duration must be at least the {window_ms} ms analysis window.")
        return EpochSpec(image_duration_ms, self.eeg.fs, window_ms, baseline_ms[0], baseline_ms[1])

    def clean(self, epoch: np.ndarray, spec: EpochSpec) -> np.ndarray:
        """Notch + bandpass and subtract the baseline mean per channel, in microvolts."""
        data = np.asarray(epoch, dtype=np.float64)
        expected = (spec.n_samples, len(self.channels))
        if data.shape != expected or not np.isfinite(data).all():
            raise ValueError(f"Expected finite epoch of shape {expected}")
        return clean_epoch(self.sos, data, spec)

    def _spectrum(self, post: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
        if self.eeg.psd_method == "welch":
            nperseg = min(len(post), fs)
            return welch(post, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, axis=0)
        spectrum = np.abs(np.fft.rfft(post * np.hanning(len(post))[:, None], axis=0)) ** 2
        return np.fft.rfftfreq(len(post), 1 / fs), spectrum

    def extract(self, epoch: np.ndarray, spec: EpochSpec) -> np.ndarray:
        data = self.clean(epoch, spec)
        fs = spec.fs
        # Spectral and signal-shape features use the post-onset samples only.
        post = data[spec.onset_index:]
        frequencies, spectrum = self._spectrum(post, fs)

        def power(lo: float, hi: float, indices: tuple[int, ...]) -> float:
            mask = (frequencies >= lo) & (frequencies < hi)
            if not mask.any():
                raise ValueError(f"Epoch too short to resolve {lo:g}-{hi:g} Hz")
            return float(np.log10(spectrum[mask][:, indices].mean() + 1e-12))

        bands = dict(self.eeg.bands_hz)
        out: list[float] = []
        for family in self.eeg.features:
            if family == "band_power":
                for channel in range(len(self.channels)):
                    for _, (lo, hi) in self.eeg.bands_hz:
                        out.append(power(lo, hi, (channel,)))
            elif family == "region_band_power":
                for _, indices in self._regions:
                    for _, (lo, hi) in self.eeg.bands_hz:
                        out.append(power(lo, hi, indices))
            elif family == "fractal_dimension":
                for channel in post.T:
                    out.append(higuchi_fd(channel, self.eeg.fractal_kmax))
            elif family == "asymmetry":
                for _, _, right, left in self._pairs:
                    for _, (lo, hi) in self.eeg.bands_hz:
                        out.append(power(lo, hi, (right,)) - power(lo, hi, (left,)))
            elif family == "beta_alpha_ratio":
                if "alpha" in bands and "beta" in bands:
                    for channel in range(len(self.channels)):
                        out.append(power(*bands["beta"], (channel,)) - power(*bands["alpha"], (channel,)))
            elif family == "hjorth":
                for channel in post.T:
                    out.extend(hjorth(channel))
            elif family == "katz_fd":
                for channel in post.T:
                    out.append(katz_fd(channel))
            elif family == "petrosian_fd":
                for channel in post.T:
                    out.append(petrosian_fd(channel))
            elif family == "variance":
                for channel in post.T:
                    out.append(float(np.log10(np.var(channel) + 1e-12)))
            elif family == "erp":
                for lo_ms, hi_ms in self.eeg.erp_windows_ms:
                    lo = round((lo_ms - spec.start_ms) * fs / 1000)
                    hi = round(((spec.end_ms if hi_ms is None else hi_ms) - spec.start_ms) * fs / 1000)
                    if hi > len(data) or hi <= lo:
                        raise ValueError(f"ERP window {lo_ms}-{hi_ms} ms does not fit in the epoch")
                    for _, indices in self._regions:
                        out.append(float(data[lo:hi][:, indices].mean()))
        result = np.asarray(out, dtype=np.float64)
        assert len(result) == len(self.names)
        return result


class ArtifactError(ValueError):
    """The epoch was rejected before feature extraction."""


class TargetFeatures:
    """One feature extractor per target, built from a configuration profile.

    ``positions`` optionally restrict every target to a subset of the
    configured montage (indices); each target uses the intersection with its
    own electrodes. Each target may also have its own analysis window: the
    epoch spans the longest one and shorter targets use its leading part (same
    onset and baseline). Epochs passed to :meth:`extract` hold all configured
    channels.
    """

    def __init__(self, eeg: EEGSettings, profile: Profile, positions: tuple[int, ...] | None = None):
        self.eeg = eeg
        self.profile = profile
        positions = tuple(range(len(eeg.electrodes))) if positions is None else tuple(positions)
        allowed = {eeg.electrodes[i] for i in positions}
        self.extractors: dict[str, FeatureExtractor] = {}
        self.channel_indices: dict[str, tuple[int, ...]] = {}
        self.windows: dict[str, int | None] = {}
        self.problems: dict[str, str] = {}
        for target, spec in profile.targets:
            wanted = set(eeg.electrodes if spec.electrodes is None else spec.electrodes)
            names = tuple(n for n in eeg.electrodes if n in wanted and n in allowed)
            try:
                sub = eeg.for_target(names, spec.bands, spec.features)
                if spec.bandpass_hz is not None:   # feature filter only; artifact rejection keeps eeg's
                    sub = replace(sub, bandpass_hz=spec.bandpass_hz)
                extractor = FeatureExtractor(sub)
                if spec.window_ms is not None and spec.window_ms < extractor.min_image_duration_ms - END_BEFORE_OFFSET_MS:
                    raise ValueError(f"window {spec.window_ms} ms is shorter than the "
                                     f"{extractor.min_image_duration_ms - END_BEFORE_OFFSET_MS} ms its ERP windows need")
            except ValueError as exc:
                self.problems[target] = str(exc)
                continue
            self.extractors[target] = extractor
            self.windows[target] = spec.window_ms
            self.channel_indices[target] = tuple(eeg.electrodes.index(n) for n in sub.electrodes)
        self.union_channels = tuple(sorted({i for idx in self.channel_indices.values() for i in idx}))
        self.sos = make_filter(eeg)

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(self.extractors)

    @property
    def names(self) -> dict[str, tuple[str, ...]]:
        return {t: e.names for t, e in self.extractors.items()}

    def n_features(self) -> dict[str, int]:
        return {t: len(e.names) for t, e in self.extractors.items()}

    def describe(self) -> dict:
        return {t: {"electrodes": list(e.eeg.electrodes), "bands": [b for b, _ in e.eeg.bands_hz],
                    "features": list(e.eeg.features), "n_features": len(e.names),
                    "window_ms": self.windows[t], "bandpass_hz": list(e.eeg.bandpass_hz)}
                for t, e in self.extractors.items()}

    @property
    def epoch_window_ms(self) -> int | None:
        """Analysis window of the shared epoch: the longest target window (None = until image end)."""
        windows = list(self.windows.values())
        if not windows:
            return self.profile.window_ms
        return None if any(w is None for w in windows) else max(windows)

    @property
    def min_image_duration_ms(self) -> int:
        need = [w if w is not None else e.min_image_duration_ms
                for e, w in ((self.extractors[t], self.windows[t]) for t in self.extractors)]
        return max(need, default=700)

    def epoch_spec(self, image_duration_ms: int) -> EpochSpec:
        if image_duration_ms < self.min_image_duration_ms:
            raise ValueError(f"Image duration must be at least {self.min_image_duration_ms / 1000:g} s "
                             "for the selected analysis windows.")
        return EpochSpec(image_duration_ms, self.eeg.fs, self.epoch_window_ms,
                         self.profile.baseline_ms[0], self.profile.baseline_ms[1])

    def target_spec(self, target: str, spec: EpochSpec) -> EpochSpec:
        """The part of an image epoch that ``target`` analyses (its own window, same onset and baseline).

        Sustained-state windows (no baseline) are used whole by every target.
        """
        if spec.baseline_end_ms is None or self.windows.get(target) == spec.window_ms:
            return spec
        return replace(spec, window_ms=self.windows[target])

    def state_spec(self, window_s: float) -> EpochSpec:
        """Sliding-window layout for sustained states (calibration/test): whole window de-meaned."""
        window_ms = round(window_s * 1000)
        return EpochSpec(window_ms, self.eeg.fs, window_ms, 0, None)

    def reference(self, epoch: np.ndarray) -> np.ndarray:
        data = np.asarray(epoch, dtype=np.float64)
        if self.eeg.reference == "car":
            data = data - data.mean(axis=1, keepdims=True)
        return data

    def check_artifacts(self, epoch: np.ndarray, spec: EpochSpec) -> None:
        """Raise ArtifactError for flat/non-finite channels or amplitudes above the configured limit."""
        data = np.asarray(epoch, dtype=np.float64)[:, self.union_channels]
        if not np.isfinite(data).all():
            raise ArtifactError("non-finite samples")
        if np.ptp(data, axis=0).min() < .05:
            raise ArtifactError("flat channel")
        limit = self.eeg.artifact_max_abs_uv
        if limit is not None:
            peak = float(np.abs(clean_epoch(self.sos, data, spec)).max())
            if peak > limit:
                raise ArtifactError(f"amplitude {peak:.0f} µV > {limit:g} µV")

    def extract(self, epoch: np.ndarray, spec: EpochSpec) -> dict[str, np.ndarray]:
        """Feature vector per target from one epoch of all configured channels."""
        data = self.reference(epoch)
        self.check_artifacts(data, spec)
        out = {}
        for t, e in self.extractors.items():
            own = self.target_spec(t, spec)
            out[t] = e.extract(data[:own.n_samples, self.channel_indices[t]], own)
        return out
