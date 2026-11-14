"""Config-driven, inspectable EEG features.

Band power and fractal dimension (the methods in ``basic_info.md``), plus
optional hemispheric asymmetry and evoked amplitudes, are common affective EEG
descriptors. These features are hypotheses, not validated biomarkers. The
montage, filters, bands, regions and feature families come from
``hotbrain.config``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from .config import EEGSettings

START_MS = -200          # epoch start relative to image onset (baseline)
END_BEFORE_OFFSET_MS = 300  # epoch ends this long before the image disappears


@dataclass(frozen=True)
class EpochSpec:
    image_duration_ms: int = 1000
    fs: int = 250

    @property
    def end_ms(self) -> int:
        return self.image_duration_ms - END_BEFORE_OFFSET_MS

    @property
    def n_samples(self) -> int:
        return round((self.end_ms - START_MS) * self.fs / 1000)

    @property
    def baseline_samples(self) -> int:
        return round(-START_MS * self.fs / 1000)


def make_filter(eeg: EEGSettings) -> np.ndarray:
    """Bandpass (order 3 Butterworth) plus optional notch, as one SOS cascade."""
    sos = butter(3, eeg.bandpass_hz, btype="bandpass", fs=eeg.fs, output="sos")
    if eeg.notch_hz is not None:
        b, a = iirnotch(eeg.notch_hz, Q=30, fs=eeg.fs)
        sos = np.vstack([tf2sos(b, a), sos])
    return sos


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


class FeatureExtractor:
    def __init__(self, eeg: EEGSettings):
        self.eeg = eeg
        self.sos = make_filter(eeg)
        index = {name: i for i, name in enumerate(eeg.electrodes)}
        self._regions = [(name, tuple(index[m] for m in members)) for name, members in eeg.regions]
        self._pairs = [(r, l, index[r], index[l]) for r, l in eeg.asymmetry_pairs]
        names = []
        for family in eeg.features:
            if family == "band_power":
                for region, _ in self._regions:
                    for band, (lo, hi) in eeg.bands_hz:
                        names.append(f"{region} {band} {lo:g}-{hi:g} Hz log power")
            elif family == "fractal_dimension":
                for electrode in eeg.electrodes:
                    names.append(f"{electrode} Higuchi fractal dimension")
            elif family == "asymmetry":
                for right, left, _, _ in self._pairs:
                    for band, _ in eeg.bands_hz:
                        names.append(f"{right}-{left} {band} log-power asymmetry")
            elif family == "erp":
                for lo, hi in eeg.erp_windows_ms:
                    for region, _ in self._regions:
                        names.append(f"{region} ERP {lo}-{'end' if hi is None else hi} ms")
        self.names: tuple[str, ...] = tuple(names)

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

    def epoch_spec(self, image_duration_ms: int) -> EpochSpec:
        if image_duration_ms < self.min_image_duration_ms:
            raise ValueError(f"Image duration must be at least {self.min_image_duration_ms / 1000:g} s "
                             "for the configured ERP windows.")
        return EpochSpec(image_duration_ms, self.eeg.fs)

    def clean(self, epoch: np.ndarray, spec: EpochSpec) -> np.ndarray:
        """Notch + bandpass and subtract the pre-onset per-channel mean, in microvolts."""
        data = np.asarray(epoch, dtype=np.float64)
        expected = (spec.n_samples, len(self.channels))
        if data.shape != expected or not np.isfinite(data).all():
            raise ValueError(f"Expected finite epoch of shape {expected}")
        data = sosfiltfilt(self.sos, data, axis=0)
        return data - data[:spec.baseline_samples].mean(axis=0)

    def extract(self, epoch: np.ndarray, spec: EpochSpec) -> np.ndarray:
        data = self.clean(epoch, spec)
        fs = spec.fs
        # Spectral features use the post-onset samples only.
        post = data[spec.baseline_samples:]
        spectrum = np.abs(np.fft.rfft(post * np.hanning(len(post))[:, None], axis=0)) ** 2
        frequencies = np.fft.rfftfreq(len(post), 1 / fs)

        def power(lo: float, hi: float, indices: tuple[int, ...]) -> float:
            mask = (frequencies >= lo) & (frequencies < hi)
            if not mask.any():
                raise ValueError(f"Epoch too short to resolve {lo:g}-{hi:g} Hz")
            return float(np.log10(spectrum[mask][:, indices].mean() + 1e-12))

        out: list[float] = []
        for family in self.eeg.features:
            if family == "band_power":
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
            elif family == "erp":
                for lo_ms, hi_ms in self.eeg.erp_windows_ms:
                    lo = round((lo_ms - START_MS) * fs / 1000)
                    hi = round(((spec.end_ms if hi_ms is None else hi_ms) - START_MS) * fs / 1000)
                    if hi > len(data) or hi <= lo:
                        raise ValueError(f"ERP window {lo_ms}-{hi_ms} ms does not fit in the epoch")
                    for _, indices in self._regions:
                        out.append(float(data[lo:hi][:, indices].mean()))
        result = np.asarray(out, dtype=np.float64)
        assert len(result) == len(self.names)
        return result
