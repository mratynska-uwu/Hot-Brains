"""Fixed, inspectable EEG features for the Unicorn's usual 8-channel montage.

Band power and hemispheric asymmetry are common affective EEG descriptors;
early and late evoked amplitudes are candidate picture-response descriptors.
These features are hypotheses, not validated valence biomarkers.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.signal import butter, sosfiltfilt

FS = 250
CHANNELS = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")
START_MS, END_MS = -200, 700
N_SAMPLES = int((END_MS - START_MS) * FS / 1000)


@dataclass(frozen=True)
class EpochSpec:
    image_duration_ms: int = 1000

    def __post_init__(self):
        if self.image_duration_ms < 700:
            raise ValueError("Image duration must be at least 0.7 s for the late EEG feature window.")

    @property
    def end_ms(self) -> int:
        return self.image_duration_ms - 300

    @property
    def n_samples(self) -> int:
        return round((self.end_ms - START_MS) * FS / 1000)


DEFAULT_EPOCH = EpochSpec()
FEATURE_NAMES = (
    "Fz theta 4-8 Hz", "Fz alpha 8-13 Hz", "Fz beta 13-30 Hz",
    "central theta 4-8 Hz", "central alpha 8-13 Hz", "central beta 13-30 Hz",
    "posterior theta 4-8 Hz", "posterior alpha 8-13 Hz", "posterior beta 13-30 Hz",
    "PO8-PO7 alpha log power", "PO8-PO7 beta log power",
    "C4-C3 alpha log power", "C4-C3 beta log power",
    "Fz ERP 100-200 ms", "posterior ERP 100-200 ms",
    "Fz ERP 200-350 ms", "posterior ERP 200-350 ms",
    "Pz ERP 350 ms to epoch end", "posterior ERP 350 ms to epoch end",
    "posterior late absolute ERP (salience candidate)",
)
_SOS = butter(3, (1, 35), btype="bandpass", fs=FS, output="sos")


def clean_epoch(epoch: np.ndarray, spec: EpochSpec = DEFAULT_EPOCH) -> np.ndarray:
    """Bandpass and subtract the -200..0 ms per-channel mean, in microvolts."""
    data = np.asarray(epoch, dtype=np.float64)
    if data.shape != (spec.n_samples, len(CHANNELS)) or not np.isfinite(data).all():
        raise ValueError(f"Expected finite epoch of shape {(spec.n_samples, len(CHANNELS))}")
    data = sosfiltfilt(_SOS, data, axis=0)
    return data - data[:50].mean(axis=0)


def extract(epoch: np.ndarray, spec: EpochSpec = DEFAULT_EPOCH) -> np.ndarray:
    data = clean_epoch(epoch, spec)
    # The first 200 ms is baseline; spectral features use stimulus samples.
    post = data[50:]
    spectrum = np.abs(np.fft.rfft(post * np.hanning(len(post))[:, None], axis=0)) ** 2
    frequencies = np.fft.rfftfreq(len(post), 1 / FS)

    def power(lo: float, hi: float, indices: tuple[int, ...]) -> float:
        mask = (frequencies >= lo) & (frequencies < hi)
        return float(np.log10(spectrum[mask][:, indices].mean() + 1e-12))

    out: list[float] = []
    for indices in ((0,), (1, 2, 3), (4, 5, 6, 7)):
        for band in ((4, 8), (8, 13), (13, 30)):
            out.append(power(*band, indices))
    for right, left in ((7, 5), (3, 1)):
        for band in ((8, 13), (13, 30)):
            out.append(power(*band, (right,)) - power(*band, (left,)))

    def amplitude(lo_ms: int, hi_ms: int, indices: tuple[int, ...]) -> float:
        lo = int((lo_ms - START_MS) * FS / 1000)
        hi = int((hi_ms - START_MS) * FS / 1000)
        return float(data[lo:hi][:, indices].mean())

    for lo, hi, index_groups in (
        (100, 200, ((0,), (5, 6, 7))),
        (200, 350, ((0,), (5, 6, 7))),
        (350, spec.end_ms, ((4,), (5, 6, 7))),
    ):
        for indices in index_groups:
            out.append(amplitude(lo, hi, indices))
    out.append(abs(amplitude(350, spec.end_ms, (5, 6, 7))))
    result = np.asarray(out, dtype=np.float64)
    assert len(result) == len(FEATURE_NAMES) == 20
    return result
