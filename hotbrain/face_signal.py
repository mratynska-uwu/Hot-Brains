"""Predeclared frequency-tagged response measure for the face pilot."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

FS = 250
BLOCK_S = 20
BASE_HZ = 6.0
ODDBALL_HZ = BASE_HZ / 5
ELECTRODES = (5, 6, 7)  # PO7, Oz, PO8
HARMONICS = (1, 2, 3, 4)


def tagged_response(eeg: np.ndarray, fs: int = FS) -> dict:
    """Mean oddball-frequency SNR across preselected posterior sites/harmonics.

    A response establishes visual discrimination of the repeated change. It
    cannot by itself establish an emotion label or therapeutic improvement.
    """
    data = np.asarray(eeg, dtype=float)
    expected = round(BLOCK_S * fs)
    if data.shape != (expected, 8) or not np.isfinite(data).all():
        raise ValueError("Need a complete real 20-second, eight-channel EEG block")
    selected = data[:, ELECTRODES]
    if np.any(np.ptp(selected, axis=0) < 0.5) or np.any(np.ptp(selected, axis=0) > 1000):
        raise ValueError("Flat or unusually large posterior EEG")
    sos = butter(3, (0.3, 30), btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, detrend(selected, axis=0), axis=0)
    spectrum = np.abs(np.fft.rfft(filtered * np.hanning(expected)[:, None], axis=0))
    freqs = np.fft.rfftfreq(expected, 1 / fs)
    ratios = []
    for harmonic in HARMONICS:
        at = int(np.argmin(abs(freqs - ODDBALL_HZ * harmonic)))
        # Neighbouring bins estimate local noise without selecting a favourable
        # frequency after looking at this participant's data.
        noise = np.median(spectrum[[at - 4, at - 3, at + 3, at + 4], :], axis=0)
        ratios.append(spectrum[at, :] / np.maximum(noise, 1e-12))
    ratios = np.asarray(ratios)
    per_electrode = np.mean(ratios, axis=0)
    return {"ratio": float(np.mean(per_electrode)),
            "per_electrode": {name: float(value) for name, value in zip(("PO7", "Oz", "PO8"), per_electrode)},
            "harmonics": [float(x) for x in np.mean(ratios, axis=1)],
            "provisional_response": bool(float(np.mean(per_electrode)) >= 1.7)}
