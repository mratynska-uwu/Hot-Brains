"""Conservative, signal-based per-channel checks for the live EEG display.

These checks cannot measure electrode impedance or prove scalp contact. They
only flag flat or unusually large recent signals in the stream we receive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import sosfiltfilt

from .config import EEGSettings
from .features import make_filter

WINDOW_SECONDS = 2


@dataclass(frozen=True)
class ChannelQuality:
    good: bool | None
    detail: str


class SignalQuality:
    def __init__(self, eeg: EEGSettings):
        self.eeg = eeg
        self.sos = make_filter(eeg)
        self.window = WINDOW_SECONDS * eeg.fs

    def waiting(self, reason: str = "Waiting for EEG") -> list[ChannelQuality]:
        return [ChannelQuality(None, reason) for _ in self.eeg.electrodes]

    def assess(self, samples: np.ndarray) -> list[ChannelQuality]:
        """Check each electrode's last two seconds of currently streamed EEG."""
        data = np.asarray(samples, dtype=np.float64)
        n = len(self.eeg.electrodes)
        if data.ndim != 2 or data.shape[1] != n:
            raise ValueError(f"Expected EEG array with {n} channels")
        if len(data) < self.window:
            return self.waiting(f"Waiting for {WINDOW_SECONDS} seconds of EEG")
        low, high = self.eeg.bandpass_hz
        recent = data[-self.window:].copy()
        finite = np.isfinite(recent).all(axis=0)
        recent[:, ~finite] = 0  # Keep one bad electrode from affecting the others.
        filtered = sosfiltfilt(self.sos, recent, axis=0)
        rms_values = np.sqrt(np.mean(filtered ** 2, axis=0))
        swings = np.ptp(filtered, axis=0)
        results = []
        for index in range(n):
            if not finite[index]:
                results.append(ChannelQuality(False, "Non-finite samples"))
                continue
            rms = float(rms_values[index])
            swing = float(swings[index])
            detail = f"{low:g}–{high:g} Hz RMS {rms:.1f} µV; range {swing:.1f} µV"
            if rms < 1 or swing < 4:
                results.append(ChannelQuality(False, f"Flat or very weak signal · {detail}"))
            elif rms > 75 or swing > 300:
                results.append(ChannelQuality(False, f"Unusually large signal · {detail}"))
            else:
                results.append(ChannelQuality(True, f"Amplitude in provisional range · {detail}"))
        return results
