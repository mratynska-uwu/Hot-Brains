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
        recent = data[-self.window:]
        results = []
        for index in range(n):
            finite = np.isfinite(recent[:, index])
            valid_percent = 100 * float(finite.mean())
            if valid_percent < 80:
                results.append(ChannelQuality(False, f"Only {valid_percent:.1f}% real scans in the last 2 s"))
                continue
            # A dropped Bluetooth scan is not evidence that every electrode
            # lost contact. Filter an uninterrupted real segment; never fill
            # missing scans or let a gap become a large filter transient.
            edges = np.diff(np.r_[False, finite, False].astype(np.int8))
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            lengths = ends - starts
            longest = int(np.argmax(lengths))
            if lengths[longest] < self.eeg.fs // 2:
                results.append(ChannelQuality(None, "Waiting for 0.5 s of uninterrupted real EEG"))
                continue
            segment = recent[starts[longest]:ends[longest], index]
            filtered = sosfiltfilt(self.sos, segment)
            rms = float(np.sqrt(np.mean(filtered ** 2)))
            swing = float(np.ptp(filtered))
            detail = (f"{low:g}–{high:g} Hz RMS {rms:.1f} µV; range {swing:.1f} µV; "
                      f"{valid_percent:.1f}% real scans, {len(segment) / self.eeg.fs:.1f} s checked")
            if rms < 1 or swing < 4:
                results.append(ChannelQuality(False, f"Flat or very weak signal · {detail}"))
            elif rms > 75 or swing > 300:
                results.append(ChannelQuality(False, f"Unusually large signal · {detail}"))
            else:
                results.append(ChannelQuality(True, f"Amplitude in provisional range · {detail}"))
        return results
