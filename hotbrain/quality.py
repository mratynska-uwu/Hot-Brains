"""Conservative, signal-based per-channel checks for the live EEG display.

These checks cannot measure electrode impedance or prove scalp contact. They
only flag flat or unusually large recent signals in the stream we receive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .features import CHANNELS, FS


WINDOW_SAMPLES = 2 * FS
_SOS = butter(3, (1, 35), btype="bandpass", fs=FS, output="sos")


@dataclass(frozen=True)
class ChannelQuality:
    good: bool | None
    detail: str


def assess_signal_quality(samples: np.ndarray) -> list[ChannelQuality]:
    """Check each electrode's last two seconds of currently streamed EEG."""
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(CHANNELS):
        raise ValueError(f"Expected EEG array with {len(CHANNELS)} channels")
    if len(data) < WINDOW_SAMPLES:
        return [ChannelQuality(None, "Waiting for 2 seconds of EEG") for _ in CHANNELS]

    recent = data[-WINDOW_SAMPLES:]
    results = []
    for channel in recent.T:
        if not np.isfinite(channel).all():
            results.append(ChannelQuality(False, "Non-finite samples"))
            continue
        filtered = sosfiltfilt(_SOS, channel)
        rms = float(np.sqrt(np.mean(filtered ** 2)))
        swing = float(np.ptp(filtered))
        detail = f"1–35 Hz RMS {rms:.1f} µV; range {swing:.1f} µV"
        if rms < 1 or swing < 4:
            results.append(ChannelQuality(False, f"Flat or very weak signal · {detail}"))
        elif rms > 75 or swing > 300:
            results.append(ChannelQuality(False, f"Unusually large signal · {detail}"))
        else:
            results.append(ChannelQuality(True, f"Amplitude in provisional range · {detail}"))
    return results
