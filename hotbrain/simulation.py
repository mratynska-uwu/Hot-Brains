"""Simulated EEG source for testing the full pipeline without a headset.

This is the one, explicitly opt-in exception to "the app never generates
substitute EEG": it exists only for testing the GUI, sessions and classifiers,
and every simulated run is labelled SIMULATED in the GUI, manifests and file
names. It is never used unless the operator ticks "Simulate EEG" or starts the
app with ``--simulate``.

``SimulatedDevice`` has the same interface as ``acquisition.DeviceManager``
(``stream_ready``, ``readiness_issue``, ``mark_image``, ``ring`` ...). It streams
8 Unicorn-like channels in real time into the same marker-aligned
``SampleRing``; image markers are written into a marker column on the
simulator's own sample grid at the Qt paint time, so epochs are cut exactly
like with the real headset (without the Unicorn's 40 ms device delay).

Background: low-frequency (AR(1)) noise + white noise + a 10 Hz alpha rhythm
+ 50 Hz mains hum, about 5-10 µV after filtering. Every image adds an ERP-like
bump at ~300 ms. While an image is shown, each target whose label is 1 adds an
oscillation of ``effect_uv`` µV at its own frequency (1st target 10 Hz, 2nd
20 Hz, 3rd 6 Hz, ...). With ``effect_uv = 0`` the EEG carries no label
information, so classifiers should stay at chance; that is a useful check.
"""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from scipy.signal import lfilter

from .acquisition import SampleRing
from .config import EEGSettings

DEVICE_CHANNELS = EEGSettings.UNICORN_CHANNELS
MARKER_COLUMN = DEVICE_CHANNELS          # frames are 8 EEG columns + 1 marker column
EFFECT_FREQUENCIES_HZ = (10.0, 20.0, 6.0, 30.0)
TICK_MS = 40


class SimulatedDevice(QObject):
    status_changed = Signal(str)
    connected_changed = Signal(bool)

    def __init__(self, eeg: EEGSettings, target_names: list[str], effect_uv: float = 5.0,
                 seed: int | None = None, autostart: bool = True):
        super().__init__()
        self.eeg = eeg
        self.target_names = list(target_names)
        self.effect_uv = effect_uv
        self.oscar = False
        self.serial = None
        self.retention_s = 20.0
        self.ring = SampleRing(self.retention_s, eeg.fs, eeg.device_channels, device_delay_ms=0)
        self.connected = False
        self.rng = np.random.default_rng(seed)
        self.t0: float | None = None
        self.generated = 0
        self.stimuli: list[tuple[float, float, dict[str, int]]] = []
        self._pending_markers: dict[int, int] = {}
        self._next_marker = 0
        self._ar_state = np.zeros((1, DEVICE_CHANNELS))
        self._phase = self.rng.uniform(0, 2 * np.pi, DEVICE_CHANNELS)
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)
        if autostart:
            QTimer.singleShot(0, self.start)

    # --- DeviceManager-compatible interface -------------------------------
    def configure(self, oscar: bool, serial: str | None = None):
        """OSCAR and the serial number do not apply to simulated EEG."""

    def set_retention(self, seconds: float):
        self.retention_s = max(20.0, seconds)
        self.ring.retention_s = self.retention_s

    def stream_ready(self, max_age_s: float = .25) -> bool:
        # Samples are produced on the Qt timer, which pauses while the GUI does
        # blocking work (loading images, saving models). Catch up first so the
        # simulated stream is never stale just because the event loop was busy.
        if self.connected and self.t0 is not None:
            self.advance(time.perf_counter())
        ring = self.ring
        return (self.connected and ring.received_samples >= self.eeg.fs // 2
                and 0 <= time.perf_counter() - ring.last_arrival <= max_age_s)

    def readiness_issue(self) -> str:
        if not self.connected:
            return "simulation is not running"
        if self.ring.received_samples < self.eeg.fs // 2:
            return "waiting for the first half-second of simulated EEG"
        if time.perf_counter() - self.ring.last_arrival > .25:
            return "simulated EEG frames are stale"
        return "ready"

    def mark_image(self, paint_stamp: float) -> tuple[int, float]:
        """Put the image onset on the simulator's sample grid; it has no marker transport delay."""
        code = self.place_marker(paint_stamp)
        if not self.stream_ready():
            raise RuntimeError("simulated EEG stream is not ready")
        return code, 0.0

    def close(self):
        self._timer.stop()
        if self.connected:
            self.connected = False
            self.connected_changed.emit(False)

    # --- simulation -------------------------------------------------------
    def start(self, t0: float | None = None):
        self.t0 = time.perf_counter() if t0 is None else t0
        self.generated = 0
        self.connected = True
        self._timer.start()
        self.connected_changed.emit(True)
        self.status_changed.emit(f"Simulated EEG · {self.eeg.fs} Hz · effect {self.effect_uv:g} µV "
                                 "(no headset; for testing only)")

    def place_marker(self, stamp: float) -> int:
        """Register a marker at the sample that corresponds to ``stamp``; returns its code."""
        self._next_marker += 1
        index = round((stamp - self.t0) * self.eeg.fs)
        if index < self.generated:
            # That sample was already streamed: record the marker at its exact index.
            with self.ring._lock:
                self.ring._markers[self._next_marker] = index
        else:
            self._pending_markers[index] = self._next_marker
        return self._next_marker

    def mark_stimulus(self, onset: float, duration_s: float, labels: dict[str, int]):
        """Called at image onset so the image's labels can shape the coming samples."""
        self.stimuli.append((onset, onset + duration_s, dict(labels)))

    def _tick(self):
        self.advance(time.perf_counter())

    def advance(self, now: float):
        """Generate every sample due up to ``now`` (perf_counter seconds)."""
        if self.t0 is None:
            return
        total = int((now - self.t0) * self.eeg.fs)
        n = total - self.generated
        if n <= 0:
            return
        indices = self.generated + np.arange(n)
        times = self.t0 + indices / self.eeg.fs
        frame = np.zeros((n, DEVICE_CHANNELS + 1))
        frame[:, :DEVICE_CHANNELS] = self._generate(times)
        for index in [i for i in self._pending_markers if i < total]:
            frame[index - self.generated, MARKER_COLUMN] = self._pending_markers.pop(index)
        self.generated = total
        self.ring.append(frame, marker_channel=MARKER_COLUMN)
        horizon = times[-1] - self.retention_s
        self.stimuli = [s for s in self.stimuli if s[1] + 1.0 >= horizon]

    def _generate(self, times: np.ndarray) -> np.ndarray:
        n = len(times)
        white = self.rng.normal(0, 2.0, (n, DEVICE_CHANNELS))
        slow, self._ar_state = lfilter([1.0], [1.0, -0.95], self.rng.normal(0, 2.0, (n, DEVICE_CHANNELS)),
                                       axis=0, zi=self._ar_state * 0.95)
        t = times[:, None]
        data = slow + white
        data += 4.0 * np.sin(2 * np.pi * 10.0 * t + self._phase)
        data += 2.0 * np.sin(2 * np.pi * 50.0 * t)
        for onset, end, labels in self.stimuli:
            if end < times[0] - 1.0 or onset > times[-1]:
                continue
            # ERP-like positivity ~300 ms after every image (no label information).
            data += 6.0 * np.exp(-0.5 * ((t - onset - 0.3) / 0.08) ** 2)
            shown = ((t >= onset) & (t < end)).astype(np.float64)
            for i, name in enumerate(self.target_names):
                if labels.get(name):
                    frequency = EFFECT_FREQUENCIES_HZ[i % len(EFFECT_FREQUENCIES_HZ)]
                    data += shown * self.effect_uv * np.sin(2 * np.pi * frequency * t + self._phase)
        return data
