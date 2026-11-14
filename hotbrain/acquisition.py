"""g.Pype Unicorn acquisition with sample-aligned stimulus markers.

Classification uses markers placed on g.Pype's amplifier sample timeline,
not host estimates of when Bluetooth frames were acquired. Physical display
onset still needs a photodiode measurement for research-grade timing.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from .config import EEGSettings
from .features import EpochSpec, START_MS


class SampleRing:
    """Keep EEG and trial markers in the same integer sample coordinate system."""

    def __init__(self, retention_s: float = 20, fs: int = 250, channels: tuple[int, ...] = tuple(range(8)),
                 device_delay_ms: float = 40):
        self.fs = fs
        self.channels = list(channels)
        # HybridBlack.DEVICE_DELAY_MS: host receipt trails acquisition. g.Pype
        # fits to receipt stamps and explicitly does not remove device delay.
        self.device_delay_samples = round(device_delay_ms * fs / 1000)
        self._lock = threading.Lock()
        self._blocks: deque[tuple[int, np.ndarray, np.ndarray]] = deque()
        self._markers: dict[int, int] = {}
        self._last_arrival = 0.0
        self._received_samples = 0
        self.retention_s = retention_s

    def append(self, frame: np.ndarray, marker_channel: int | None = None,
               validation_channel: int | None = None):
        raw = np.asarray(frame, dtype=np.float64)
        if raw.ndim != 2 or not len(raw) or raw.shape[1] <= max(self.channels):
            raise ValueError("Invalid EEG frame shape")
        values = raw[:, self.channels].copy()
        finite = np.isfinite(values).all(axis=1)
        if marker_channel is not None:
            if marker_channel >= raw.shape[1] or marker_channel in self.channels:
                raise ValueError("Invalid marker channel")
            markers = raw[:, marker_channel]
            if not np.isfinite(markers).all():
                raise ValueError("Non-finite trial markers")
        else:
            markers = np.zeros(len(raw))
        if validation_channel is not None:
            if validation_channel >= raw.shape[1] or validation_channel in self.channels:
                raise ValueError("Invalid validation channel")
            valid = (raw[:, validation_channel] == 1) & finite
        else:
            valid = finite
        arrival = time.perf_counter()
        with self._lock:
            first = self._received_samples
            self._blocks.append((first, values, valid.copy()))
            for offset in np.flatnonzero(markers):
                code = markers[offset]
                if code > 0 and code.is_integer():
                    self._markers[int(code)] = first + int(offset)
            self._last_arrival = arrival
            self._received_samples += len(values)
            oldest = self._received_samples - round(self.retention_s * self.fs)
            while self._blocks and self._blocks[0][0] + len(self._blocks[0][1]) <= oldest:
                self._blocks.popleft()
            first_retained = self._blocks[0][0]
            self._markers = {code: index for code, index in self._markers.items() if index >= first_retained}

    @property
    def latest(self) -> float:
        """Latest frame arrival time, for the display only; never used for epochs."""
        with self._lock:
            return self._last_arrival

    @property
    def last_arrival(self) -> float:
        """Host time of the latest real SDK frame, independent of estimated EEG timestamps."""
        with self._lock:
            return self._last_arrival

    @property
    def received_samples(self) -> int:
        with self._lock:
            return self._received_samples

    def recent(self, seconds: float = 4) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            last = self._received_samples
            first = max(0, last - round(seconds * self.fs))
            blocks = [(max(i, first), x[max(0, first - i):].copy(), v[max(0, first - i):].copy())
                      for i, x, v in self._blocks
                      if i + len(x) > first]
            arrival = self._last_arrival
        if not blocks:
            return np.empty(0), np.empty((0, len(self.channels)))
        indices = np.concatenate([np.arange(i, i + len(x)) for i, x, _ in blocks])
        # Approximate times label the plot only. Epochs use marker sample indices.
        times = arrival + (indices - (last - 1)) / self.fs
        values = np.concatenate([x for _, x, _ in blocks])
        valid = np.concatenate([v for _, _, v in blocks])
        values[~valid] = np.nan
        return times, values

    def valid_recent(self, seconds: float = .5) -> bool:
        """Require genuine scans throughout the latest readiness window."""
        needed = round(seconds * self.fs)
        with self._lock:
            if self._received_samples < needed:
                return False
            found = 0
            for _, _, valid in reversed(self._blocks):
                take = min(needed - found, len(valid))
                if not valid[-take:].all():
                    return False
                found += take
                if found == needed:
                    return True
        return False

    def any_valid_recent(self, seconds: float = 1.0) -> bool:
        """Distinguish a continuing real EEG stream from filled-in scans only."""
        needed = round(seconds * self.fs)
        with self._lock:
            found = 0
            for _, _, valid in reversed(self._blocks):
                take = min(needed - found, len(valid))
                if valid[-take:].any():
                    return True
                found += take
                if found >= needed:
                    break
        return False

    def marker_sample(self, code: int) -> int | None:
        with self._lock:
            return self._markers.get(code)

    def eeg_onset_sample(self, code: int, paint_to_marker_ms: float = 0) -> int | None:
        marker = self.marker_sample(code)
        return (None if marker is None else marker + self.device_delay_samples
                - round(paint_to_marker_ms * self.fs / 1000))

    def epoch(self, code: int, spec: EpochSpec, paint_to_marker_ms: float = 0) -> np.ndarray | None:
        return self.epoch_result(code, spec, paint_to_marker_ms)[0]

    def epoch_result(self, code: int, spec: EpochSpec,
                     paint_to_marker_ms: float = 0) -> tuple[np.ndarray | None, str]:
        """Return a completed epoch or why it cannot yet be used."""
        if spec.fs != self.fs:
            raise ValueError("Epoch sample rate differs from acquisition")
        with self._lock:
            marker = self._markers.get(code)
            if marker is None or not self._blocks:
                return None, "waiting_marker"
            first = (marker + self.device_delay_samples
                     - round(paint_to_marker_ms * self.fs / 1000)
                     + round(START_MS * self.fs / 1000))
            last = first + spec.n_samples
            if first < self._blocks[0][0]:
                return None, "expired_epoch"
            if last > self._received_samples:
                return None, "waiting_samples"
            pieces = [(x[max(0, first - i):min(len(x), last - i)].copy(),
                       v[max(0, first - i):min(len(v), last - i)].copy())
                      for i, x, v in self._blocks if i < last and i + len(x) > first]
        result = np.concatenate([x for x, _ in pieces]) if pieces else np.empty((0, len(self.channels)))
        if len(result) != spec.n_samples:
            return None, "incomplete_epoch"
        if not all(v.all() for _, v in pieces):
            return None, "invalid_scans"
        return result, "ready"


class DeviceManager(QObject):
    status_changed = Signal(str)
    connected_changed = Signal(bool)
    _attempt_succeeded = Signal(object, object, object, int)
    _attempt_failed = Signal(str, int)

    def __init__(self, eeg: EEGSettings, serial: str | None = None):
        super().__init__()
        self.eeg = eeg
        self.retention_s = 20.0
        self.ring = self._new_ring()
        self.pipeline = None
        self.source = None
        self.marker = None
        self._next_marker = 0
        self.connected = False
        self.oscar = True
        self.serial = serial
        self._last_attempt = 0.0
        self._connected_at = 0.0
        self._last_error = ""
        self._attempting = False
        self._generation = 0
        self._attempt_succeeded.connect(self._connected)
        self._attempt_failed.connect(self._failed)
        self._retry = QTimer(self)
        self._retry.setInterval(3000)
        self._retry.timeout.connect(self._tick)
        self._retry.start()
        QTimer.singleShot(0, self._tick)

    def configure(self, oscar: bool, serial: str | None = None):
        if self.oscar != oscar or self.serial != serial:
            self.oscar = oscar
            self.serial = serial
            self._attempting = False
            self.disconnect()
            self._last_attempt = 0
            self._tick()

    def _new_ring(self) -> SampleRing:
        return SampleRing(self.retention_s, self.eeg.fs, self.eeg.device_channels)

    def set_retention(self, seconds: float):
        self.retention_s = max(20.0, seconds)
        self.ring.retention_s = self.retention_s

    def stream_ready(self, max_age_s: float = .25) -> bool:
        """Require a current SDK stream and enough samples for the pre-onset baseline."""
        ring = self.ring
        return self.stream_live(max_age_s) and ring.valid_recent(.5)

    def stream_live(self, max_age_s: float = .5) -> bool:
        """Keep trials running through isolated bad scans, but not a real stream loss."""
        ring = self.ring
        return (self.connected and ring.received_samples >= self.eeg.fs // 2
                and 0 <= time.perf_counter() - ring.last_arrival <= max_age_s
                and ring.any_valid_recent(1.0)
                and self.pipeline is not None
                and self.pipeline.get_condition() in ("Healthy", "Warning"))

    def readiness_issue(self) -> str:
        if not self.connected or self.pipeline is None:
            return "Unicorn is not connected"
        condition = self.pipeline.get_condition()
        if condition != "Healthy":
            return f"g.Pype pipeline is {condition}"
        if self.ring.received_samples < self.eeg.fs // 2:
            return "waiting for the first half-second of EEG"
        if time.perf_counter() - self.ring.last_arrival > .25:
            return "EEG frames are stale"
        if not self.ring.any_valid_recent(1.0):
            return "no real Unicorn scans in the last second"
        if not self.ring.valid_recent(.5):
            return "waiting for 0.5 s of clean EEG before starting"
        return "ready"

    def mark_image(self, paint_stamp: float) -> tuple[int, float]:
        """Put this image onset on the amplifier's sample grid via g.Pype."""
        if not self.stream_live() or self.marker is None:
            raise RuntimeError("EEG timing stream is not ready")
        if time.perf_counter() - paint_stamp > .020:
            raise RuntimeError("image paint-to-marker delay exceeded 20 ms")
        self._next_marker += 1
        emission_age_ms = (time.perf_counter() - paint_stamp) * 1000
        self.marker.emit(self._next_marker)
        completion_age_ms = (time.perf_counter() - paint_stamp) * 1000
        if completion_age_ms > 20:
            raise RuntimeError(f"image paint-to-marker delay was {completion_age_ms:.1f} ms")
        return self._next_marker, emission_age_ms

    def _tick(self):
        now = time.perf_counter()
        if self.connected:
            # A stream that stopped producing data has disconnected.
            if now - max(self.ring.last_arrival, self._connected_at) > 2.5:
                self.status_changed.emit("EEG stream stalled; reconnecting…")
                self.disconnect()
            else:
                return
        if self._attempting or now - self._last_attempt < 3:
            return
        self._last_attempt = now
        self._attempting = True
        generation = self._generation
        oscar = self.oscar
        serial = self.serial
        threading.Thread(target=self._connect_worker, args=(generation, oscar, serial), daemon=True).start()

    def _connect_worker(self, generation: int, oscar: bool, serial: str | None):
        pipeline = None
        try:
            import gpype as gp

            manager = self
            # The Unicorn driver can fill missing scans. Its validation flag
            # (column 10 here) is essential for rejecting those samples.
            source = gp.HybridBlack(serial=serial, enable_oscar=oscar, include_accel=False,
                                    include_gyro=False, include_aux=True, frame_size=4)
            marker = gp.Marker()
            router = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])

            class RingSink(gp.Sink):
                def __init__(self):
                    super().__init__(input_ports=[gp.IPort.Configuration()])

                def step(self, data):
                    frame = data.get("in")
                    ring = manager.ring
                    if generation == manager._generation and frame is not None and len(frame):
                        # Router appends the aligned marker channel after the
                        # eight EEG and three auxiliary channels.
                        if frame.shape[1] != 12:
                            raise ValueError("Expected 8 EEG, 3 auxiliary, and 1 marker channel")
                        ring.append(frame, marker_channel=11, validation_channel=10)
                    return {}

            pipeline = gp.Pipeline()
            pipeline.connect(source, router["in1"])
            pipeline.connect(marker, router["in2"])
            pipeline.connect(router, RingSink())
            pipeline.start()
            self._attempt_succeeded.emit(pipeline, source, marker, generation)
        except Exception as exc:
            if pipeline is not None:
                try:
                    pipeline.stop()
                    pipeline.close()
                except Exception:
                    pass
            self._attempt_failed.emit(f"{type(exc).__name__}: {exc}", generation)

    def _connected(self, pipeline, source, marker, generation: int):
        if generation != self._generation:
            try:
                pipeline.stop()
                pipeline.close()
            except Exception:
                pass
            return
        self._attempting = False
        self.pipeline = pipeline
        self.source = source
        self.marker = marker
        self.connected = True
        self._connected_at = time.perf_counter()
        self.connected_changed.emit(True)
        self.status_changed.emit("Unicorn " + (self.serial or "(first available)") +
                                 " opened; waiting for EEG samples · OSCAR " + ("on" if self.oscar else "off"))

    def _failed(self, reason: str, generation: int):
        if generation != self._generation:
            return
        self._attempting = False
        message = f"EEG unavailable; retrying ({reason})"
        if message != self._last_error:
            self.status_changed.emit(message)
            self._last_error = message

    def disconnect(self):
        self._generation += 1
        was_connected = self.connected
        self.connected = False
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            try:
                self.pipeline.close()
            except Exception:
                pass
        self.pipeline = None
        self.source = None
        self.marker = None
        self.ring = self._new_ring()
        if was_connected:
            self.connected_changed.emit(False)

    def close(self):
        self._retry.stop()
        self.disconnect()
