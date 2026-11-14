"""g.Pype Unicorn acquisition with retry and a bounded, timestamped EEG ring.

The host-side timestamps are estimates from frame arrival and documented
device/OSCAR delays. They are suitable for prototyping, not hardware-verified
stimulus timing. A photodiode/trigger check is needed for research use.
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
    """Keeps the configured electrodes (device_channels) of the last seconds of EEG."""

    def __init__(self, retention_s: float = 20, fs: int = 250, channels: tuple[int, ...] = tuple(range(8))):
        self.fs = fs
        self.channels = list(channels)
        self._lock = threading.Lock()
        self._blocks: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self._latest = 0.0
        self._last_arrival = 0.0
        self._received_samples = 0
        self._end_time: float | None = None
        self.retention_s = retention_s

    def append(self, frame: np.ndarray, delay_s: float):
        raw = np.asarray(frame, dtype=np.float64)
        if raw.ndim != 2 or not len(raw) or raw.shape[1] <= max(self.channels):
            raise ValueError("Invalid EEG frame shape")
        values = raw[:, self.channels].copy()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite EEG samples")
        arrival = time.perf_counter()
        observed_end = arrival - delay_s
        with self._lock:
            # Bluetooth frames arrive in bursts. Re-timestamping every frame
            # from callback time creates artificial 20-70 ms gaps. Anchor the
            # first frame, then advance by the amplifier's fixed sampling rate.
            end = observed_end if self._end_time is None else self._end_time + len(values) / self.fs
            times = end - np.arange(len(values) - 1, -1, -1) / self.fs
            self._blocks.append((times, values))
            self._latest = end
            self._last_arrival = arrival
            self._received_samples += len(values)
            self._end_time = end
            while self._blocks and self._blocks[0][0][-1] < end - self.retention_s:
                self._blocks.popleft()

    @property
    def latest(self) -> float:
        with self._lock:
            return self._latest

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
            blocks = [(t.copy(), x.copy()) for t, x in self._blocks if t[-1] >= self._latest - seconds]
        if not blocks:
            return np.empty(0), np.empty((0, len(self.channels)))
        return np.concatenate([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks])

    def epoch(self, onset: float, spec: EpochSpec) -> np.ndarray | None:
        targets = onset + (START_MS / 1000) + np.arange(spec.n_samples) / spec.fs
        with self._lock:
            blocks = [(t.copy(), x.copy()) for t, x in self._blocks if t[-1] >= targets[0] - 0.1 and t[0] <= targets[-1] + 0.1]
        if not blocks:
            return None
        times = np.concatenate([b[0] for b in blocks])
        values = np.concatenate([b[1] for b in blocks])
        order = np.argsort(times)
        times, values = times[order], values[order]
        keep = np.r_[True, np.diff(times) > 1e-5]
        times, values = times[keep], values[keep]
        if times[0] > targets[0] or times[-1] < targets[-1]:
            return None
        # A Bluetooth dropout cannot be safely interpolated into an epoch.
        inside = (times >= targets[0] - .02) & (times <= targets[-1] + .02)
        if np.diff(times[inside]).max(initial=0) > 6 / spec.fs:
            return None
        return np.column_stack([np.interp(targets, times, values[:, ch]) for ch in range(values.shape[1])])


class DeviceManager(QObject):
    status_changed = Signal(str)
    connected_changed = Signal(bool)
    _attempt_succeeded = Signal(object, object, int)
    _attempt_failed = Signal(str, int)

    def __init__(self, eeg: EEGSettings, serial: str | None = None):
        super().__init__()
        self.eeg = eeg
        self.retention_s = 20.0
        self.ring = self._new_ring()
        self.pipeline = None
        self.source = None
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

    def stream_ready(self, max_age_s: float = 1.0) -> bool:
        """Require a current SDK stream and enough samples for the pre-onset baseline."""
        ring = self.ring
        return (self.connected and ring.received_samples >= self.eeg.fs // 2
                and 0 <= time.perf_counter() - ring.last_arrival <= max_age_s)

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
            source = gp.HybridBlack(serial=serial, enable_oscar=oscar, include_accel=False,
                                    include_gyro=False, include_aux=False, frame_size=4)

            class RingSink(gp.Sink):
                def __init__(self):
                    super().__init__(input_ports=[gp.IPort.Configuration()])

                def step(self, data):
                    frame = data.get("in")
                    ring = manager.ring
                    if generation == manager._generation and frame is not None and len(frame):
                        oscar_samples = source.oscar_delay_samples or 0
                        try:
                            ring.append(frame, .040 + oscar_samples / manager.eeg.fs)
                        except ValueError:
                            # Invalid SDK frames do not count as a live stream.
                            pass
                    return {}

            pipeline = gp.Pipeline()
            pipeline.connect(source, RingSink())
            pipeline.start()
            self._attempt_succeeded.emit(pipeline, source, generation)
        except Exception as exc:
            if pipeline is not None:
                try:
                    pipeline.stop()
                    pipeline.close()
                except Exception:
                    pass
            self._attempt_failed.emit(f"{type(exc).__name__}: {exc}", generation)

    def _connected(self, pipeline, source, generation: int):
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
        self.ring = self._new_ring()
        if was_connected:
            self.connected_changed.emit(False)

    def close(self):
        self._retry.stop()
        self.disconnect()
