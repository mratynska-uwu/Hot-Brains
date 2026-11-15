"""Simulated EEG source for testing the whole pipeline without a headset.

This is the one, explicitly opt-in exception to "the app never generates
substitute EEG". It exists only to test the GUI, sessions, features, reports
and classifiers. It is used only when the operator picks **EEG source →
Simulated EEG** or starts the app with ``run.py --simulate``. Every simulated
run is marked ``"eeg_source": "simulated"`` in its manifest, and its saved
files and reports start with ``SIMULATED_``.

``SimulatedDevice`` has the interface of ``acquisition.DeviceManager``
(``stream_ready``, ``stream_live``, ``readiness_issue``, ``mark_image``,
``ring`` ...). It streams the eight Unicorn channels in real time into the same
marker-aligned ``SampleRing``. Image and cue markers are written to a marker
column on the simulator's own sample grid at the Qt paint time, so epochs are
cut exactly as with the headset (without the Unicorn's 40 ms device delay).

Signal model (µV, per electrode; electrodes not named below get small weights):

* Background: slow AR(1) drift, white noise, a posterior-dominant 10 Hz alpha
  rhythm with slow amplitude waxing and waning, weak frontal theta and central
  beta, 50 Hz mains hum (removed by the notch) and, optionally, eye blinks at
  Fz every few seconds.
* Every image or cue: a visual evoked response (P1, N1 posterior; P3 parietal).
  It carries no label information.
* Scene images, from the image's labels (``effect_uv`` scales every effect):
  - arousal 1 (emotional): early posterior negativity (EPN, ~250 ms), a
    centro-parietal late positive potential (LPP, 0.45-1.4 s) and posterior
    alpha suppression while the image is shown;
  - valence 1 (positive): left (C3, PO7) alpha suppression and more
    fronto-central 20 Hz beta while the image is shown;
  - valence 0 (negative): right (C4, PO8) alpha suppression and a larger early
    negativity (~200 ms, negativity bias);
  - no label (e.g. neutral images for valence): nothing.
* Taste trials: jaw and tongue EMG 0.5-4 s after the cue (trips the EMG guard
  before the analysis window), then from +2 s to the end of tasting:
  sweet = more frontal-midline theta and right-central alpha suppression,
  salty = more central low beta (16 Hz).

With ``effect_uv = 0`` the EEG carries no label information, so the
classifiers should stay at chance: a useful check of the evaluation itself.
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
TICK_MS = 40
DEFAULT_EFFECT_UV = 4.0
TASTE_SWEET = 1                           # hotbrain.taste.SWEET (label 1); salty is 0

# Spatial weights by electrode name (10-20 positions of the Unicorn cap and neighbours).
POSTERIOR = {"Oz": 1.0, "O1": 1.0, "O2": 1.0, "PO7": .9, "PO8": .9, "PO3": .9, "PO4": .9, "Pz": .6,
             "P3": .7, "P4": .7, "Cz": .3, "C3": .3, "C4": .3, "Fz": .1}
PARIETAL = {"Pz": 1.0, "P3": .9, "P4": .9, "Cz": .7, "CPz": .9, "C3": .5, "C4": .5, "PO7": .6, "PO8": .6,
            "Oz": .4, "Fz": .3}
ALPHA_UV = {"Oz": 8.0, "O1": 8.0, "O2": 8.0, "PO7": 7.0, "PO8": 7.0, "Pz": 6.0, "P3": 6.0, "P4": 6.0,
            "Cz": 4.0, "C3": 4.0, "C4": 4.0, "Fz": 3.0}
FRONTAL_THETA = {"Fz": 1.0, "Cz": .8, "C3": .4, "C4": .4, "Pz": .3}
CENTRAL = {"C3": 1.0, "C4": 1.0, "Cz": .8, "Fz": .4, "Pz": .3}
BLINK = {"Fz": 1.0, "Cz": .35, "C3": .3, "C4": .3, "Pz": .15}
LEFT = {"C3": 1.0, "PO7": 1.0, "P3": 1.0, "O1": .8}
RIGHT = {"C4": 1.0, "PO8": 1.0, "P4": 1.0, "O2": .8}


def _weights(names: tuple[str, ...], table: dict[str, float], default: float = .1) -> np.ndarray:
    return np.asarray([table.get(name, default) for name in names], dtype=np.float64)


def _bump(tau: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((tau - center) / width) ** 2)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class SimulatedDevice(QObject):
    """Real-time synthetic EEG with the ``DeviceManager`` interface (testing only)."""

    status_changed = Signal(str)
    connected_changed = Signal(bool)
    simulated = True

    def __init__(self, eeg: EEGSettings, effect_uv: float = DEFAULT_EFFECT_UV, seed: int | None = None,
                 blinks: bool = True, autostart: bool = True):
        super().__init__()
        self.eeg = eeg
        self.effect_uv = float(effect_uv)
        self.blinks = blinks
        self.oscar = False
        self.serial = None
        self.retention_s = 20.0
        self.ring = SampleRing(self.retention_s, eeg.fs, eeg.device_channels, device_delay_ms=0)
        self.connected = False
        self.rng = np.random.default_rng(seed)
        self.t0: float | None = None
        self.generated = 0
        self.stimuli: list[tuple[float, float, dict, float]] = []
        self._pending_markers: dict[int, int] = {}
        self._next_marker = 0
        names = tuple(eeg.electrodes)
        n = len(names)
        # Electrode i is device column eeg.device_channels[i].
        self._columns = list(eeg.device_channels)
        self._w = {key: _weights(names, table, default) for key, table, default in (
            ("posterior", POSTERIOR, .1), ("parietal", PARIETAL, .2), ("alpha", ALPHA_UV, 3.0),
            ("theta", FRONTAL_THETA, .1), ("central", CENTRAL, .2), ("blink", BLINK, .05),
            ("left", LEFT, 0.0), ("right", RIGHT, 0.0))}
        self._ar_state = np.zeros((1, n))
        self._mid_state = np.zeros((1, n))
        self._env_state = np.zeros((1, n))
        # Alpha amplitude envelope: AR(1) with a ~0.7 s time constant, so alpha waxes and wanes.
        self._env_pole = float(np.exp(-1.0 / (0.7 * eeg.fs)))
        self._phase = self.rng.uniform(0, 2 * np.pi, (6, n))
        self._blink_times: list[tuple[float, float]] = []
        self._next_blink: float | None = None
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

    def _catch_up(self):
        # Samples are produced on a Qt timer, which pauses while the GUI does
        # blocking work (loading images, saving models). Catch up first so the
        # stream never looks stale just because the event loop was busy.
        if self.connected and self.t0 is not None:
            self.advance(time.perf_counter())

    def stream_live(self, max_age_s: float = .5) -> bool:
        self._catch_up()
        ring = self.ring
        return (self.connected and ring.received_samples >= self.eeg.fs // 2
                and 0 <= time.perf_counter() - ring.last_arrival <= max_age_s)

    def stream_ready(self, max_age_s: float = .25) -> bool:
        return self.stream_live(max_age_s) and self.ring.valid_recent(.5)

    def readiness_issue(self) -> str:
        if not self.connected:
            return "simulation is not running"
        if self.ring.received_samples < self.eeg.fs // 2:
            return "waiting for the first half-second of simulated EEG"
        if time.perf_counter() - self.ring.last_arrival > .25:
            return "simulated EEG frames are stale"
        return "ready"

    def mark_image(self, paint_stamp: float) -> tuple[int, float]:
        """Put an onset on the simulator's sample grid; there is no marker transport delay."""
        if not self.stream_live():
            raise RuntimeError("simulated EEG stream is not ready")
        return self.place_marker(paint_stamp), 0.0

    def disconnect(self):
        self._timer.stop()
        if self.connected:
            self.connected = False
            self.connected_changed.emit(False)

    def close(self):
        self.disconnect()

    # --- simulation -------------------------------------------------------
    def start(self, t0: float | None = None):
        self.t0 = time.perf_counter() if t0 is None else t0
        self.generated = 0
        self.connected = True
        self._timer.start()
        self.connected_changed.emit(True)
        self.status_changed.emit(self.description())

    def description(self) -> str:
        return (f"SIMULATED EEG · {self.eeg.fs} Hz · effect {self.effect_uv:g} µV · no headset, for testing only")

    def set_effect(self, effect_uv: float):
        self.effect_uv = float(effect_uv)

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

    def mark_stimulus(self, onset: float, duration_s: float, labels: dict):
        """Called at an image or cue onset so its labels can shape the coming samples.

        Recognised keys: ``valence`` and ``arousal`` (scene images) and
        ``taste`` (taste trials); a value of None means "no label".
        """
        # Trial-to-trial variability: each stimulus gets its own effect strength (mean 1, SD 0.5).
        gain = max(0.0, float(self.rng.normal(1.0, 0.5)))
        self.stimuli.append((float(onset), float(onset) + float(duration_s), dict(labels), gain))

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
        times = self.t0 + (self.generated + np.arange(n)) / self.eeg.fs
        frame = np.zeros((n, DEVICE_CHANNELS + 1))
        frame[:, self._columns] = self._generate(times)
        for index in [i for i in self._pending_markers if i < total]:
            frame[max(0, index - self.generated), MARKER_COLUMN] = self._pending_markers.pop(index)
        self.generated = total
        self.ring.append(frame, marker_channel=MARKER_COLUMN)
        horizon = times[-1] - self.retention_s
        self.stimuli = [s for s in self.stimuli if s[1] + 3.0 >= horizon]
        self._blink_times = [b for b in self._blink_times if b[0] + 1.0 >= times[0]]

    def _generate(self, times: np.ndarray) -> np.ndarray:
        n, w, ph = len(times), self._w, self._phase
        channels = len(self._columns)
        t = times[:, None]
        # AR(1) processes; lfilter's returned state continues them seamlessly across chunks.
        slow, self._ar_state = lfilter([1.0], [1.0, -0.97], self.rng.normal(0, 1.5, (n, channels)),
                                       axis=0, zi=self._ar_state)
        mid, self._mid_state = lfilter([1.0], [1.0, -0.85], self.rng.normal(0, 4.0, (n, channels)),
                                       axis=0, zi=self._mid_state)
        pole = self._env_pole
        env, self._env_state = lfilter([1.0], [1.0, -pole], self.rng.normal(0, np.sqrt(1 - pole ** 2), (n, channels)),
                                       axis=0, zi=self._env_state)
        data = slow + mid + self.rng.normal(0, 3.0, (n, channels))
        data += 3.0 * np.sin(2 * np.pi * 50.0 * t)
        data += 2.0 * w["theta"] * np.sin(2 * np.pi * 6.0 * t + ph[1])
        data += 1.5 * w["central"] * np.sin(2 * np.pi * 20.0 * t + ph[2])
        alpha_gain = np.ones((n, channels))
        for onset, end, labels, gain in self.stimuli:
            if end + 2.5 < times[0] or onset > times[-1]:
                continue
            tau = t - onset
            shown = ((tau >= 0) & (t < end)).astype(np.float64)
            data += self._visual_erp(tau)
            if "taste" in labels:
                data, alpha_gain = self._taste(data, alpha_gain, t, tau, end - onset, labels.get("taste"), gain)
            else:
                data, alpha_gain = self._scene(data, alpha_gain, t, tau, shown, labels, gain)
        modulation = np.clip(1.0 + 0.5 * env, 0.1, None)
        data += w["alpha"] * modulation * np.clip(alpha_gain, 0.2, 1.0) * np.sin(2 * np.pi * 10.0 * t + ph[0])
        if self.blinks:
            data += self._blinks(times)
        return data

    def _visual_erp(self, tau: np.ndarray) -> np.ndarray:
        w = self._w
        return (w["posterior"] * (3.0 * _bump(tau, .10, .025) - 4.0 * _bump(tau, .17, .03))
                + 4.0 * w["parietal"] * _bump(tau, .35, .08))

    def _scene(self, data, alpha_gain, t, tau, shown, labels, gain):
        w, e = self._w, self.effect_uv * gain
        arousal, valence = labels.get("arousal"), labels.get("valence")
        if arousal == 1:
            data = data - 0.6 * e * w["posterior"] * _bump(tau, .25, .04)
            lpp = _sigmoid((tau - .45) / .05) * _sigmoid((1.4 - tau) / .2)
            data = data + e * w["parietal"] * lpp
            alpha_gain = alpha_gain - 0.08 * e * w["posterior"] * shown * (tau >= .3)
        if valence == 1:
            alpha_gain = alpha_gain - 0.05 * e * w["left"] * shown
            data = data + 0.3 * e * w["central"] * shown * np.sin(2 * np.pi * 20.0 * t + self._phase[4])
        elif valence == 0:
            alpha_gain = alpha_gain - 0.05 * e * w["right"] * shown
            data = data - 0.4 * e * w["posterior"] * _bump(tau, .20, .05)
        return data, alpha_gain

    def _taste(self, data, alpha_gain, t, tau, duration, taste, gain):
        w, e = self._w, self.effect_uv * gain
        n, channels = data.shape
        # Hand, jaw and tongue muscle activity while the tastant is placed (label independent).
        emg = (_sigmoid((tau - .5) / .1) * _sigmoid((4.0 - tau) / .2))
        if emg.max() > 1e-3:
            data = data + 25.0 * emg * self.rng.normal(0, 1.0, (n, channels))
        sustained = _sigmoid((tau - 2.0) / .3) * _sigmoid((duration - tau) / .3)
        if taste == TASTE_SWEET:
            data = data + 0.6 * e * w["theta"] * sustained * np.sin(2 * np.pi * 6.0 * t + self._phase[5])
            alpha_gain = alpha_gain - 0.06 * e * w["right"] * sustained
        elif taste is not None:
            data = data + 0.5 * e * w["central"] * sustained * np.sin(2 * np.pi * 16.0 * t + self._phase[5])
        return data, alpha_gain

    def _blinks(self, times: np.ndarray) -> np.ndarray:
        """Eye blinks every ~2-6 s, mostly at Fz (40-90 µV)."""
        if self._next_blink is None:
            self._next_blink = times[0] + self.rng.uniform(1, 4)
        while self._next_blink <= times[-1] + .5:
            self._blink_times.append((self._next_blink, self.rng.uniform(40, 90)))
            self._next_blink += self.rng.uniform(2, 6)
        out = np.zeros((len(times), len(self._columns)))
        for center, amplitude in self._blink_times:
            if times[0] - .5 <= center <= times[-1] + .5:
                out += amplitude * _bump(times[:, None] - center, 0.0, .08) * self._w["blink"]
        return out


def is_simulated(device) -> bool:
    return bool(getattr(device, "simulated", False))
