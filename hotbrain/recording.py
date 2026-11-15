"""Append-only Unicorn session capture, independent of model saving.

The numeric stream is the output of g.Pype's HybridBlack source. With OSCAR
enabled it has already passed through OSCAR; it is not the amplifier's ADC data.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


COLUMNS = ("ring_sample", "stream_generation", *(f"eeg_{i + 1}_uV" for i in range(8)),
           "battery", "counter", "scan_valid", "marker")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


class SessionRecorder:
    """Keep acquisition callbacks free of file I/O; persist all session scans."""

    def __init__(self, directory: Path, participant: str, session: str, metadata: dict):
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in participant) or "unnamed"
        self.stem = directory / f"{session}_{safe}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        self.data_path = self.stem.with_suffix(".eeg.f64")
        self.events_path = self.stem.with_suffix(".events.jsonl")
        self.meta_path = self.stem.with_suffix(".meta.json")
        self.session_path = self.stem.with_suffix(".session.json")
        self._queue: queue.Queue[tuple[str, object] | None] = queue.Queue(maxsize=5000)
        self._lock = threading.Lock()
        self._accepting = True
        self.rows = 0
        self.dropped_blocks = 0
        self.error: str | None = None
        self.metadata = {"format": "interleaved_rows", "dtype": "<f8", "columns": COLUMNS,
                         "fs": metadata["fs"], "oscar_enabled": metadata["oscar_enabled"],
                         "source": "g.Pype HybridBlack output before app feature filters",
                         "started_at": metadata["started_at"], "complete": False,
                         "data_file": self.data_path.name, "events_file": self.events_path.name,
                         "session_file": self.session_path.name}
        # Create the manifest and trial settings before the first scan. A crash
        # leaves a readable partial binary stream and an image/label event log.
        _atomic_json(self.meta_path, self.metadata)
        _atomic_json(self.session_path, metadata)
        self._thread = threading.Thread(target=self._write, name="eeg-recording", daemon=True)
        self._thread.start()

    def append(self, frame: np.ndarray, first_sample: int, generation: int) -> None:
        rows = np.empty((len(frame), len(COLUMNS)), dtype="<f8")
        rows[:, 0] = np.arange(first_sample, first_sample + len(frame))
        rows[:, 1] = generation
        rows[:, 2:] = frame
        with self._lock:
            if not self._accepting:
                return
            try:
                self._queue.put_nowait(("rows", rows))
            except queue.Full:
                self.dropped_blocks += 1

    def event(self, value: dict) -> None:
        with self._lock:
            if not self._accepting:
                raise RuntimeError("EEG recorder is closed")
            # Losing a trial label would make recorded EEG unusable. Stop the
            # session if the writer cannot keep up rather than silently omit it.
            self._queue.put_nowait(("event", dict(value)))

    def _write(self) -> None:
        try:
            with self.data_path.open("xb") as data, self.events_path.open("x", encoding="utf-8") as events:
                last_flush = time.monotonic()
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    kind, value = item
                    if kind == "rows":
                        data.write(value.tobytes())
                        self.rows += len(value)
                    else:
                        events.write(json.dumps(value, default=str) + "\n")
                        events.flush()
                    if time.monotonic() - last_flush >= 1:
                        data.flush()
                        last_flush = time.monotonic()
                data.flush()
                events.flush()
                os.fsync(data.fileno())
                os.fsync(events.fileno())
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def close(self, session_metadata: dict) -> None:
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        self.metadata.update({"rows": self.rows, "dropped_blocks": self.dropped_blocks,
                              "complete": self.error is None and self.dropped_blocks == 0,
                              "error": self.error,
                              "stopped_at": datetime.now().astimezone().isoformat()})
        _atomic_json(self.meta_path, self.metadata)
        _atomic_json(self.session_path, session_metadata)


def load_recording(meta_path: Path) -> tuple[np.ndarray, dict]:
    """Read the complete rows, including invalid scans and marker samples."""
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = np.fromfile(meta_path.parent / metadata["data_file"], dtype=metadata["dtype"])
    width = len(metadata["columns"])
    return rows[:len(rows) // width * width].reshape(-1, width), metadata
