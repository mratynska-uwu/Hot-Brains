import json

import numpy as np

from hotbrain.recording import COLUMNS, SessionRecorder, load_recording


def test_session_recorder_keeps_invalid_scans_markers_and_events(tmp_path):
    run = {"fs": 250, "oscar_enabled": True, "started_at": "now", "participant_id": "p/1"}
    recorder = SessionRecorder(tmp_path, "p/1", "training", run)
    frame = np.zeros((4, 12), dtype=float)
    frame[:, :8] = np.arange(32).reshape(4, 8)
    frame[:, 10] = [1, 0, 1, 1]
    frame[2, 11] = 7
    recorder.append(frame, 123, 3)
    recorder.event({"kind": "image_onset", "number": 1, "marker_code": 7, "image": "joy.png"})
    run["trial_log"] = [{"number": 1, "eeg": "invalid_scans"}]
    recorder.close(run)

    rows, metadata = load_recording(recorder.meta_path)
    assert rows.shape == (4, len(COLUMNS))
    assert rows[:, 0].tolist() == [123, 124, 125, 126]
    assert np.all(rows[:, 1] == 3)
    assert rows[:, 12].tolist() == [1, 0, 1, 1]
    assert rows[2, 13] == 7
    assert metadata["complete"] and metadata["rows"] == 4
    assert json.loads(recorder.events_path.read_text().strip())["image"] == "joy.png"
    assert json.loads(recorder.session_path.read_text())["trial_log"][0]["eeg"] == "invalid_scans"
