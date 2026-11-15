import numpy as np
from PIL import ImageChops

from hotbrain.face_pilot import FACE_COUNT, FacePilot
from hotbrain.face_signal import BASE_HZ, BLOCK_S, tagged_response
from hotbrain.face_stimuli import render_face


def test_generated_faces_keep_identity_and_change_expression():
    neutral = render_face(0, "neutral", 0)
    happy = render_face(0, "happy", 80)
    sad = render_face(0, "sad", 80)
    assert ImageChops.difference(neutral, happy).getbbox()
    assert ImageChops.difference(neutral, sad).getbbox()
    assert ImageChops.difference(happy, sad).getbbox()
    assert FACE_COUNT == 120 and BASE_HZ * BLOCK_S == 120


def test_frequency_tag_detects_coherent_1p2_hz_component():
    rng = np.random.default_rng(4)
    t = np.arange(BLOCK_S * 250) / 250
    eeg = rng.normal(0, 3, (len(t), 8))
    eeg[:, 5:8] += 6 * np.sin(2 * np.pi * 1.2 * t)[:, None]
    result = tagged_response(eeg)
    assert result["ratio"] > 1.7
    assert result["provisional_response"]


def test_pilot_plan_stays_within_15_minutes():
    # The plan itself does not need a device or display.
    blocks = FacePilot._make_plan(None)
    assert len(blocks) == 19
    assert sum(b.phase == "pre" for b in blocks) == 4
    assert sum(b.phase == "train" for b in blocks) == 10
    assert sum(b.phase == "post" for b in blocks) == 4
    # Each quiz has at most three six-second response stages plus 2.5 s display.
    worst_case = len(blocks) * BLOCK_S + 18 * (3 * 6 + 5 * 0.5 + 1.5)
    assert worst_case < 15 * 60
