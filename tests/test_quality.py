import numpy as np

from hotbrain.features import FS
from hotbrain.quality import assess_signal_quality


def test_each_electrode_is_checked_separately():
    t = np.arange(2 * FS) / FS
    samples = np.tile((15 * np.sin(2 * np.pi * 10 * t))[:, None], (1, 8))
    samples[:, 2] = 0
    samples[:, 6] = 500 * np.sin(2 * np.pi * 10 * t)

    ratings = assess_signal_quality(samples)
    assert [rating.good for rating in ratings] == [True, True, False, True, True, True, False, True]
    assert "Flat" in ratings[2].detail
    assert "large" in ratings[6].detail


def test_quality_waits_for_enough_samples_and_flags_invalid_channel():
    assert all(r.good is None for r in assess_signal_quality(np.zeros((FS, 8))))
    samples = np.ones((2 * FS, 8))
    samples[0, 4] = np.nan
    assert assess_signal_quality(samples)[4].good is False
