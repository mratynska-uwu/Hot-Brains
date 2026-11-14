import numpy as np

from hotbrain.config import CONFIG, EEGSettings
from hotbrain.quality import SignalQuality

EEG = EEGSettings.from_dict(CONFIG["eeg"])
FS = EEG.fs
N = len(EEG.electrodes)


def test_each_electrode_is_checked_separately():
    t = np.arange(2 * FS) / FS
    samples = np.tile((15 * np.sin(2 * np.pi * 10 * t))[:, None], (1, N))
    samples[:, 2] = 0
    samples[:, 6] = 500 * np.sin(2 * np.pi * 10 * t)

    ratings = SignalQuality(EEG).assess(samples)
    assert [rating.good for rating in ratings] == [True, True, False, True, True, True, False, True]
    assert "Flat" in ratings[2].detail
    assert "large" in ratings[6].detail


def test_quality_waits_for_enough_samples_and_flags_invalid_channel():
    quality = SignalQuality(EEG)
    assert all(r.good is None for r in quality.assess(np.zeros((FS, N))))
    t = np.arange(2 * FS) / FS
    samples = np.tile((15 * np.sin(2 * np.pi * 10 * t))[:, None], (1, N))
    samples[0, 4] = np.nan
    ratings = quality.assess(samples)
    assert ratings[4].good is False
    assert ratings[0].good is True
