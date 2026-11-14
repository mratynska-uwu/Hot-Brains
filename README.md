# Hot Brain scene training prototype

Participant-specific EEG decoding of emotional valence and arousal with the g.tec Unicorn headset.

## Project goals

1. **Train and validate a valence–arousal classifier.** Train it on 80% of the training dataset and validate it on the held-out 20%.
2. **Run a test session.** The participant sees images from a test dataset (images not chosen yet), each with a known valence and arousal. Their task is to bring their own valence and arousal to the image's values. The valence and arousal decoded from their EEG are shown live during the session.

The app has three session types for this: **Training**, **Validation** and **Test**. By default it trains separate **valence** and **arousal** models, one for each selected classifier (logistic regression, SVM, ...). Labels come from each image's metadata CSV. Each model predicts a trial from its EEG **before** that trial is added to its fit. No image pixels or labels enter any model as predictors.

The papers behind the design are summarised in Polish in [`docs/papers/`](docs/papers/), one `<paper>_podsumowanie.md` per paper. The list of possible improvements and a critical review of the current state are in [`docs/optymalizacje_i_ocena_krytyczna.md`](docs/optymalizacje_i_ocena_krytyczna.md).

## Sessions

Pick the session type in the **Session** selector of the training tab.

| Session | Images | Models | What you see |
|---|---|---|---|
| **Training** | the 80% training split; trials balanced across folders | learn after every trial (test-before-train) | rolling accuracy per target × classifier, linear weights |
| **Validation** | the 20% validation split, each image once | frozen: from the last training run or **Load saved models…** | next to the image, a valence–arousal plane with the target ◯ and the EEG estimate ● for each image; a table of accuracy, balanced accuracy and majority-class baseline |
| **Test** | `data/test` (each image once) | frozen | the image with its target ◯; the estimate ● updates continuously (every `feedback.update_s`, smoothed) with a trail and distance to the target |

Details:

- **Split.** The 80/20 split is stratified per folder and reproducible: `split.seed` in `CONFIG`, 114 + 59 training and 29 + 15 validation images. Validation images never appear in training.
- **Plane coordinates.**
  - Target ◯: the image's rating, scaled to [−1, 1] around the class threshold. 0 is the class boundary; ±1 is the most extreme image in the training dataset.
  - EEG estimate ●: `2·P(class 1) − 1` from the classifier chosen in **Live display uses**. It is the classifier's confidence, not a calibrated rating.
- **Test images.** Put them in `data/test/` with `data/test.csv` containing `image_name`, `valence` and `arousal` (same definitions as the training CSVs). Until then, the app offers to run the test session with the validation images, which is useful for checking the flow.
- **Test timing.** Set in `CONFIG["test_session"]`: `trial_s` (15 s) per image and `rest_s` (5 s) between images.
- **Image duration.** The models must have been trained with the same image duration, because it defines the features. The app checks this before starting.
- **Saving.**
  - Training saves the models.
  - Validation and test save `validation_*.json` / `test_*.json` in `saved_models/`. The file holds per-trial predictions and scores and, for test, the whole live decoding trace with the mean and final distance to the target for each trial.
  - Validation also saves its epochs.

## Run

On macOS Apple silicon, Windows x64, or Linux x64, use Python 3.10–3.14. Run from the `Hot-Brains` folder.

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-device.txt
.venv/bin/python run.py              # Unicorn headset
.venv/bin/python run.py --simulate   # simulated EEG, no headset
```

Windows (PowerShell or cmd):

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements-device.txt
.venv\Scripts\python run.py              # Unicorn headset
.venv\Scripts\python run.py --simulate   # simulated EEG, no headset
```

`requirements.txt` is enough without a headset, including simulation. `requirements-device.txt` adds the g.Pype device drivers. The app keeps retrying the Unicorn connection while the Qt loop remains responsive. On macOS, grant Bluetooth permission to the application running Python. The official g.Pype package supplies `HybridBlack` and the optional `enable_oscar=True` path. [g.Pype repository](https://github.com/gtec-neurotechnology/gpype) · [device reference](https://gpype.gtec.at/content/7_sdk_reference/backend_sources/hybrid_black.html) · [platform notes](https://gpype.gtec.at/content/1_basic_concepts/supported_devices.html)

### Simulated EEG

Tick **EEG source → Simulate EEG** in the training tab, or start with `--simulate`, to test the whole pipeline without a headset: presentation, epochs, features, signal checks, training and saving. The simulator (`hotbrain/simulation.py`) streams 8 channels at the configured rate in real time:

- background: low-frequency noise, white noise, a 10 Hz alpha rhythm and 50 Hz mains hum;
- every image: an ERP-like bump at ~300 ms;
- while an image whose target label is 1 is shown: an extra oscillation of **Simulated effect** µV, at 10 Hz for the 1st target (valence) and 20 Hz for the 2nd (arousal).

Sanity check: with the effect at 0 µV the EEG carries no label information, so accuracies should stay near 50%. With 5–8 µV they should climb well above chance within a few dozen trials.

Simulated runs are marked `"eeg_source": "simulated"` in the manifest, and their files start with `SIMULATED_`. They are for testing only, never participant data. OSCAR settling is skipped in simulation.

## Configuration (`hotbrain/config.py`)

Everything that differs between experiments lives in one dictionary, `CONFIG`. `run.py` passes it to `launch(root, config)`, so you can also pass a modified copy. It is validated at startup.

| Key | Default | Meaning |
|---|---|---|
| `datasets` | `["training/joy", "training/sadness"]` | Subfolders of `data_dir` (`data/`). Each one needs a sibling CSV with the same name (`data/training/joy.csv`), one row per image. Trials are balanced across these folders. |
| `image_column` | `image_name` | CSV column holding the file name. |
| `eeg` | from `basic_info.md` | `sampling_rate` 250, `electrodes` Fz, C3, Cz, C4, P3, Pz, P4, Oz, `bandpass_hz` 1–48, `notch_hz` 50, `bands_hz` alpha/beta/gamma, `features` band power + fractal dimension. Also `regions`, `asymmetry_pairs`, `erp_windows_ms`, `fractal_kmax` and `device_channels`. |
| `targets` | valence, arousal | Each target gets one value per image: a single CSV column (`"column": "aroused_mean"`) or a weighted sum of columns (`"columns": {"positive_mean": 1, "negative_mean": -1}`). The binary label comes from that value. With `threshold: None` the value must already be 0/1. A number or `"median"` makes label 1 mean the value is above the threshold. To analyse something else, add or replace entries (e.g. `approach_mean`). |
| `classifiers` | `["logreg", "svm"]` | Classifiers ticked by default in the GUI. |

Defaults:

- The metadata CSVs (`joy.csv`, `sadness.csv`, `records.csv`) have ready-made `valence` and `arousal` columns, appended as the last two columns. The targets read these columns. To compare with another dataset, give its CSV the same two columns on the same definition. A dataset without them can use `"columns": {"positive_mean": 1, "negative_mean": -1}` instead.
- **Valence** (column `valence`) = `positive_mean − negative_mean`, so a positive value means the image was rated more positive than negative. The label is positive when valence > 0. In this dataset that matches the folders exactly: all joy images are positive and all sadness images negative.
- **Arousal** (column `arousal`) = `aroused_mean`, labelled high when above the dataset median (2.35). In this dataset `is_arousing` is 1 for every image, so it cannot be used. Arousal and valence labels are correlated here (joy images are rated more aroused on average), so their accuracies are not independent.
- The continuous valence/arousal values are saved with the trial log and epochs, so you can relabel or run regression offline.
- **`device_channels`**: `None` assumes the eight electrodes are on Unicorn channels 1–8 in the listed order. The Unicorn's stock order is Fz, C3, Cz, C4, Pz, PO7, Oz, PO8. If the wiring differs, list the channel index (0–7) for each electrode.
- **Bandpass** is 1–48 Hz so that the gamma band (30–45 Hz) is kept. The 50 Hz notch stays on. If a configured band falls outside the bandpass, the app shows a warning.

**Adding a classifier:** in `hotbrain/classifiers.py` call `register_classifier("key", "Label", factory)`, where `factory` returns any unfitted scikit-learn estimator. It appears as a checkbox in the GUI. Every estimator is wrapped in a `StandardScaler` pipeline. Built-in keys: `logreg` (L2 logistic regression), `svm` (RBF), `svm_linear`.

## Operator flow

1. Set a participant ID and Unicorn serial (pre-filled with `UN-2019.03.01`; leave blank to use the first available device).
2. Choose a finite trial count (a multiple of the number of dataset folders; default 160) or continuous mode.
   - Finite runs shuffle an exactly balanced list of folders.
   - Continuous mode draws a folder uniformly at random.
   - Each folder cycles through its images without replacement before reusing them.
3. Tick the classifiers to compare.
4. Set image duration (default 1.0 s) and fixation minimum/maximum (default 0.6–1.0 s, drawn uniformly). Choose OSCAR and optional pretrial settling.
5. Wait until the device status says connected, then start. Trials shown during connection setup have no complete EEG epoch.
6. The image tab opens in an independent window; maximize it on the participant display. The other two tabs can also detach.
7. Stop with the button or the selected stop key (Escape by default), or let a finite run end.
8. Save the fitted models (needs at least ten usable epochs with both labels for a model) or discard the run. With no EEG, the trial sequence still runs and no model or accuracy is fabricated.

The training tab shows a table and a plot with the rolling accuracy of the last 20 **test-before-train** predictions for every target/classifier pair. It also shows standardized weights for linear models.

## EEG epochs and features

The epoch starts **−200 ms** before image onset and ends **300 ms before image offset**: +700 ms for a 1 s image. Preprocessing is notch + bandpass (from config) and −200 to 0 ms baseline subtraction. `eeg.features` picks the feature families, computed on the post-onset part of the epoch. The defaults, the methods listed in `basic_info.md`, give 32 features:

- `band_power`: log power per electrode (8) × band (alpha, beta, gamma), giving 24 features;
- `fractal_dimension`: Higuchi fractal dimension per electrode (`fractal_kmax` 10), giving 8 features.

Optional families:

- `region_band_power`: log power averaged over each region (frontal Fz, central C3/Cz/C4, parietal P3/Pz/P4, occipital Oz) × band;
- `asymmetry`: right-minus-left log power for C4–C3 and P4–P3 × band;
- `erp`: mean amplitude per region in 100–200 ms, 200–350 ms and 350 ms to epoch end.

**Higuchi fractal dimension** measures how irregular or complex a signal is. The signal is resampled with steps k = 1…kmax, and the curve length L(k) is measured at each step. For a smooth signal L(k) shrinks roughly like 1/k, giving FD ≈ 1. For white noise it shrinks like 1/k², giving FD ≈ 2. FD is the slope of log L(k) against log(1/k). Unlike band power, it captures the shape of the signal across all frequencies at once, so it can register changes in complexity that a single band misses.

Longer images change the feature definitions, so keep the duration consistent when training and applying a saved model. The minimum image duration is 0.7 s and grows if the ERP windows need more. These are candidate measures, not established emotion signatures for this headset. [Emotional picture EEG feature study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5476954/) · [Affective picture ERP study](https://pubmed.ncbi.nlm.nih.gov/18068150/) · [Temporal decoding of affective scenes](https://pubmed.ncbi.nlm.nih.gov/35931307/)

The EEG features window plots live features from a rolling window. The **Show** selector has four views: *one band · all electrodes* (e.g. alpha on Fz…Oz), *one electrode · all bands*, *fractal dimension · all electrodes*, or a *single feature* (any feature, including optional families). Each electrode or band gets its own small plot, with its own y scale, a linked time axis and the current value in its title. It also shows one live signal-check light per configured electrode over a top-view head diagram. A light is green when the filtered signal has an RMS of 1–75 µV and a peak-to-peak range of 4–300 µV. These are amplitude checks on the stream (after OSCAR when enabled), **not** impedance.

## Saved files

`saved_models/` contains:

- `<stem>.json`: manifest with participant, full config, channel and feature order, dataset images and resolved thresholds, timing, trial log with labels and every model's held-out predictions, and per-model accuracy.
- `<stem>.<target>.<classifier>.joblib`: one fitted scikit-learn pipeline per model.
- `<stem>.epochs.npz`: accepted epochs (µV) with per-target labels (`label_<target>`), continuous values (`value_<target>`) and groups.

`hotbrain.training.predict_saved(path, features)` returns P(label 1), or the 0/1 prediction for SVMs, which have no probabilities. joblib uses pickle, so only load files you created.

## OSCAR and timing limits

OSCAR is activated through `HybridBlack(enable_oscar=True)`. The optional pretrial period lets its online processing settle while fixation is displayed. g.Pype does not expose an OSCAR calibration export/import API, so a saved calibration cannot be reused after a restart. [g.Pype OSCAR FAQ](https://gpype.gtec.at/content/5_faq/index.html)

Onset is timestamped after Qt paints the image. EEG sample times are anchored to the first frame's host arrival minus the Unicorn's documented device delay and g.Pype's reported OSCAR delay, then advanced at the configured sampling rate. This is a software timing approximation: validate it against a photodiode or hardware trigger before using accuracy as research evidence.

`first_try.py` is a separate g.Pype example, not part of the trainer. `archive/training_dataset/` holds the old two emoji images. On macOS, `macos/Hot Brain Trainer.app` is a launcher bundle; copy it to the Desktop after creating `.venv` in this repository.

## Changelog: config-driven datasets, targets and classifiers

What changed compared with the original happy/sad trainer:

- **Dataset**: the app reads `data/training/joy` and `data/training/sadness` (143 + 74 images) instead of `training_dataset/happy_images|sad_images`. The two old emoji images moved to `archive/`. Labels come from the metadata CSV next to each folder, and switching datasets only needs the `datasets` list in `CONFIG`.
- **Single config** (`hotbrain/config.py`): montage, sampling rate, filters, bands, feature families, targets and classifiers live in one validated dictionary, matching `basic_info.md`.
- **EEG processing**:
  - electrodes are Fz, C3, Cz, C4, P3, Pz, P4, Oz (previously Fz, C3, Cz, C4, Pz, PO7, Oz, PO8);
  - 1–48 Hz bandpass plus a 50 Hz notch (previously 1–35 Hz, no notch);
  - bands are alpha, beta, gamma (previously theta, alpha, beta);
  - features are per-electrode band power + Higuchi fractal dimension, 32 by default. Region-averaged band power, asymmetry and ERP features are optional;
  - the live feature plot can show one band on all electrodes, all bands of one electrode, or all fractal dimensions.
- **Targets**: separate valence (`positive_mean − negative_mean > 0`) and arousal (`aroused_mean` > median) classification. Each target is defined by one column or a weighted sum of columns, so other targets can be added.
- **Data**: `valence` and `arousal` columns added to `data/training/*.csv` for easy comparison with other datasets. All original columns are unchanged.
- **Simulated EEG**: a GUI toggle and a `run.py --simulate` flag stream synthetic EEG with an adjustable label-dependent effect, for testing without a headset.
- **Per-electrode features**: band power is per electrode (32 features by default). The live feature plot shows one band on all electrodes, all bands of one electrode, or all fractal dimensions.
- **Sessions**:
  - reproducible 80/20 split;
  - a validation session with frozen models, balanced accuracy and majority baseline;
  - a test session with continuous, smoothed valence/arousal decoding;
  - a live valence–arousal plane next to the image;
  - loading saved models;
  - per-session logs.
- **Docs**: Polish summaries of the papers in `docs/papers/`, and a document on possible improvements with a critical review in `docs/optymalizacje_i_ocena_krytyczna.md`.
- **Classifiers**: a registry (`hotbrain/classifiers.py`) with logistic regression, RBF SVM and linear SVM. Every target × selected classifier is trained and evaluated test-before-train in parallel, and the GUI shows each one's rolling accuracy and linear weights.
- **Trial plan**: trials are balanced across dataset folders, and the trial count must be a multiple of the number of folders.
- **Saving**: one `.joblib` per model, a manifest with the full config, resolved thresholds and per-trial predictions, and epochs with labels and continuous target values.
- **GUI**: the electrode map follows the configured montage and has 10-20 positions for common electrodes.
- **Tests**: updated for the new montage and features. New tests cover the real dataset, config validation, target formulas, fractal dimension and multi-model training.

## Checks

```bash
.venv/bin/python -m pytest -q
```
