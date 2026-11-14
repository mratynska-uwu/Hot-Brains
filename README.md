# Hot Brain scene training prototype

This is the training stage for participant-specific EEG classifiers of emotional scenes. By default it trains separate **valence** and **arousal** models, each with every selected classifier (logistic regression, SVM, ...). Each image's labels come from its metadata CSV. Each model predicts a trial from its EEG **before** that trial is added to its fit. No image pixels or labels enter any model as predictors.

## Run

On macOS Apple silicon, Windows x64, or Linux x64, use Python 3.10–3.14:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-device.txt
.venv/bin/python run.py
```

For presentation debugging without a headset, `requirements.txt` is enough. The app keeps retrying the Unicorn connection while the Qt loop remains responsive. On macOS, grant Bluetooth permission to the application running Python. The official g.Pype package supplies `HybridBlack` and the optional `enable_oscar=True` path. [g.Pype repository](https://github.com/gtec-neurotechnology/gpype) · [device reference](https://gpype.gtec.at/content/7_sdk_reference/backend_sources/hybrid_black.html) · [platform notes](https://gpype.gtec.at/content/1_basic_concepts/supported_devices.html)

## Configuration (`hotbrain/config.py`)

Everything that differs between experiments lives in one dictionary, `CONFIG`. `run.py` passes it to `launch(root, config)`, so you can also pass a modified copy. It is validated at startup.

| Key | Default | Meaning |
|---|---|---|
| `datasets` | `["training/joy", "training/sadness"]` | Subfolders of `data_dir` (`data/`). Each one needs a sibling CSV with the same name (`data/training/joy.csv`), one row per image. Trials are balanced across these folders. |
| `image_column` | `image_name` | CSV column holding the file name. |
| `eeg` | from `basic_info.md` | `sampling_rate` 250, `electrodes` Fz, C3, Cz, C4, P3, Pz, P4, Oz, `bandpass_hz` 1–48, `notch_hz` 50, `bands_hz` alpha/beta/gamma, `features` band power + fractal dimension. Also `regions`, `asymmetry_pairs`, `erp_windows_ms`, `fractal_kmax` and `device_channels`. |
| `targets` | valence, arousal | Each target gets one value per image: a single CSV column (`"column": "aroused_mean"`) or a weighted sum of columns (`"columns": {"positive_mean": 1, "negative_mean": -1}`). The binary label comes from that value. With `threshold: None` the value must already be 0/1. A number or `"median"` makes label 1 mean the value is above the threshold. To analyse something else, add or replace entries (e.g. `approach_mean`). |
| `classifiers` | `["logreg", "svm", "forest"]` | Models checked by default for live comparison. |

Defaults:

- The metadata CSVs (`joy.csv`, `sadness.csv`, `records.csv`) have ready-made `valence` and `arousal` columns, appended as the last two columns. The targets read these columns. To compare with another dataset, give its CSV the same two columns on the same definition. A dataset without them can use `"columns": {"positive_mean": 1, "negative_mean": -1}` instead.
- **Valence** (column `valence`) = `positive_mean − negative_mean`, so a positive value means the image was rated more positive than negative. The label is positive when valence > 0. In this dataset that matches the folders exactly: all joy images are positive and all sadness images negative.
- **Arousal** (column `arousal`) = `aroused_mean`, labelled high when above the dataset median (2.35). In this dataset `is_arousing` is 1 for every image, so it cannot be used. Arousal and valence labels are correlated here (joy images are rated more aroused on average), so their accuracies are not independent.
- The continuous valence/arousal values are saved with the trial log and epochs, so you can relabel or run regression offline.
- **`device_channels`**: `None` assumes the eight electrodes are on Unicorn channels 1–8 in the listed order. The Unicorn's stock order is Fz, C3, Cz, C4, Pz, PO7, Oz, PO8. If the wiring differs, list the channel index (0–7) for each electrode.
- **Bandpass** is 1–48 Hz so that the gamma band (30–45 Hz) is kept. The 50 Hz notch stays on. If a configured band falls outside the bandpass, the app shows a warning.

**Adding a classifier:** in `hotbrain/classifiers.py` call `register_classifier("key", "Label", factory)`, where `factory` returns any unfitted scikit-learn estimator. It appears in the GUI's training checkboxes. Every estimator is wrapped in a `StandardScaler` pipeline. Estimators without `predict_proba` are wrapped in 3-fold sigmoid calibration, including scaling within each calibration fold. Built-in keys: `logreg` (L2 logistic regression), `svm` (RBF), `svm_linear`, `forest` (48 trees, maximum depth 4, minimum 4 samples per leaf). Scaling does not change a tree's split order. The forest is a nonlinear comparator; with 160 trials it is not guaranteed to outperform the linear models. It exposes native, uncalibrated class probabilities.

## Operator flow

1. Set a participant ID and Unicorn serial (pre-filled with `UN-2019.03.01`; leave blank to use the first available device).
2. Choose a finite trial count (a multiple of the number of dataset folders; default 160) or continuous mode.
   - Finite runs shuffle an exactly balanced list of folders.
   - Continuous mode draws a folder uniformly at random.
   - Each folder cycles through its images without replacement before reusing them.
3. Check any combination of models under **Train models**. Logistic regression, RBF SVM and shallow random forest are checked by default; linear SVM is also available. Each selected model gets its own accuracy curve and row for each target.
4. Set image duration (default 4.5 s) and fixation minimum/maximum (default 2–3 s, drawn uniformly). Choose OSCAR and optional pretrial settling.
5. Wait until the device status says connected, then start. Trials shown during connection setup have no complete EEG epoch.
6. At launch, training progress, image presentation, and EEG features open by default as three separate, draggable windows arranged side by side. Move or maximize any window independently, including the image window on the participant display.
7. Stop with the button or the selected stop key (Escape by default), or let a finite run end.
8. After stopping, use the **Save** checkbox in each fitted target/model row to choose exactly which models to keep; all fitted models are initially checked. Then click **Save fitted models**, or discard the run. A model needs at least ten usable epochs with both labels to be fitted. With no EEG, the trial sequence still runs and no model or accuracy is fabricated.

During a run, the training settings collapse to leave room for the live plot. The training window shows a table and simultaneous curves with the rolling accuracy of the last 20 **test-before-train** predictions for every target/classifier pair. The table also shows the latest P(class 1) (positive valence or high arousal), save checkboxes after stopping, and standardized weights for linear models.

## EEG epochs and features

The epoch starts **−200 ms** before image onset and ends **300 ms before image offset**: +4200 ms for the default 4.5 s image (+700 ms for a 1 s image). Preprocessing is notch + bandpass (from config) and −200 to 0 ms baseline subtraction. `eeg.features` picks the feature families, computed on the post-onset part of the epoch. The defaults, the methods listed in `basic_info.md`, give 20 features:

- `band_power`: log power per region (frontal Fz, central C3/Cz/C4, parietal P3/Pz/P4, occipital Oz) × band (alpha, beta, gamma);
- `fractal_dimension`: Higuchi fractal dimension per electrode (`fractal_kmax` 10).

Optional families:

- `asymmetry`: right-minus-left log power for C4–C3 and P4–P3 × band;
- `erp`: mean amplitude per region in 100–200 ms, 200–350 ms and 350 ms to epoch end.

Longer images change the feature definitions, so keep the duration consistent when training and applying a saved model. The minimum image duration is 0.7 s and grows if the ERP windows need more. These are candidate measures, not established emotion signatures for this headset. [Emotional picture EEG feature study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5476954/) · [Affective picture ERP study](https://pubmed.ncbi.nlm.nih.gov/18068150/) · [Temporal decoding of affective scenes](https://pubmed.ncbi.nlm.nih.gov/35931307/)

The EEG features window shows one live signal-check light per configured electrode over a top-view head diagram. A light is green when the filtered signal has an RMS of 1–75 µV and a peak-to-peak range of 4–300 µV. These are amplitude checks on the stream (after OSCAR when enabled), **not** impedance.

## Saved files

`saved_models/` contains:

- `<stem>.json`: manifest with participant, full config, channel and feature order, dataset images and resolved thresholds, timing, trial log with labels and every model's held-out predictions, per-model accuracy, and the list of models selected for saving.
- `<stem>.<target>.<classifier>.joblib`: one fitted scikit-learn pipeline per selected model.
- `<stem>.epochs.npz`: accepted epochs (µV) with per-target labels (`label_<target>`), continuous values (`value_<target>`) and groups.

`hotbrain.training.predict_saved(path, features)` returns P(label 1) for logistic regression, calibrated SVMs and random forest. Existing SVM files saved by an earlier version have no calibrated probability output and must be retrained for this helper. SVM probabilities use cross-validated sigmoid calibration; forest probabilities are averages of tree leaf class frequencies, not calibrated. With very few EEG trials, do not interpret any of them as validated emotional intensity. joblib uses pickle, so only load files you created.

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
  - features are band power + Higuchi fractal dimension, 20 by default. Asymmetry and ERP features are optional.
- **Targets**: separate valence (`positive_mean − negative_mean > 0`) and arousal (`aroused_mean` > median) classification. Each target is defined by one column or a weighted sum of columns, so other targets can be added.
- **Data**: `valence` and `arousal` columns added to `data/training/*.csv` for easy comparison with other datasets. All original columns are unchanged.
- **Classifiers**: a registry (`hotbrain/classifiers.py`) with logistic regression, RBF SVM, linear SVM and shallow random forest. The GUI checkboxes select any combination. Every target × selected classifier is refitted on each completed EEG trial and evaluated test-before-train; the GUI shows each one's rolling accuracy and linear weights where available. SVMs use cross-validated sigmoid calibration to provide probabilities.
- **Trial plan**: trials are balanced across dataset folders, and the trial count must be a multiple of the number of folders.
- **Saving**: save checkboxes select fitted target/model pairs. One `.joblib` per selected model, a manifest with the full config, resolved thresholds and per-trial predictions, and epochs with labels and continuous target values.
- **GUI**: the electrode map follows the configured montage and has 10-20 positions for common electrodes.
- **Tests**: updated for the new montage and features. New tests cover the real dataset, config validation, target formulas, fractal dimension and multi-model training.

## Checks

```bash
.venv/bin/python -m pytest -q
```
