# Hot Brain scene training prototype

Participant-specific EEG experiments with the g.tec Unicorn headset.

## Automatic face EEG pilot (default launch)

Run `.venv/bin/python run.py` or use the desktop launcher. The app opens three
windows: progress, participant faces, and live EEG/feature/quality diagnostics.
It connects to Unicorn `UN-2019.03.01` with OSCAR off by default (wet-electrode
pilot). Once real EEG streams and at least two of PO7/Oz/PO8 pass the provisional
signal checks, the pilot starts after a five-second countdown. No training can
start on simulated data. Press **Escape** anywhere to stop; the run also stops
after 15 minutes at most. The three windows can be moved or maximised.

The pilot uses *generated schematic faces* with fixed identity, background,
lighting and position within each 20-second block. The neutral face appears
at 6 Hz, with a happy or sad morph every fifth face (1.2 Hz). One neutral-only
control block, four baseline blocks, ten adaptive training blocks and four
final blocks are planned. Each expression block is followed by a slower
five-face question: press **0** if none changed or **1–5** for the changed
position, then **H/S** for happy/sad and **1–5** for perceived strength. The
question times out if no key is pressed. The EEG metric is the predeclared
1.2–4.8 Hz harmonic response on PO7, Oz and PO8, divided by nearby spectral
noise. A training level becomes subtler only when that metric clears the
provisional threshold and the question is answered correctly. The threshold
and these faces have not been clinically validated.

Raw real EEG, scan-validity flags, block markers, face timing, questions and
answers are saved automatically in `recordings/face_pilot_*`. `output/` holds
reproducibly rendered face PNGs. These folders are excluded from Git. The
progress chart shows the EEG response per block. Live diagnostic plots are
separate from the face response analysis.

**Interpretation:** a tagged EEG response shows sensitivity to a repeated
visual change; it does not prove that the participant understood an emotion.
The schematic faces are an acquisition/interaction pilot, not validated social
stimuli. Qt paint timing is logged and blocks with missing or irregular paints
are rejected. Verify actual light-onset timing with a photodiode, especially
through the VR display, before treating the EEG measure as research-grade.
For the previous scene-training app, launch `run.py --scenes`.

## Project goals

1. **Train and evaluate a valence–arousal classifier.** Use test-before-train evaluation when the balanced run has no repeated images, or interleave held-out validation images in the same run when images repeat. Select the checkpoint with the lowest held-out Brier loss.
2. **Run a test session.** The participant sees images from a test dataset (images not chosen yet), each with a known valence and arousal. Their task is to bring their own valence and arousal to the image's values. The valence and arousal decoded from their EEG are shown live during the session.

The app has three session types for this: **Training**, **Validation** and **Test**. By default it trains a **valence** model with L2 logistic regression; the GUI can also select arousal and other classifiers. Labels come from each image's metadata CSV. Each model predicts a trial from its EEG **before** that trial is added to its fit. No image pixels or labels enter any model as predictors.

The papers behind the design are summarised in Polish in [`docs/papers/`](docs/papers/), one `<paper>_podsumowanie.md` per paper. The list of possible improvements and a critical review of the current state are in [`docs/optymalizacje_i_ocena_krytyczna.md`](docs/optymalizacje_i_ocena_krytyczna.md).

## Sessions

Pick the session type in the **Session** selector of the training window. The image presentation and EEG features open as separate windows.

| Session | Images | Models | What you see |
|---|---|---|---|
| **Training** | unique images in prequential mode; in split mode, training and held-out images interleaved in one run, balanced across valence classes | fit only on training images; keep checkpoints every 20 usable training epochs; select each model's best checkpoint on the complete held-out EEG set | rolling training accuracy and training/validation Brier loss per model, linear weights |
| **Validation** | optional legacy session for models that have **not** used these images for checkpoint selection | frozen models | valence–arousal plane and held-out accuracy; unavailable after an interleaved split run because those images already selected its checkpoint |
| **Test** | `data/test` (each image once) | frozen | the image with its target ◯; the estimate ● updates continuously (every `feedback.update_s`, smoothed) with a trail and distance to the target |

Details:

- **Evaluation choice.** Prequential mode predicts each trial before fitting it, uses each selected image once, and shows rolling test-before-train accuracy. With the current balanced dataset it is available for at most 366 presentations. Longer or continuous runs automatically use the interleaved split option. The 80/20 **image identity** partition is stratified by `balance_by` and reproducible from `split.seed`; it gives 146 negative + 306 positive training images and 37 negative + 77 positive validation images. In the default 800-presentation run, 640 presentations train the model and 160 show held-out images. No held-out EEG enters fitting. Training images can repeat only within the training pool; held-out images can repeat only within the validation pool. After the run, each model's checkpoints are compared on the same complete set of usable validation epochs, using mean Brier loss; the lowest-loss checkpoint is the model offered for saving. The validation-selected score is for model selection and is not an unbiased final test score.
- **Chance reference.** A constant probability of 0.5 has Brier loss 0.25 for binary labels; chance ranking gives AUC 0.5. The split-mode loss plot marks the Brier baseline, and the final status and model manifest include each selected checkpoint's Brier loss, AUC and a flag when loss is at or above 0.25. A lowest-loss checkpoint can still perform worse than 50/50; saving it does not imply that EEG contains detectable valence information. The selected validation result is optimistic because it chooses among multiple checkpoints, so use new images for a final test.
- **Plane coordinates.**
  - Target ◯: the image's rating, scaled to [−1, 1] around the class threshold. 0 is the class boundary; ±1 is the most extreme image in the training dataset.
  - EEG estimate ●: `2·P(class 1) − 1` from the classifier chosen in **Live display uses**. It is the classifier's confidence, not a calibrated rating.
- **Test images.** Put them in `data/test/` with `data/test.csv` containing `image_name`, `valence` and `arousal` (same definitions as the training CSVs). Images used for checkpoint selection must not be reused as an independent test set.
- **Test timing.** Set in `CONFIG["test_session"]`: `trial_s` (15 s) per image and `rest_s` (5 s) between images.
- **Image duration.** The models must have been trained with the same image duration, because it defines the features. The app checks this before starting.
- **Saving.**
  - Training saves the selected best validation checkpoints in split mode, or final models in prequential mode.
  - Validation and test save `validation_*.json` / `test_*.json` in `saved_models/`. The file holds per-trial predictions and scores and, for test, the whole live decoding trace with the mean and final distance to the target for each trial.
  - Validation also saves its epochs.

## Run

On macOS Apple silicon, Windows x64, or Linux x64, use Python 3.10–3.14. Run from the `Hot-Brains` folder.

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-device.txt
.venv/bin/python run.py              # Unicorn headset
```

Windows (PowerShell or cmd):

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements-device.txt
.venv\Scripts\python run.py              # Unicorn headset
```

For inspecting the GUI without a headset, `requirements.txt` is enough. The app keeps retrying the Unicorn connection while the Qt loop remains responsive, but **training cannot start until it receives current EEG samples**. It never generates substitute EEG. `requirements-device.txt` adds the g.Pype device drivers. On macOS, grant Bluetooth permission to the application running Python. The official g.Pype package supplies `HybridBlack` and the optional `enable_oscar=True` path. [g.Pype repository](https://github.com/gtec-neurotechnology/gpype) · [device reference](https://gpype.gtec.at/content/7_sdk_reference/backend_sources/hybrid_black.html) · [platform notes](https://gpype.gtec.at/content/1_basic_concepts/supported_devices.html)

## Configuration (`hotbrain/config.py`)

Everything that differs between experiments lives in one dictionary, `CONFIG`. `run.py` passes it to `launch(root, config)`, so you can also pass a modified copy. It is validated at startup.

| Key | Default | Meaning |
|---|---|---|
| `datasets` | `[{"images": "training/images", "metadata": "training/records.csv"}]` | Image folders under `data_dir` (`data/`). An entry is either a folder with a same-named sibling CSV (`"sub/folder"` → `sub/folder.csv`) or a dict naming the folder and its CSV. The CSV needs one row per image; rows without an image file are skipped and reported. |
| `balance_by` | `"valence"` | Training trials are balanced across this target's classes and the optional 80/20 split is stratified by them. `None` balances across the dataset folders instead. |
| `image_column` | `image_name` | CSV column holding the file name. |
| `eeg` | confirmed physical cap | `sampling_rate` 250, `electrodes` Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, `bandpass_hz` 1–48, `notch_hz` 50, `bands_hz` alpha/beta/gamma, `features` band power + fractal dimension. Also `regions`, `asymmetry_pairs`, `erp_windows_ms`, `fractal_kmax` and `device_channels`. |
| `targets` | valence, arousal | Each target gets one value per image: a single CSV column (`"column": "aroused_mean"`) or a weighted sum of columns (`"columns": {"positive_mean": 1, "negative_mean": -1}`). The binary label comes from that value. With `threshold: None` the value must already be 0/1. A number or `"median"` makes label 1 mean the value is above the threshold. To analyse something else, add or replace entries (e.g. `approach_mean`). |
| `classifiers` | `["logreg"]` | Models initially checked for the valence target. Select other target/model pairs in the GUI. |

Defaults:

- The metadata CSV (`data/training/records.csv`) has ready-made `valence` and `arousal` columns, appended as the last two columns. The targets read these columns. To compare with another dataset, give its CSV the same two columns on the same definition. A dataset without them can use `"columns": {"positive_mean": 1, "negative_mean": -1}` instead.
- **Valence** (column `valence`) = `positive_mean − negative_mean`, so a positive value means the image was rated more positive than negative. The label is positive when valence > 0. The current dataset has 566 images: 383 positive and 183 negative. It also contains neutral images, whose valence lies near 0, so their labels are noisier.
- **Arousal** (column `arousal`) = `aroused_mean`, labelled high when above the dataset median (2.34; 283 low / 283 high). Use the continuous rating rather than the binary `is_arousing` flag.
- The continuous valence/arousal values are saved with the trial log and epochs, so you can relabel or run regression offline.
- **`device_channels`**: `None` maps Unicorn channels 1–8 to the eight electrodes in the listed order. The operator confirmed this cap follows Fz, C3, Cz, C4, Pz, PO7, Oz, PO8. If the wiring changes, list the zero-based channel index (0–7) for each electrode.
- **Bandpass** is 1–48 Hz so that the gamma band (30–45 Hz) is kept. The 50 Hz notch stays on. If a configured band falls outside the bandpass, the app shows a warning.

**Adding a classifier:** in `hotbrain/classifiers.py` call `register_classifier("key", "Label", factory)`, where `factory` returns any unfitted scikit-learn estimator. It appears in the GUI's training checkboxes. Every estimator is wrapped in a `StandardScaler` pipeline. Estimators without `predict_proba` are wrapped in 3-fold sigmoid calibration, including scaling within each calibration fold. Built-in keys: `logreg` (L2 logistic regression), `svm` (RBF), `svm_linear`, `forest` (48 trees, maximum depth 4, minimum 4 samples per leaf). Scaling does not change a tree's split order. The forest is a nonlinear comparator and is not guaranteed to outperform the linear models. It exposes native, uncalibrated class probabilities.

## Operator flow

1. Set a participant ID and Unicorn serial (pre-filled with `UN-2019.03.01`; leave blank to use the first available device).
2. Choose a finite trial count (a multiple of the number of balance groups; default 800) or continuous mode. The default 1.5 s image and random 1.0–1.5 s fixation make 800 presentations take about 33–40 minutes (36.7 minutes on average), plus the optional 12 s pretrial wait and any processing after the last image. The GUI shows the timing estimate before starting.
   - Finite runs shuffle an exactly balanced list of groups.
   - Continuous mode draws a group uniformly at random.
   - Each group cycles through its images without replacement before reusing them.
3. Choose **Prequential** for a finite balanced run without image repeats, or **Interleave held-out validation** when images repeat. The GUI disables prequential mode when the chosen count would repeat an image or the run is continuous; it disables split mode when there are too few images to leave training and validation examples in every group. The default 800-trial run uses interleaved validation.
4. Select any of the eight numbered electrodes for the model. The model extracts features only from those channels; the live EEG traces and signal-quality lights continue to show all eight. Changing this selection invalidates loaded models, since their feature vectors no longer match.
5. Check the **target/model** pairs to train. Valence with L2 logistic regression is the sole default. Valence and arousal can each use up to three of the four available classifiers, for at most six live curves. Selections are independent: choosing arousal/forest does not also train valence/forest.
6. Set image duration (default 1.5 s) and fixation minimum/maximum (default 1.0–1.5 s, drawn uniformly). Choose OSCAR and an optional pretrial waiting period. The wait is not an OSCAR calibration.
7. Wait until the device status says it is **streaming EEG** and the Start button becomes enabled. Starting needs current real scans, but does not require a flawless 0.5 s block. During a run, an isolated invalid scan rejects only the affected EEG epoch; a stale stream, pipeline error, or a full second without a real scan stops presentations.
8. At launch, training progress, image presentation, and EEG features open by default as three separate, draggable windows arranged side by side. Move or maximize any window independently, including the image window on the participant display.
9. Stop with the button or the selected stop key (Escape by default), or let a finite run end.
10. After stopping, wait for checkpoint scoring to finish. In split mode, the **Save** checkbox appears only for models with a selected checkpoint; this needs at least ten usable validation epochs and both labels. Choose which models to keep, then click **Save fitted models**, or discard them. The EEG and trial log are archived automatically either way. The saved model is the selected checkpoint, not the final fit.

During a run, the training settings collapse to leave room for the live plots. The first plot shows rolling accuracy of the last 20 **test-before-train** predictions for every target/classifier pair. In split mode, a second plot shows rolling Brier loss for training-stream predictions (before each fit) and held-out validation predictions; two models produce four loss curves. These live curves are diagnostics. Checkpoint selection after the final presentation rescores every stored checkpoint on the same complete held-out EEG set. The table also shows the latest P(class 1) (positive valence or high arousal), save checkboxes after stopping, and standardized weights for linear models. The EEG feature window updates only during an active run. It freezes when training stops and clears its values if the stream is lost.

g.Pype acquires EEG in its pipeline thread. Completed trial epochs enter one ordered training worker, which extracts features and refits the selected models without blocking Qt's stimulus timers. Only the GUI thread changes widgets; it picks up finished predictions on its 100 ms poll. Queued trials are trained in presentation order, and Save stays unavailable until all pending trials finish. Each trial log records `training_queue_ms` and `training_ms` so delays can be checked after a run. The live feature plot and signal lights still do small calculations on the GUI thread. Python threads improve responsiveness here, but they are not a hard real-time guarantee: verify actual image timing with a photodiode before research use.

## EEG epochs and features

The epoch starts **−200 ms** before image onset and ends **300 ms before image offset**: +1200 ms for the default 1.5 s image (+700 ms for a 1 s image). Preprocessing is notch + bandpass (from config) and −200 to 0 ms baseline subtraction. `eeg.features` picks the feature families, computed on the post-onset part of the epoch. With all eight electrodes selected, the defaults listed in `basic_info.md` give 32 features; selecting fewer electrodes reduces the model's feature vector:

- `band_power`: log power per electrode (8) × band (alpha, beta, gamma), giving 24 features;
- `fractal_dimension`: Higuchi fractal dimension per electrode (`fractal_kmax` 10), giving 8 features.

Optional families:

- `region_band_power`: log power averaged over each region (frontal Fz, central C3/Cz/C4, parietal Pz, posterior PO7/Oz/PO8) × band;
- `asymmetry`: right-minus-left log power for C4–C3 and PO8–PO7 × band;
- `erp`: mean amplitude per region in 100–200 ms, 200–350 ms and 350 ms to epoch end.

**Higuchi fractal dimension** measures how irregular or complex a signal is. The signal is resampled with steps k = 1…kmax, and the curve length L(k) is measured at each step. For a smooth signal L(k) shrinks roughly like 1/k, giving FD ≈ 1. For white noise it shrinks like 1/k², giving FD ≈ 2. FD is the slope of log L(k) against log(1/k). Unlike band power, it captures the shape of the signal across all frequencies at once, so it can register changes in complexity that a single band misses.

Longer images change the feature definitions, so keep the duration consistent when training and applying a saved model. The minimum image duration is 0.7 s and grows if the ERP windows need more. These are candidate measures, not established emotion signatures for this headset. [Emotional picture EEG feature study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5476954/) · [Affective picture ERP study](https://pubmed.ncbi.nlm.nih.gov/18068150/) · [Temporal decoding of affective scenes](https://pubmed.ncbi.nlm.nih.gov/35931307/)

The EEG diagnostics window has independent **Live EEG** and **Live features** switches. Both are on initially and can be paused or resumed before, during, or after a session. **Recheck EEG connection** starts a connection attempt immediately if the stream is unavailable; it is disabled during a session. A source label distinguishes a real live stream, paused plots of real data, an ongoing connection check, and no stream. The EEG plot shows four seconds of the real eight-channel stream in stacked traces. Each channel is median-centered for display, with 200 µV between trace centers; invalid scans appear as gaps. Diagnostic feature values use clean, rolling 1 s EEG windows; the model still uses the image-onset window selected in the training controls. ERP values in the live plot are not image-onset responses. OSCAR can be toggled from either window outside a session, which reconnects the device. No trace is generated or replayed when the headset is disconnected, and training requires a current real stream.

The **Features** tab plots live features from a rolling window. The **Show** selector offers one band across all electrodes, one electrode across all bands, fractal dimension across all electrodes, or a single feature. The brain map and eight Good/Bad/Waiting ratings use the last 2 s of actual EEG and continue during training. Each channel is rated from an uninterrupted real segment of at least 0.5 s; an occasional invalid Bluetooth scan no longer makes all eight electrodes look bad. A channel is Good when at least 80% of those 2 s are valid, its 1–48 Hz RMS is 1–75 µV, and its peak-to-peak range is 4–300 µV. Ratings clear if the stream is lost. These provisional amplitude checks are **not** impedance or proof that the cap is being worn.

## Saved files

Every session automatically creates files in ignored `recordings/`, including when you discard the models or close the app mid-run:

- `<stem>.eeg.f64`: append-only, little-endian float64 rows from the g.Pype Unicorn source. Columns are ring sample index, stream generation, all eight EEG channels (µV), battery, counter, scan-validity flag, and image marker. This includes invalid scans and fixation periods. The file is roughly 101 MB per hour at 250 Hz. A partial file remains readable after a process crash.
- `<stem>.meta.json`: exact column order, sample rate, OSCAR setting, row count, dropped-block count, and recording completion status. `hotbrain.recording.load_recording(meta_path)` loads the rows and metadata.
- `<stem>.events.jsonl`: each image onset with its image path, labels, marker code, and onset time. Events are written as trials happen, so labels survive an interrupted session.
- `<stem>.session.json`: settings and final trial log, including EEG acceptance and model predictions. This is updated at the end; for an interrupted run the event log still retains onsets.
- `<stem>.epochs.npz`: usable, unfiltered eight-channel epochs plus labels and `trial_role` (`training` or `validation`), for quick offline model experiments. This is saved at the end when there are usable epochs. Recompute different epoch windows from the continuous stream and event markers.

These are the acquisition samples **before the app's notch, bandpass, and baseline processing**. When OSCAR is on, g.Pype's source has already applied OSCAR, so the recording cannot reconstruct the pre-OSCAR waveform. For an unprocessed source comparison, run a separate session with OSCAR off. The archive is not automatically uploaded to GitHub.

For offline modelling, load `<stem>.epochs.npz` with `np.load(...)`; `eeg_uV` has shape `(trials, samples, 8)`, and `trial_number`, `label_valence`, `label_arousal`, `value_valence` and `value_arousal` align by row. For different time windows, call `hotbrain.recording.load_recording(...)` on the manifest and use the event/marker columns to cut new epochs. Check `scan_valid == 1` before treating a sample as observed EEG.

`saved_models/` contains:

- `<stem>.json`: manifest with participant, full config, channel and feature order, dataset images and resolved thresholds, timing, per-trial role and predictions, each selected checkpoint's training-trial count and held-out Brier loss, and the list of models selected for saving.
- `<stem>.<target>.<classifier>.joblib`: one fitted scikit-learn pipeline per selected model.
- The model manifest points to the matching `recordings/<stem>.epochs.npz`, with accepted epochs from all eight acquired channels (µV), including channels omitted from the model, per-target labels (`label_<target>`), continuous values (`value_<target>`) and groups. The manifest records the model electrode selection and feature order.

`hotbrain.training.predict_saved(path, features)` returns P(label 1) for logistic regression, calibrated SVMs and random forest. Existing SVM files saved by an earlier version have no calibrated probability output and must be retrained for this helper. SVM probabilities use cross-validated sigmoid calibration; forest probabilities are averages of tree leaf class frequencies, not calibrated. With very few EEG trials, do not interpret any of them as validated emotional intensity. joblib uses pickle, so only load files you created.

## OSCAR and timing limits

OSCAR is activated through `HybridBlack(enable_oscar=True)`. The optional pretrial wait displays fixation before the first image; the app does not perform or verify a separate OSCAR calibration. g.Pype does not expose an OSCAR calibration export/import API, so a saved calibration cannot be reused after a restart. [g.Pype OSCAR FAQ](https://gpype.gtec.at/content/5_faq/index.html)

### Visual EEG check before training

Training starts with an optional, default-on Gabor-versus-gray check. It shows 120 randomized trials: 80 for fitting a small L2 logistic model and 40 later held-out trials for scoring. Each stimulus lasts 0.8 s, with a random 0.5–0.8 s fixation beforehand, so presentation takes about 2.6–3.2 minutes; a 5 s break follows before the scene experiment starts. The check uses a fixed −200 to +500 ms EEG epoch and predeclared 70–150, 150–250 and 250–400 ms evoked-response means on electrodes 6–8 (PO7, Oz, PO8). The GUI shows held-out balanced accuracy, AUC, Brier loss and a conservative detection flag. It is an acquisition/timing sanity check, not evidence that valence can be decoded. Gabor trials are marked and preserved in the continuous session recording but **never fit the scene models**. If usable scans are insufficient, the result is inconclusive and the scene experiment still starts after the break.

Scene models continue to use all eight electrodes by default, including 6–8. Posterior signals can contain useful visual and affect-related information. They can also encode low-level image differences, so a successful scene classifier must later be checked on new stimuli (especially faces) and ideally against a scene-model run with electrodes 6–8 unticked. The Gabor check always uses 6–8 regardless of the scene-model checkbox selection.

Each image is marked immediately after Qt paints it. The image, fixation and optional settling periods use precise Qt timers; coarse timers could end an image early. A g.Pype `Marker` and `Router` place the image event in the same stream as the Unicorn's EEG. Epochs use the marker's integer sample index, plus **10 samples (40 ms)** for the Unicorn's documented acquisition-to-host delay, minus the measured Qt paint-to-marker interval rounded to samples. They no longer use the first Bluetooth frame's arrival time or assume that the headset clock stays exactly at 250 Hz throughout a run. OSCAR's processing delay is handled within g.Pype's event placement. The app rejects any epoch containing a scan whose Unicorn validation flag is 0 (the driver may fill lost scans), as well as non-finite samples, and logs `invalid_scans` for that trial. A brief invalid scan no longer stops the run. The run does stop when frames are stale for more than 500 ms, when no real scan appears for one second, or when g.Pype reports a condition other than Healthy or Warning. The trial log records the marker code, sample index, corrected EEG onset index, and Qt paint-to-marker delay; a delay over 20 ms stops the run. [g.Pype time and synchronisation](https://gpype.gtec.at/content/3_architecture_design/time_synchronisation.html) · [Unicorn validation channel](https://gpype.gtec.at/content/2_gpype_training/season3_gearing_up/s3e4_hybrid_black.html)

The Qt paint callback still precedes the screen's actual light output by an unmeasured display and compositor delay. The 40 ms device delay is a documented nominal value, not a calibration of this specific setup. For research-grade stimulus timing, measure the screen with a photodiode and use that measurement to verify or correct the offset before interpreting millisecond EEG windows. [g.Pype timing limits](https://gpype.gtec.at/content/3_architecture_design/time_synchronisation.html)

`first_try.py` is a separate g.Pype example, not part of the trainer. `archive/training_dataset/` holds the old two emoji images. On macOS, `macos/Hot Brain Trainer.app` is a launcher bundle; copy it to the Desktop after creating `.venv` in this repository.

## Changelog: config-driven datasets, targets and classifiers

What changed compared with the original happy/sad trainer:

- **Dataset**: the app reads one image folder, `data/training/images` (566 images), with `data/training/records.csv` as its metadata. Earlier versions used `data/training/joy` and `data/training/sadness` with 143 + 74 images. The two old emoji images moved to `archive/`. Switching datasets only needs the `datasets` list in `CONFIG`; a dataset entry may name its CSV explicitly.
- **Single config** (`hotbrain/config.py`): montage, sampling rate, filters, bands, feature families, targets and classifiers live in one validated dictionary, matching `basic_info.md`.
- **EEG processing**:
  - the cap-confirmed electrode order is Fz, C3, Cz, C4, Pz, PO7, Oz, PO8. An earlier config incorrectly substituted P3 and P4 and mislabeled channels 5–8; models saved with that config should not be treated as having the corrected electrode metadata;
  - 1–48 Hz bandpass plus a 50 Hz notch (previously 1–35 Hz, no notch);
  - bands are alpha, beta, gamma (previously theta, alpha, beta);
  - features are per-electrode band power + Higuchi fractal dimension, 32 by default. Region-averaged band power, asymmetry and ERP features are optional;
  - the live feature plot can show one band on all electrodes, all bands of one electrode, or all fractal dimensions.
- **Targets**: separate valence (`positive_mean − negative_mean > 0`) and arousal (`aroused_mean` > median) classification. Each target is defined by one column or a weighted sum of columns, so other targets can be added.
- **Data**: `valence` and `arousal` columns added to `data/training/*.csv` for easy comparison with other datasets. All original columns are unchanged.
- **Per-electrode features**: band power is per electrode (32 features by default). The live feature plot shows one band on all electrodes, all bands of one electrode, or all fractal dimensions.
- **Sessions**:
  - reproducible 80/20 split;
  - a validation session with frozen models, balanced accuracy and majority baseline;
  - a test session with continuous, smoothed valence/arousal decoding;
  - a live valence–arousal plane next to the image;
  - loading saved models;
  - per-session logs.
- **Docs**: Polish summaries of the papers in `docs/papers/`, and a document on possible improvements with a critical review in `docs/optymalizacje_i_ocena_krytyczna.md`.
- **Classifiers**: a registry (`hotbrain/classifiers.py`) with logistic regression, RBF SVM, linear SVM and shallow random forest. The GUI selects individual target/model pairs, up to three per target. Each selected pair is refitted in the training worker after every usable EEG trial and evaluated test-before-train; the GUI shows rolling accuracy and linear weights where available. SVMs use cross-validated sigmoid calibration to provide probabilities.
- **Trial plan**: trials are balanced across the `balance_by` groups (default: valence classes) or across dataset folders, and the trial count must be a multiple of the number of groups.
- **Saving**: save checkboxes select fitted target/model pairs. One `.joblib` per selected model, a manifest with the full config, resolved thresholds and per-trial predictions, and epochs with labels and continuous target values.
- **GUI**: the electrode map follows the configured montage and has 10-20 positions for common electrodes.
- **Tests**: updated for the new montage and features. New tests cover the real dataset, config validation, target formulas, fractal dimension and multi-model training.

## Checks

```bash
.venv/bin/python -m pytest -q
```
