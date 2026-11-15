# Hot Brain scene training prototype

Participant-specific EEG decoding of emotional valence and arousal with the g.tec Unicorn headset.

## Project goals

1. **Train and evaluate a valence–arousal classifier.** Use test-before-train evaluation when the chosen balanced run has no repeated images, or reserve 20% of images for a separate validation session.
2. **Run a test session.** The participant sees images from a test dataset (images not chosen yet), each with a known valence and arousal. Their task is to bring their own valence and arousal to the image's values. The valence and arousal decoded from their EEG are shown live during the session.

The app has three session types for this: **Training**, **Validation** and **Test**. By default it trains a **valence** model with an RBF-kernel SVM; the GUI can also select arousal and other classifiers. Labels come from each image's metadata CSV. Each model predicts a trial from its EEG **before** that trial is added to its fit. No image pixels or labels enter any model as predictors.

The papers behind the design are summarised in Polish in [`docs/papers/`](docs/papers/), one `<paper>_podsumowanie.md` per paper. The list of possible improvements and a critical review of the current state are in [`docs/optymalizacje_i_ocena_krytyczna.md`](docs/optymalizacje_i_ocena_krytyczna.md).

## Sessions

Pick the session type in the **Session** selector of the training window. The image presentation and EEG features open as separate windows.

| Session | Images | Models | What you see |
|---|---|---|---|
| **Training** | all available images in prequential mode, or the training portion of the split; only clear-cut images, trials balanced across the three stimulus conditions negative / positive / neutral (`conditions`, `balance_by`) | learn after every trial (test-before-train) | rolling accuracy per target × classifier, linear weights |
| **Validation** | the held-out 20% when split mode was used, each image once | frozen: from a compatible split-mode training run or **Load saved models…** | next to the image, a valence–arousal plane with the target ◯ and the EEG estimate ● for each image; a table of accuracy, balanced accuracy and majority-class baseline |
| **Test** | `data/test` (each image once) | frozen | the image with its target ◯; the estimate ● updates continuously (every `feedback.update_s`, smoothed) with a trail and distance to the target |
| **Calibration (goal 2)** | the most clear-cut training images, `calibration.blocks` blocks balanced across the stimulus conditions | learn block-wise from sliding windows of a held state | each image for `block_s` (20 s): the participant holds its emotion; the plane shows the target and the block's mean estimate |
| **Resting baseline** | none (fixation cross) | none | `resting.duration_s` (90 s) of eyes-open rest, saved as the participant's individual reference |

Details:

- **Evaluation choice.** Prequential mode predicts each trial before fitting it, uses each selected image once, and shows the rolling test-before-train accuracy. With balanced classes it is available only when the presentation count is no larger than the smallest class count times the number of classes. With the current dataset this is 177 presentations (3 conditions × 59 neutral images). Longer or continuous runs automatically use the split option. In split mode the 80/20 partition is stratified by `balance_by` and reproducible from `split.seed`; it gives 73 + 197 + 47 training and 18 + 49 + 12 validation images (negative + positive + neutral). Repeated training images never enter the validation subset. Run the separate Validation session for the held-out score; the training graph remains an exploratory stream diagnostic.
- **Plane coordinates.**
  - Target ◯: the image's rating, scaled to [−1, 1] around the class threshold. 0 is the class boundary; ±1 is the most extreme image in the training dataset.
  - EEG estimate ●: `2·P(class 1) − 1` from the classifier chosen in **Live display uses**. It is the classifier's confidence, not a calibrated rating.
- **Test images.** Put them in `data/test/` with `data/test.csv` containing `image_name`, `valence` and `arousal` (same definitions as the training CSVs). With compatible split-mode models, the app can offer the held-out validation images as a fallback.
- **Test timing.** Set in `CONFIG["test_session"]`: `trial_s` (15 s) per image and `rest_s` (5 s) between images.
- **Calibration → test.** After a Calibration session the frozen models are "state" models trained on 2 s windows of a held emotion (skipping the first 2 s of each block; windows of one block are predicted before any of them is learned). A Test session with these models decodes the same 2 s windows continuously, smoothed (`feedback.smoothing` 0.1). Models trained on image epochs decode a window laid out like their training epochs.
- **Feedback.** The ● turns green on target (distance < `feedback.on_target_distance`, 0.35), yellow when close and red when far; reaching the target beeps once. Each test trial logs the mean and final distance and the share of time on target. **Practice test run** shows `feedback.practice_images` (3) images and saves with a `practice_` prefix.
- **Validation p-value.** The validation table adds a label-permutation p-value of the balanced accuracy (predictions held fixed).
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
| `balance_by` | `"conditions"` | Training trials are balanced across the stimulus conditions and the optional 80/20 split is stratified by them. A target name balances across that target's classes instead, `None` across the dataset folders. |
| `conditions` | negative · positive · neutral | Stimulus conditions as rules on metadata CSV columns (a list of allowed values, `{"not": [...]}` or `{"min": x, "max": y}`). **Only images matching a condition are shown**, so every trial is a clear-cut example for at least one target. Defaults: *negative* / *positive* = an emotional category (`target_emotion` not `Neutral`) with valence ≤ −2 / ≥ +2 and arousal ≥ 2.2 (the lowest third of emotional images is left out); *neutral* = category `Neutral`. With the current dataset: 91 / 246 / 59 images; 170 images with weak valence or low arousal are not shown. |
| `image_column` | `image_name` | CSV column holding the file name. |
| `eeg` | confirmed physical cap | `sampling_rate` 250, `electrodes` Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, `bandpass_hz` 1–48, `notch_hz` 50, `bands_hz` theta/alpha/beta/gamma, `psd_method` `"welch"`, `reference` `None` (or `"car"`), `artifact_max_abs_uv` 150, `features` (live plot) band power + fractal dimension. Also `regions`, `asymmetry_pairs`, `erp_windows_ms`, `fractal_kmax` and `device_channels`. |
| `profiles` / `default_profile` | configurations 1, 2 and 3 / `1` | GUI **presets**. Each profile sets the baseline and, separately for valence and arousal, the electrodes, bands, feature families and analysis window (`window_ms` per profile, optionally overridden per target). The GUI loads a preset into the per-target tabs, where every setting can still be edited. See *Feature configurations* below. |
| `presentation` | image 3 s · fixation 1.5–2 s · neutral 20 s | Defaults for image duration, what is shown between images (`"fixation"` or `"neutral"`), the random fixation range and the neutral-image duration. 3 s images give about 12 trials per minute (20 s images: about 2.5); the evoked responses used by configuration 3 are over within about 1 s. |
| `neutral_images` | `"training/images/neutrals"` | Folder of neutral control images shown between trials in neutral mode. No CSV is needed; images are shuffled and repeat when the run is longer than the folder. The loader ignores this subfolder, but its images are copies of `Neutral`-category images in the main folder, which are now trials of the *neutral* condition. In neutral mode they would also be seen outside trials, so keep `"fixation"` or point this at a folder without trial images. |
| `targets` | valence, arousal | Each target gets one value per image: a single CSV column (`"column": "aroused_mean"`) or a weighted sum of columns (`"columns": {"positive_mean": 1, "negative_mean": -1}`). The binary label comes from that value, or from the image's stimulus condition when the target has `"conditions": {condition: label}` (unlisted conditions get no label; a target without a column takes the label as its value). With `threshold: None` the value must already be 0/1. A number or `"median"` makes label 1 mean the value is above the threshold. To analyse something else, add or replace entries (e.g. `approach_mean`). |
| `classifiers` | `["svm"]` | Models initially checked for the valence target. Select other target/model pairs in the GUI. |
| `calibration` | 8 blocks × 20 s, 2 s windows, 1 s step, skip 2 s | Calibration session for goal 2. |
| `resting` | `duration_s` 90 | Eyes-open resting reference recording. |
| `stop_rule` | off · valence ≥ 65% over 40 predictions | Optional automatic stop when a model's rolling balanced accuracy reaches the threshold (GUI **Stopping rule**). |
| `report` | 5 folds, 100 permutations | Training-quality report settings. |

Defaults:

- The metadata CSV (`data/training/records.csv`) has ready-made `valence` and `arousal` columns, appended as the last two columns. The targets read these columns. To compare with another dataset, give its CSV the same two columns on the same definition. A dataset without them can use `"columns": {"positive_mean": 1, "negative_mean": -1}` instead.
- **Valence** (column `valence`) = `positive_mean − negative_mean`, so a positive value means the image was rated more positive than negative. Labels come from the conditions: *negative* 0, *positive* 1 (only images with \|valence\| ≥ 2 are shown). *Neutral* images have no valence label; they used to be labelled positive because their mean rating is +2.3.
- **Arousal** is **emotional vs neutral**: *neutral* 0, *negative* and *positive* 1, and the value on the plane is that label (±1). This replaces the median split of `aroused_mean` (2.35), whose classes overlapped heavily. Arousing pictures of either valence evoke a larger late positive potential than neutral ones, the most robust single-trial EEG contrast for pictures. Every trial now has an arousal label and every emotional trial a valence label, so no trial is wasted. The median-split definition (`"threshold": "median"`, `exclude_middle`) still works when `conditions` is removed.
- Test datasets must have the columns the conditions use (`target_emotion`, `valence`, `arousal`).
- The continuous valence/arousal values are saved with the trial log and epochs, so you can relabel or run regression offline.
- **`device_channels`**: `None` maps Unicorn channels 1–8 to the eight electrodes in the listed order. The operator confirmed this cap follows Fz, C3, Cz, C4, Pz, PO7, Oz, PO8. If the wiring changes, list the zero-based channel index (0–7) for each electrode.
- **Bandpass** is 1–48 Hz so that the gamma band (30–45 Hz) is kept. The 50 Hz notch stays on. If a configured band falls outside the bandpass, the app shows a warning.

**Adding a classifier:** in `hotbrain/classifiers.py` call `register_classifier("key", "Label", factory)`, where `factory` returns any unfitted scikit-learn estimator. It appears in the GUI's training checkboxes. Every estimator is wrapped in a `StandardScaler` pipeline. Estimators without `predict_proba` are wrapped in 3-fold sigmoid calibration, including scaling within each calibration fold. Built-in keys: `logreg` (L2 logistic regression), `svm` (RBF), `svm_linear`, `forest` (48 trees, maximum depth 4, minimum 4 samples per leaf), `logreg_select` (logistic regression whose number of features, by ANOVA F, and C are chosen by an inner 3-fold cross-validation on the training data only), `ridge` (ridge regression on the continuous valence/arousal value in [−1, 1]; the sign gives the class and the value drives the plane directly) and `lda_shrinkage` (linear discriminant analysis with Ledoit–Wolf covariance shrinkage and equal class priors, the small-sample model for configuration 3). Logistic regression and both SVMs use `class_weight="balanced"`. Scaling does not change a tree's split order. The forest is a nonlinear comparator; with 160 trials it is not guaranteed to outperform the linear models. It exposes native, uncalibrated class probabilities.

## Operator flow

1. Set a participant ID and Unicorn serial (pre-filled with `UN-2019.03.01`; leave blank to use the first available device).
2. Choose a finite trial count (a multiple of the number of balance groups; default 160) or continuous mode.
   - Finite runs shuffle an exactly balanced list of groups.
   - Continuous mode draws a group uniformly at random.
   - Each group cycles through its images without replacement before reusing them.
3. Choose **Prequential** for a finite balanced run without image repeats, or **Hold out** to reserve validation images. The GUI disables prequential mode when the chosen count would repeat an image or the run is continuous; it disables split mode when there are too few images to leave training and validation examples in every group.
4. Under **Features and classifiers**, pick a **Preset** (1, 2 or 3; configuration 3 is for short runs with shrinkage LDA, the last entry opens the taste experiment), then fine-tune each target in its own tab (**Valence**, **Arousal**): electrodes, frequency bands, feature families, analysis window (0 = whole image) and classifiers. The baseline, filters, reference and artifact threshold stay shared, because both targets are computed from the same epoch. **Reset to preset** discards the edits. The summary under the tabs lists each target's electrodes, bands, window and feature count, and warns when a family is skipped (e.g. β/α without both bands). The live EEG traces and signal-quality lights always show all eight electrodes. Any change unloads previously loaded models, since their feature vectors no longer match.
5. Tick the classifiers in each target's tab. Valence with the RBF-kernel SVM is the sole default. Valence and arousal can each use up to three of the available models. **Valence and arousal are always separate classifiers**: each has its own feature vector and its own fitted model, accuracy row, weights and report section. Choosing arousal/forest does not also train valence/forest.
6. Set the image duration (default 3 s) and choose what appears **between images**: a **fixation cross** (random 1.5–2 s) or a **neutral image** (control condition, default 20 s, from `data/training/images/neutrals`). Neutral images are not marked as stimulus onsets; each trial's log records what preceded it (`before_image`). The presentation window shows the timing and the estimated run length. Choose OSCAR and an optional pretrial waiting period. The wait is not an OSCAR calibration.
7. Wait until the device status says it is **streaming EEG** and the Start button becomes enabled. Starting needs 0.5 s of clean scans. During a run, an isolated invalid scan rejects only the affected EEG epoch; a stale stream, pipeline error, or a full second without a real scan stops presentations.
8. At launch, training progress, image presentation, and EEG features open by default as three separate, draggable windows arranged side by side. Move or maximize any window independently, including the image window on the participant display.
9. Stop with the button or the selected stop key (Escape by default), or let a finite run end.
10. After stopping, use the **Save** checkbox in each fitted target/model row to choose exactly which models to keep; all fitted models are initially checked. Then click **Save fitted models**, or discard the run. A model needs at least ten usable epochs with both labels to be fitted.

During a run, the training settings collapse to leave room for the live plot. The training window shows a table and simultaneous curves with the rolling accuracy of the last 20 **test-before-train** predictions for every target/classifier pair. In prequential mode these predictions are the selected evaluation; in split mode the frozen-model Validation session provides the held-out score. The table also shows the latest P(class 1) (positive valence or high arousal), save checkboxes after stopping, and standardized weights for linear models. The EEG feature window updates only during an active run. It freezes when training stops and clears its values if the stream is lost.

g.Pype acquires EEG in its pipeline thread. Completed trial epochs enter one ordered training worker, which extracts features and refits the selected models without blocking Qt's stimulus timers. Only the GUI thread changes widgets; it picks up finished predictions on its 100 ms poll. Queued trials are trained in presentation order, and Save stays unavailable until all pending trials finish. Each trial log records `training_queue_ms` and `training_ms` so delays can be checked after a run. The live feature plot and signal lights still do small calculations on the GUI thread. Python threads improve responsiveness here, but they are not a hard real-time guarantee: verify actual image timing with a photodiode before research use.

## EEG epochs and features

The epoch starts at the baseline start, **−500 ms** before image onset, and ends at the configuration's analysis window: **+450 ms** in configuration 2, or 300 ms before image offset in configuration 1 (+2700 ms for the default 3 s image). Preprocessing is optional common average reference, notch + bandpass (from config) and **−500 to −200 ms** baseline subtraction, which keeps anticipatory activity just before onset out of the baseline. An epoch is rejected when any model channel exceeds **±150 µV** after filtering (`artifact_max_abs_uv`), is flat or has invalid scans; the report counts the reasons. Spectra use Welch's method (`psd_method`). Features use the post-onset samples.

### Feature configurations

| | Configuration 1 · full spectrum | Configuration 2 · 450 ms | Configuration 3 · small sample |
|---|---|---|---|
| Analysis window | onset → 300 ms before image offset | onset → +450 ms | onset → +3000 ms |
| Feature filter | 1–48 Hz | 1–48 Hz | **0.1–30 Hz** (artifact rejection still 1–48 Hz) |
| **Valence** | all 8 electrodes · theta, alpha, beta, gamma · band power + Higuchi FD + C4−C3 / PO8−PO7 asymmetry (48 features) | **Fz, C3, Cz, C4** (frontal + central) · **beta, gamma** · band power (8 features) | all but Fz · ERP region means (P1/N1, EPN, P3, LPP) + C4−C3 / PO8−PO7 theta/alpha asymmetry (16 features) |
| **Arousal** | all 8 electrodes · theta, alpha, beta, gamma · band power + Higuchi FD + log β/α ratio (48 features) | **PO7, Oz, PO8** (occipital) · **alpha** · band power (3 features) | all but Fz · ERP region means (P1/N1, EPN, P3, LPP) + central/parietal/posterior theta/alpha power (18 features) |

Feature families available to any target: `band_power` (log power per electrode × band), `fractal_dimension` (Higuchi), `region_band_power`, `asymmetry` (right − left log power per pair × band), `beta_alpha_ratio`, `hjorth` (mobility, complexity), `katz_fd`, `petrosian_fd`, `variance` (log), `erp` (mean amplitude per region in 100–200, 200–300, 300–450 and 450–800 ms: P1/N1, EPN, P3, LPP). A profile or a single target may set its own feature-extraction `bandpass_hz`. Edit `CONFIG["profiles"]` to change or add configurations.

**Configuration 3** is designed for runs with few trials (about 40–100). Train it with **Shrinkage LDA** (`lda_shrinkage`) for both valence and arousal; select those checkboxes, because the default is valence/SVM. Its design rules:

- **Few features, averaged over space and time.** 16–18 features instead of 48. Each feature is a mean over a region and a time window, which averages out noise in a single trial.
- **Well-replicated picture-viewing markers.** For arousal, the centro-parietal late positive potential (LPP) and the posterior EPN grow with arousal for both pleasant and unpleasant pictures. Arousing pictures also suppress posterior alpha. For valence, there is only weak evidence: an early negativity bias (P1/N1, EPN) and right-minus-left asymmetry. Classical frontal F3/F4 alpha asymmetry cannot be computed with this cap.
- **Keep trials.** Fz is left out because blinks reach it and would reject the whole epoch. Gamma and beta are left out because on dry electrodes they mostly carry facial EMG, which follows valence through smiling and frowning, not the brain.
- **A 0.1 Hz high-pass for the features.** With the 1–48 Hz filter applied to the epoch, a 4 µV LPP keeps about 0.7 µV in the 450–800 ms window. At 0.1 Hz it keeps about 3 µV (see `tests/test_profiles.py`).
- **Shrinkage LDA.** The covariance is shrunk with the analytic Ledoit–Wolf intensity, so the model stays well-conditioned even with fewer trials than features. This is the standard choice for ERP classification with few trials (Blankertz et al. 2011, *NeuroImage* 56:814).

Expect about 55–65% balanced accuracy, possibly more for arousal. A simulation with a 4 µV LPP in about 20 µV single-trial noise gave about 57–59% for arousal at 40–80 trials. Valence is the weaker target with this montage. Use the report's permutation p-value and band ablation to check a participant's result.

**Higuchi fractal dimension** measures how irregular or complex a signal is. The signal is resampled with steps k = 1…kmax, and the curve length L(k) is measured at each step. For a smooth signal L(k) shrinks roughly like 1/k, giving FD ≈ 1. For white noise it shrinks like 1/k², giving FD ≈ 2. FD is the slope of log L(k) against log(1/k). Unlike band power, it captures the shape of the signal across all frequencies at once, so it can register changes in complexity that a single band misses.

Longer images change the feature definitions, so keep the duration consistent when training and applying a saved model. The minimum image duration is 0.7 s and grows if the ERP windows need more. These are candidate measures, not established emotion signatures for this headset. [Emotional picture EEG feature study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5476954/) · [Affective picture ERP study](https://pubmed.ncbi.nlm.nih.gov/18068150/) · [Temporal decoding of affective scenes](https://pubmed.ncbi.nlm.nih.gov/35931307/)

Before and after training, the EEG diagnostics window shows four seconds of the real eight-channel stream in stacked traces. Each channel is median-centered for display, with 200 µV between trace centers; invalid scans appear as gaps. OSCAR can be toggled from either window outside a session, which reconnects the device. The raw plot pauses during sessions; the feature history updates during sessions and freezes afterward. No trace is generated or replayed when the headset is disconnected.

The **Features** tab plots live features from a rolling window. The **Show** selector offers one band across all electrodes, one electrode across all bands, fractal dimension across all electrodes, or a single feature. The brain map and eight Good/Bad/Waiting ratings use a filtered 2 s window of the actual stream and continue during training. A channel is Good when its RMS is 1–75 µV and its peak-to-peak range is 4–300 µV. Ratings clear if the stream is lost. These provisional amplitude checks are **not** impedance or proof that the cap is being worn.

## Training-quality report

After every Training and Calibration run the app writes `training_reports/<session>_<participant>_cfg<N>_<time>.md` and `.json` (computed in a background thread; the status line says when it is saved). It has one section per target, **Valence** and **Arousal**, each with its features and, for every trained model:

- **Model parameters** of the final fitted model: estimator, probability method, hyperparameters, values chosen by inner cross-validation (e.g. `select__k`, `C`, ridge `alpha`) and the largest standardized weights or forest importances;
- **Prequential** metrics of the predictions made during the run (each before its trial was learned): accuracy, balanced accuracy, precision, recall, specificity, F1, ROC AUC, Brier score, confusion matrix and the majority-class baseline;
- **Cross-validation** of the same model on all collected trials (stratified, 5 folds) with the same metrics; regression models add MAE and Pearson r;
- **Permutation test**: p-value of the cross-validated balanced accuracy against shuffled labels (up to 100 permutations, fewer for slow models within a time budget);
- **Band ablation**: cross-validated balanced accuracy without each band and with that band alone, e.g. to check whether gamma helps or only adds muscle noise;
- **Possible label issues**: trials whose out-of-fold probability for their own label is below 0.2 (or cleanlab's choice when installed).

The header summarises data quality: presented trials, EEG status counts, rejection rate and artifact reasons, calibration windows and the stopping-rule result.

## Saved files

`saved_models/` contains:

- `<stem>.json`: manifest with participant, full config, channel and feature order, dataset images and resolved thresholds, timing, trial log with labels and every model's held-out predictions, per-model accuracy, and the list of models selected for saving.
- `<stem>.<target>.<classifier>.joblib`: one fitted scikit-learn pipeline per selected model.
- `<stem>.epochs.npz`: accepted epochs from all eight acquired channels (µV), including channels omitted from the model, with per-target labels (`label_<target>`, −1 = excluded middle), continuous values (`value_<target>`) and groups. The manifest records the configuration, per-target electrodes and feature order (`feature_names_by_target`) and, for calibration models, `model_kind: "state"` with the window length.
- `resting_<participant>_<time>.npz/.json`: the resting recording and the median log band power per electrode and band.

`hotbrain.training.predict_saved(path, features)` returns P(label 1) for logistic regression, calibrated SVMs and random forest. Existing SVM files saved by an earlier version have no calibrated probability output and must be retrained for this helper. SVM probabilities use cross-validated sigmoid calibration; forest probabilities are averages of tree leaf class frequencies, not calibrated. With very few EEG trials, do not interpret any of them as validated emotional intensity. joblib uses pickle, so only load files you created.

## OSCAR and timing limits

OSCAR is activated through `HybridBlack(enable_oscar=True)`. The optional pretrial wait displays fixation before the first image; the app does not perform or verify a separate OSCAR calibration. g.Pype does not expose an OSCAR calibration export/import API, so a saved calibration cannot be reused after a restart. [g.Pype OSCAR FAQ](https://gpype.gtec.at/content/5_faq/index.html)

Each image is marked immediately after Qt paints it. The image, fixation and optional settling periods use precise Qt timers; coarse timers could end a 3 s image about 150 ms early. A g.Pype `Marker` and `Router` place the image event in the same stream as the Unicorn's EEG. Epochs use the marker's integer sample index, plus **10 samples (40 ms)** for the Unicorn's documented acquisition-to-host delay, minus the measured Qt paint-to-marker interval rounded to samples. They no longer use the first Bluetooth frame's arrival time or assume that the headset clock stays exactly at 250 Hz throughout a run. OSCAR's processing delay is handled within g.Pype's event placement. The app rejects any epoch containing a scan whose Unicorn validation flag is 0 (the driver may fill lost scans), as well as non-finite samples, and logs `invalid_scans` for that trial. A brief invalid scan no longer stops the run. The run does stop when frames are stale for more than 500 ms, when no real scan appears for one second, or when g.Pype reports a condition other than Healthy or Warning. The trial log records the marker code, sample index, corrected EEG onset index, and Qt paint-to-marker delay; a delay over 20 ms stops the run. [g.Pype time and synchronisation](https://gpype.gtec.at/content/3_architecture_design/time_synchronisation.html) · [Unicorn validation channel](https://gpype.gtec.at/content/2_gpype_training/season3_gearing_up/s3e4_hybrid_black.html)

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
- **Per-target configurations and training optimisation** (accepted items of the critical review): selectable configurations 1/2 with separate electrodes, bands and features for valence and arousal (configuration 2: 450 ms window, arousal = occipital alpha, valence = fronto-central beta/gamma); theta band; Welch PSD; optional CAR; ±150 µV epoch rejection; −500…−200 ms baseline; β/α ratio, Hjorth, Katz/Petrosian FD and variance features; balanced class weights; feature selection with nested CV; ridge regression for the V–A plane; arousal middle tercile excluded and thresholds from the training split; training-quality report after every run (metrics, permutation test, band ablation, label issues, model parameters per target); calibration session with 2 s state windows for the test; resting baseline; stopping rule; colour/beep feedback and practice runs; validation permutation p-values.
- **Docs**: Polish summaries of the papers in `docs/papers/`, and a document on possible improvements with a critical review in `docs/optymalizacje_i_ocena_krytyczna.md`.
- **Classifiers**: a registry (`hotbrain/classifiers.py`) with logistic regression, RBF SVM, linear SVM, shallow random forest, shrinkage LDA and ridge regression. The GUI selects individual target/model pairs, up to three per target. Each selected pair is refitted in the training worker after every usable EEG trial and evaluated test-before-train; the GUI shows rolling accuracy and linear weights where available. SVMs use cross-validated sigmoid calibration to provide probabilities.
- **Trial plan**: trials are balanced across the `balance_by` groups (default: valence classes) or across dataset folders, and the trial count must be a multiple of the number of groups.
- **Saving**: save checkboxes select fitted target/model pairs. One `.joblib` per selected model, a manifest with the full config, resolved thresholds and per-trial predictions, and epochs with labels and continuous target values.
- **GUI**: the electrode map follows the configured montage and has 10-20 positions for common electrodes.
- **Tests**: updated for the new montage and features. New tests cover the real dataset, config validation, target formulas, fractal dimension and multi-model training.

## Taste experiment: "Gorzki smak porażki i słodko-słony sukces (?)"

This is a separate experiment that classifies **sweet vs salty** taste. Open it by choosing **Gorzki smak porażki i słodko-słony sukces (?)** in the **Preset** (Configuration) list. The three windows are replaced by the taste windows, and the running Unicorn connection is handed over, so the headset does not reconnect. To return, choose **Experiment → Configuration → Scene emotion trainer**. The scene experiment itself is unchanged. The procedure and the reasons for it are documented in Polish in [`docs/smak_procedura.md`](docs/smak_procedura.md).

- **Trial**: a fixation cross for 5–7 s (the trial's own baseline is −3…−0.5 s), then the cue word **SŁODKIE** or **SŁONE** for 3 s while the participant puts the tastant on the tongue. Next comes a fixation dot that looks the same for both tastes, until +20 s. Finally "Przełknij i przepłucz usta wodą" is shown for 20 s. There are 20 trials per taste, balanced, with at most 2 equal tastes in a row; 40 trials take about 31 min.
- **Classified signal**: +5…+15 s after the cue, after the hand, jaw and tongue artifacts and the visual cue response are over. The signal is cut into 2 s windows every 1 s, and each window is expressed relative to the pre-cue baseline. Windows are rejected above ±150 µV or when the 30–45 Hz RMS exceeds 10 µV (EMG guard). A trial needs at least 3 clean windows.
- **Features** (preset 1, 18 features): Fz, C3, Cz, C4, Pz · theta, alpha, low beta (13–20 Hz) log power + C4−C3 asymmetry. Presets 2 (region means) and 3 (exploratory, 56 features) are alternatives. Defaults are shrinkage LDA and logistic regression.
- **Evaluation**: each trial is predicted before it is learned, and the trial P(sweet) is the mean over its windows. After training, a report in `training_reports/taste_*` gives trial-level cross-validation grouped by trial, so windows of one trial are never split across folds, plus a permutation test that shuffles whole-trial labels.
- **Sessions**: Training and Test (frozen models, live P(sweet) gauge). **Blind trials** show only "SMAKUJ" while the operator panel names the taste to give, which tests whether the model decodes the taste rather than the cue word. Press **B** to exclude a spoiled trial and **Escape** to stop.
- **Configuration**: `TASTE_CONFIG` in `hotbrain/taste.py`. The electrodes and wiring are taken from `CONFIG["eeg"]`.

## Checks

```bash
.venv/bin/python -m pytest -q
```
