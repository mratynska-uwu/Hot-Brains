
=======
# Hot Brain scene training prototype

This is the training stage for a participant-specific happy/sad EEG classifier. The label comes from the image folder. A held-out prediction is made from EEG **before** that trial is added to the fit. No image pixels or labels enter the model as predictors.

## Run

On macOS Apple silicon, Windows x64, or Linux x64, use Python 3.10–3.14:

```bash
cd /Users/szymonlic/Desktop/Hot-Brains
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-device.txt
.venv/bin/python run.py
```

For presentation debugging without a headset, `requirements.txt` is enough. The app keeps retrying the Unicorn connection while the Qt loop remains responsive. On macOS, grant Bluetooth permission to the application running Python. The official g.Pype package supplies `HybridBlack` and the optional `enable_oscar=True` path. [g.Pype repository](https://github.com/gtec-neurotechnology/gpype) · [device reference](https://gpype.gtec.at/content/7_sdk_reference/backend_sources/hybrid_black.html) · [platform notes](https://gpype.gtec.at/content/1_basic_concepts/supported_devices.html)

## Operator flow

1. Put images in `training_dataset/happy_images/` and `training_dataset/sad_images/`. Existing images are included. Each folder needs at least one image.
2. Set a participant ID and Unicorn serial (pre-filled with `UN-2019.03.01`; leave blank to use the first available device), then choose a finite, even trial count (default 160) or continuous mode. Finite runs shuffle an exactly balanced label list. Exact balance means each individual draw cannot also remain independent at 50%; continuous mode uses independent 50/50 draws. Each class cycles through its images without replacement before reusing them.
3. Set image duration (default 1.0 s, range 0.7–60 s) and fixation minimum/maximum (default 0.6–1.0 s, range 0.2–60 s). Each fixation duration is drawn uniformly between the chosen limits. Choose OSCAR and optional pretrial settling. For a recorded session, wait until the device status says connected before starting; trials shown during connection setup have no complete EEG epoch. Start. The image tab opens in an independent window; maximize it on the participant display. The other two tabs can also detach.
4. Stop with the button or the selected stop key (Escape by default) while any app window has focus, or let a finite run end. Choose the key before starting; it is disabled between runs. Save the fitted model if at least ten complete EEG epochs with both labels were accepted, or discard the run. With no EEG, the trial sequence still runs and no model/accuracy is fabricated.

The EEG epoch starts **−200 ms** before image onset and ends **300 ms before image offset**: +700 ms for a 1 s image, +1700 ms for a 2 s image. This is a configurable analysis convention, not a universal EEG rule. It leaves out the final 300 ms of each image and its offset response. The minimum 0.7 s image duration keeps the late ERP window nonempty. A 1–35 Hz bandpass and −200 to 0 ms baseline subtraction precede 20 features: regional theta/alpha/beta power, C3/C4 and PO7/PO8 log-power asymmetries, and early/middle/late ERP amplitudes. Spectral features use the whole post-onset epoch; late ERP and salience candidate features span 350 ms to the chosen epoch end. Early and middle ERP windows stay at 100–200 and 200–350 ms. Longer durations therefore change the feature definitions and should be kept consistent when training and applying a saved model. Absolute posterior late amplitude is a *salience candidate*. The default montage is `Fz, C3, Cz, C4, Pz, PO7, Oz, PO8`; verify electrode placement before collecting participant data. The Unicorn has no paired frontal electrodes in this montage, so frontal asymmetry is not calculated. The fit is `StandardScaler` plus L2 logistic regression. The GUI shows all standardized coefficients and the rolling accuracy of up to 20 **test-before-train** predictions. Early trials before a model exists are omitted from accuracy.

The EEG features window shows eight live signal-check lights over a top-view brain diagram, one per channel (forehead at the top). After two seconds of streaming, a light is green when that channel's 1–35 Hz signal has an RMS of 1–75 µV and a peak-to-peak range of 4–300 µV; otherwise it is red. Hover over an electrode to see the measurements and reason. Grey means disconnected or waiting for enough data. These are provisional amplitude checks on the stream available to the app (OSCAR-processed when OSCAR is enabled), **not** impedance or a verified contact-quality measurement. A plausible-looking disconnected electrode can still appear green, so inspect traces and electrode placement before recording.

Saved runs contain numeric `.npz` model weights, a `.json` manifest, and an `.epochs.npz` array of accepted EEG epochs (microvolts) in `saved_models/`. The manifest includes participant ID, feature order, channel order, OSCAR setting, image/fixation durations, epoch bounds and sample count, presentation order, onset times, held-out predictions, and any skipped EEG trials. `hotbrain.training.predict_saved(stem, features)` computes P(happy) without loading a pickle. Discarding a run removes its in-memory data without writing files.

## OSCAR and timing limits

OSCAR is activated through `HybridBlack(enable_oscar=True)`. The optional pretrial period lets its online processing settle while fixation is displayed. The public g.Pype/`gtec_oscar` interfaces inspected here do **not** expose an OSCAR calibration export/import API, so this app cannot reuse a saved OSCAR calibration after a restart; skipping the settling period uses OSCAR's current online state. The interface says this explicitly. [g.Pype OSCAR FAQ](https://gpype.gtec.at/content/5_faq/index.html)

Onset is timestamped after Qt paints the image. EEG sample times are anchored to the first frame's host arrival minus the Unicorn's documented device delay and g.Pype's reported OSCAR delay, then advanced at the fixed 250 Hz sample rate. This avoids false gaps from bursty Bluetooth callback timing but remains a software timing approximation. Before using accuracy as research evidence, validate timing against a photodiode or hardware trigger and test actual headset reconnection and OSCAR behavior. On 2026-11-14, a live macOS GUI check opened Unicorn `UN-2019.03.01` with OSCAR on and received 996 finite samples from eight channels in four seconds. The full GUI also plotted live features and reached the epoch artifact check. The headset was connected but not worn, so epochs were rejected for artifact/flat channels; this does not establish usable EEG quality. The test needs the normal macOS GUI event loop: Qt's `offscreen` platform did not service the IOBluetooth main run loop.

The two supplied pictures are emoji faces. They are useful for debugging presentation, but they must be replaced by **scenes** for the proposed scene-to-face generalization experiment. Repeating two pictures can also inflate measured performance; held-out accuracy is on new presentations, not unseen image identities.

The repository's original `first_try.py` is retained as a separate g.Pype example. It is not part of the trainer. On macOS, `macos/Hot Brain Trainer.app` is a launcher bundle; copy it to the Desktop after creating `.venv` in this repository.

## Why these features

The feature families are candidate measures, not established happy/sad signatures for this headset or timing window. Prior participant-level work evaluated frontal theta, alpha, and late positivity for emotional pictures and found limited generalization; affective picture studies identify early and late ERP windows. Posterior/central power asymmetries here are exploratory substitutes allowed by this montage. A larger, diverse image set, artifact review, participant-specific cross-validation, and an independent face test remain necessary. [Emotional picture EEG feature study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5476954/) · [Affective picture ERP study](https://pubmed.ncbi.nlm.nih.gov/18068150/) · [Temporal decoding of affective scenes](https://pubmed.ncbi.nlm.nih.gov/35931307/)

## Checks

```bash
.venv/bin/python -m pytest -q
```
>>>>>>> ef2cdcc384d70bac328ea1e2d7db20ea62c0de65
