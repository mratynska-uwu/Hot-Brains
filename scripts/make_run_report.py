"""Create the hackathon decision report from the curated saved run."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "saved_models/scene_debug_20261115_024214_587640_trimmed.json"
OUT = ROOT / "output/pdf/hot_brain_run_2026-11-15_trimmed_results.pdf"
TMP = ROOT / "tmp/pdfs"
NAVY = "#182335"
TEAL = "#148d7c"
BLUE = "#4267bd"
ORANGE = "#c6772c"


def style_charts():
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": NAVY, "text.color": NAVY,
                         "xtick.color": NAVY, "ytick.color": NAVY})


def live_series(trials, role):
    rows = [(r["number"], r["predictions"]["valence/logreg"])
            for r in trials if r["trial_role"] == role
            and "valence/logreg" in r.get("predictions", {})]
    xs = np.asarray([n for n, _ in rows])
    truth = np.asarray([int(next(r for r in trials if r["number"] == n)["labels"]["valence"] == "positive")
                        for n in xs])
    p = np.asarray([v["p_class1"] for _, v in rows])
    hit = (p >= .5) == truth
    squared = (p - truth) ** 2
    rolling = lambda a: np.asarray([np.mean(a[max(0, i - 19):i + 1]) for i in range(len(a))])
    return xs, rolling(hit), rolling(squared)


def charts(data):
    TMP.mkdir(parents=True, exist_ok=True)
    style_charts()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.2, 6.8), sharex=True,
                                      gridspec_kw={"hspace": .30})
    for role, label, color in (("training", "Training stream", TEAL),
                               ("validation", "Held-out stream", BLUE)):
        x, accuracy, brier = live_series(data["trial_log"], role)
        top.plot(x, accuracy * 100, color=color, lw=1.8, label=f"{label} (n={len(x)})")
        bottom.plot(x, brier, color=color, lw=1.8, label=label)
    top.axhline(50, ls="--", lw=1, color="#888888")
    bottom.axhline(.25, ls="--", lw=1, color="#888888", label="Constant 50/50")
    top.set(ylabel="Rolling correct (%)", ylim=(0, 100),
            title="Live predictions on retained scene trials (20-prediction window)")
    bottom.set(xlabel="Original scene presentation number", ylabel="Rolling Brier loss", ylim=(0, .72))
    for ax in (top, bottom):
        ax.grid(alpha=.18)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_xlim(1, 710)
        ax.axvspan(1, 10, color="#e1e5ea", alpha=.55)
        ax.axvspan(701, 710, color="#e1e5ea", alpha=.55)
    path = TMP / "trimmed_live_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 3.6),
                                      gridspec_kw={"width_ratios": [1.15, 1]})
    score = [data["offline_refit"], data["historic_saved_checkpoint_on_retained_validation"], data["gabor_check"]]
    names = ["Clean scene refit", "Historic checkpoint", "Gabor vs gray"]
    color = [TEAL, BLUE, ORANGE]
    x = np.arange(3)
    acc = [s["accuracy"] * 100 for s in score]
    ci = [binomtest(int(round(s["accuracy"] * (s.get("n") or s["usable_held_out"]))),
                    int(s.get("n") or s["usable_held_out"])).proportion_ci(.95, method="wilson")
          for s in score]
    low = [acc[i] - 100 * ci[i].low for i in range(3)]
    high = [100 * ci[i].high - acc[i] for i in range(3)]
    left.bar(x, acc, color=color, width=.55)
    left.errorbar(x, acc, yerr=[low, high], fmt="none", ecolor=NAVY, capsize=3)
    left.axhline(50, color="#777777", ls="--", lw=1)
    left.set_xticks(x, names, rotation=15, ha="right")
    left.set_ylim(0, 100)
    left.set_ylabel("Held-out accuracy (%)")
    left.set_title("Held-out accuracy and 95% Wilson interval")
    left.grid(axis="y", alpha=.18)
    brier = [s["brier"] for s in score]
    right.bar(x, brier, color=color, width=.55)
    right.axhline(.25, color="#777777", ls="--", lw=1)
    right.set_xticks(x, names, rotation=15, ha="right")
    right.set_ylim(0, .35)
    right.set_ylabel("Brier loss (lower better)")
    right.set_title("Probability quality vs 0.25 coin-flip")
    right.grid(axis="y", alpha=.18)
    fig.tight_layout()
    path2 = TMP / "heldout_scores.png"
    fig.savefig(path2, dpi=180, bbox_inches="tight")
    plt.close(fig)

    bins = range(1, 701, 100)
    bad = [sum(r["eeg"] != "usable" for r in data["trial_log"]
               if start <= r["number"] < start + 100) for start in bins]
    counts = [sum(start <= r["number"] < start + 100 for r in data["trial_log"]) for start in bins]
    fig, ax = plt.subplots(figsize=(10.2, 2.3))
    ax.bar([f"{start}-{min(start+99,700)}" for start in bins],
           np.asarray(bad) / np.asarray(counts) * 100, color=ORANGE, width=.7)
    ax.set(ylabel="Unusable EEG (%)", ylim=(0, 25),
           title="Trial-level EEG losses across the retained session")
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    path3 = TMP / "signal_losses.png"
    fig.savefig(path3, dpi=180, bbox_inches="tight")
    plt.close(fig)
    keys = ("logreg", "svm_linear", "svm", "forest")
    model_names = ("L2 logistic", "Linear SVM", "RBF SVM", "Forest")
    comparison = data["offline_comparison"]
    fig, axs = plt.subplots(2, 3, figsize=(10.2, 6.5),
                            gridspec_kw={"hspace": .55, "wspace": .32})
    for row, target in enumerate(("valence", "arousal")):
        entries = [comparison[f"{target}/{key}"]["held_out"] for key in keys]
        majority = max(entries[0]["class_0"], entries[0]["class_1"]) / entries[0]["n"]
        measures = (("accuracy", "Accuracy (%)", majority * 100, (30, 75), 100),
                    ("auc", "AUC", .5, (.35, .7), 1),
                    ("brier", "Brier loss", .25, (.18, .30), 1))
        for col, (key, title, baseline, limits, multiplier) in enumerate(measures):
            ax = axs[row, col]
            values = [item[key] * multiplier for item in entries]
            ax.bar(np.arange(4), values, color=[TEAL, BLUE, "#8f6db6", ORANGE], width=.67)
            ax.axhline(baseline, color="#777777", lw=1, ls="--")
            ax.set(ylim=limits, title=f"{target.title()}: {title}")
            ax.set_xticks(np.arange(4), model_names, rotation=30, ha="right", fontsize=7)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=.15)
    fig.suptitle("All four predeclared app classifiers - same 502 train / 124 held-out EEG epochs",
                 y=.99, fontsize=11)
    path4 = TMP / "all_models_comparison.png"
    fig.savefig(path4, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path, path2, path3, path4


def make_pdf(data):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    live, scores, losses, all_models = charts(data)
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name="TitleCustom", parent=ss["Title"], fontName="Helvetica-Bold",
                          fontSize=22, leading=25, textColor=colors.HexColor(NAVY), spaceAfter=10))
    ss.add(ParagraphStyle(name="SubtitleCustom", parent=ss["Normal"], fontSize=10, leading=14,
                          textColor=colors.HexColor(BLUE), spaceAfter=12))
    ss.add(ParagraphStyle(name="H1Custom", parent=ss["Heading1"], fontSize=14, leading=17,
                          textColor=colors.HexColor(NAVY), spaceBefore=12, spaceAfter=6))
    ss.add(ParagraphStyle(name="BodyCustom", parent=ss["BodyText"], fontSize=9.4, leading=13.4,
                          spaceAfter=7))
    ss.add(ParagraphStyle(name="SmallCustom", parent=ss["Normal"], fontSize=8, leading=11,
                          textColor=colors.HexColor("#4c5a70"), spaceAfter=5))
    ss.add(ParagraphStyle(name="TableHeader", parent=ss["Normal"], fontSize=8, leading=11,
                          textColor=colors.white))
    ss.add(ParagraphStyle(name="TinyCustom", parent=ss["Normal"], fontSize=7.5, leading=9.5,
                          textColor=colors.HexColor("#4c5a70")))
    ss.add(ParagraphStyle(name="CalloutCustom", parent=ss["BodyText"], fontSize=11.5, leading=16,
                          textColor=colors.HexColor(NAVY), spaceAfter=10))
    P = lambda t, sty="BodyCustom": Paragraph(t, ss[sty])
    story = [P("Hot Brain: one-run decision brief", "TitleCustom"),
             P("15 November 2026 | Participant debug | Unicorn Hybrid Black | OSCAR off | first/last 10 scene trials excluded",
               "SubtitleCustom"),
             P("Decision: change the next proof-of-concept task. The present valence model has not demonstrated reliable learning, and the Gabor control is inconclusive. Do a short, visually robust positive-control run before committing another long session to emotion labels.",
               "CalloutCustom")]
    rows = [["Measure", "Retained evidence", "Reading"],
            ["Scene clean refit", "60/124 = 48.4%\nAUC 0.535; Brier 0.2506", "Near chance; no useful signal shown"],
            ["Live held-out stream", "60/122 = 49.2%\nAUC 0.465; Brier 0.2931", "Predictions did not improve"],
            ["Historic best checkpoint", "68/124 = 54.8%\nAUC 0.559; Brier 0.2485", "Exploratory; selected on held-out data and trained before curation"],
            ["Gabor versus gray", "25/40 = 62.5%\nAUC 0.580; Brier 0.2575", "Suggestive, not a confirmed pass (p=0.077)"]]
    table = Table([[P(str(cell), "TableHeader" if index == 0 else "SmallCustom") for cell in row]
                   for index, row in enumerate(rows)],
                  colWidths=[115, 160, 240], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f0f4f7")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                               ("TOPPADDING", (0, 0), (-1, -1), 7),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                               ("LINEBELOW", (0, 0), (-1, -1), .35, colors.HexColor("#d4dce5"))]))
    model_rows = [["Target", "Model", "Train Brier", "Held-out acc", "Bal. acc", "AUC", "Held-out Brier"]]
    for target in ("valence", "arousal"):
        for key, label in (("logreg", "L2 logistic"), ("svm_linear", "Linear SVM"),
                           ("svm", "RBF SVM"), ("forest", "Forest")):
            entry = data["offline_comparison"][f"{target}/{key}"]
            train, test = entry["training"], entry["held_out"]
            model_rows.append([target, label, f"{train['brier']:.3f}",
                               f"{test['accuracy']:.1%}", f"{test['balanced_accuracy']:.1%}",
                               f"{test['auc']:.3f}", f"{test['brier']:.3f}"])
    model_table = Table([[P(str(cell), "TableHeader" if index == 0 else "TinyCustom")
                          for cell in row] for index, row in enumerate(model_rows)],
                        colWidths=[59, 100, 71, 77, 69, 54, 85], repeatRows=1)
    model_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
                                     ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f0f4f7")),
                                     ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 5),
                                     ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                                     ("TOPPADDING", (0, 0), (-1, -1), 4),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                     ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#d4dce5"))]))
    story += [table, Spacer(1, 8), P("What was removed", "H1Custom"),
              P("Scene presentations 1-10 and 701-710, inclusive, were excluded because the participant reported distraction. The curated copy contains 690 scene presentation records and 626 usable EEG epochs: 502 training and 124 held-out. The original 710-presentation manifest and continuous EEG remain untouched for audit. The Gabor control, which preceded scene trials, was not trimmed."),
              P("All four app classifiers were fit only on retained training epochs using the same 32 band-power/fractal features, filter and electrode set as the original run (with each model's existing scaling/calibration pipeline). The logistic score is the primary clean refit comparison above; all models appear on page 3. The historic checkpoint is shown only for context; it was selected from 26 candidates on this validation set and its training included now-excluded early trials."),
              P("The 95% Wilson interval for the clean refit accuracy is 39.8%-57.1%; for Gabor, 47.0%-75.8%. These intervals include chance. The held-out set has already informed model choices, so even the clean refit is exploratory rather than a fresh independent test.", "SmallCustom"),
              Image(str(scores), width=515, height=182),
              PageBreak(), P("What the recordings show", "TitleCustom"),
              P("The live curves below are rebuilt from the saved per-trial predictions after excluding scene trials 1-10 and 701-710. Each prediction was made before fitting on that trial; the validation stream never entered the fit.", "BodyCustom"),
              Image(str(live), width=515, height=343), Spacer(1, 7),
              P("The curves are volatile, and the average live held-out Brier loss is 0.2931, worse than a constant 50/50 forecast (0.25). A late 20-trial rolling segment near 60% does not establish a sustained effect.", "SmallCustom"),
              Image(str(losses), width=515, height=116),
              P("64 of the 690 retained scene trials had unusable EEG (63 invalid scans, 1 artifact). Losses occurred throughout the run. This is a trial-level acquisition concern, although it does not by itself explain the near-chance classification.", "SmallCustom"),
              PageBreak(), P("All app models: offline refit", "TitleCustom"),
              P("Each of the four app classifiers was freshly fit on the same 502 retained training EEG epochs and scored once on the same 124 retained held-out epochs. Both targets are exploratory; the held-out images had already informed earlier development.", "BodyCustom"),
              Image(str(all_models), width=515, height=328),
              model_table, Spacer(1, 8),
              P("Dashed baselines: accuracy = majority class (valence 52.4%, arousal 55.6%); AUC = 0.5; Brier = constant 50/50 (0.25). In-sample forest Brier is 0.202 for valence and 0.194 for arousal, but its held-out Brier worsens to 0.254 and 0.255. No model provides convincing held-out decoding. Arousal linear SVM has the best AUC (0.575) and Brier (0.245), yet its 55.6% accuracy only matches the majority-class baseline. Some calibrated SVM probabilities cluster near 0.5, so a marginal accuracy edge is not persuasive.", "SmallCustom"),
              PageBreak(), P("Evidence and the next 8 hours", "TitleCustom"),
              P("The clean scene valence refit does not beat a constant 50/50 forecast. The historic checkpoint is slightly above chance in accuracy but its Brier loss only barely improves on 0.25 after model selection on the same validation set. Gabor's 25/40 correct gives a one-sided exact binomial p=0.077; its Brier loss is worse than 0.25. Neither task has a convincing pass.", "BodyCustom"),
              P("Recommended decision", "H1Custom"),
              P("<b>Pivot the next demonstration to a visually robust task using the same EEG acquisition, onset markers, recording and held-out evaluation.</b> A face-versus-Gabor or comparable high-contrast visual-category task is a practical positive control. Predeclare one simple model and one primary metric, use new held-out presentations, and require clear improvement over chance before investing in another long emotional-valence run. Prior single-trial EEG work has decoded face-versus-Gabor states, with posterior sites informative [1]. A negative result in your setup would direct attention to signal quality and timing rather than emotion feature engineering."),
              P("Do not use the remaining hours to search dozens of features on this one validation set. That would reward chance patterns and make the next saved score hard to trust. A bounded offline check of one literature-motivated ERP family is reasonable, followed by a new run with a fresh test set. Simply extending the current weakly eliciting image task is not the best use of time. Published work on emotional-picture EEG has also found weak generalization from fitted features to test pictures [2]."),
              P("Gaze and stimulus strength", "H1Custom"),
              P("The application already centers each entire image in the presentation widget; it does not center the image's most salient object. For the next run, use a fixed screen-centered fixation mark and consistent image size/position, plus a clear instruction to keep gaze there. An automatic saliency-based image offset would alter retinal position on an image-by-image basis and may create a new label-correlated visual cue. Add it only after measuring gaze or showing that it is needed. Eye movements themselves can predict scene valence, which makes them a real confound [3]."),
              P("The participant reported that many pictures felt emotionally weak and mentally repeated the words 'happy' or 'sad'. That is a serious label-validity issue: a model could learn the deliberate word/task strategy instead of the image-evoked feeling. Avoid cued self-talk in the next emotion run. Use images the participant actually rates as strongly positive or negative, collect brief personal ratings, and retain image identity separation. Emotional ERP effects are often associated with motivational relevance and later potentials; they are not guaranteed by a dataset's nominal label [4]."),
              P("Sources and limits", "H1Custom"),
              P("[1] Pattern classification of EEG signals reveals perceptual and attentional states: https://pmc.ncbi.nlm.nih.gov/articles/PMC5405963/", "SmallCustom"),
              P("[2] Prediction of subjective ratings of emotional pictures by EEG features: https://pmc.ncbi.nlm.nih.gov/articles/PMC5476954/", "SmallCustom"),
              P("[3] scene-valence eye movements: https://pmc.ncbi.nlm.nih.gov/articles/PMC4583411/", "SmallCustom"),
              P("[4] affective picture late positive potential: https://pmc.ncbi.nlm.nih.gov/articles/PMC2993239/", "SmallCustom"),
              P("Report generated from the curated manifest and epoch archive. No new headset recording was taken. Accuracy and p-values are descriptive for this one participant and one run.", "SmallCustom")]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=40, rightMargin=40,
                            topMargin=40, bottomMargin=40, title="Hot Brain trimmed run decision brief")
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#69768a"))
        canvas.drawString(40, 22, "Hot Brain | 15 Nov 2026 | curated analysis")
        canvas.drawRightString(A4[0] - 40, 22, str(doc.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(make_pdf(data))
