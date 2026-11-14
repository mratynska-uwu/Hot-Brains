import argparse
from pathlib import Path

from hotbrain.config import CONFIG
from hotbrain.gui import launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hot Brain scene training GUI")
    parser.add_argument("--simulate", action="store_true",
                        help="start with simulated EEG instead of the Unicorn headset (can also be toggled in the GUI)")
    args = parser.parse_args()
    # Edit CONFIG in hotbrain/config.py (datasets, electrodes, filters, bands,
    # targets, classifiers) or pass a modified copy here.
    raise SystemExit(launch(Path(__file__).resolve().parent, CONFIG, simulate=args.simulate))
