import argparse
from pathlib import Path

from hotbrain.config import CONFIG
from hotbrain.gui import launch


if __name__ == "__main__":
    # Edit CONFIG in hotbrain/config.py (datasets, electrodes, filters, bands,
    # targets, classifiers) or pass a modified copy here.
    parser = argparse.ArgumentParser(description="Hot Brain EEG trainer")
    parser.add_argument("--simulate", action="store_true",
                        help="start with simulated EEG instead of the Unicorn headset (testing only; "
                             "can also be switched in the GUI under Headset → EEG source)")
    args = parser.parse_args()
    raise SystemExit(launch(Path(__file__).resolve().parent, CONFIG, simulate=args.simulate))
