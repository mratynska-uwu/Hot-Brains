from pathlib import Path
import argparse

from hotbrain.config import CONFIG
from hotbrain.gui import launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hot Brain trainer")
    parser.add_argument("--results", type=Path, help="Open a saved run in the results viewer")
    args = parser.parse_args()
    # Edit CONFIG in hotbrain/config.py (datasets, electrodes, filters, bands,
    # targets, classifiers) or pass a modified copy here.
    raise SystemExit(launch(Path(__file__).resolve().parent, CONFIG, args.results))
