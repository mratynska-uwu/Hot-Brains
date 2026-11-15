from pathlib import Path
import argparse

from hotbrain.config import CONFIG
from hotbrain.gui import launch
from hotbrain.face_pilot import launch_face_pilot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hot Brain trainer")
    parser.add_argument("--results", type=Path, help="Open a saved run in the results viewer")
    parser.add_argument("--scenes", action="store_true", help="Open the previous scene-training application")
    args = parser.parse_args()
    # Edit CONFIG in hotbrain/config.py (datasets, electrodes, filters, bands,
    # targets, classifiers) or pass a modified copy here.
    root = Path(__file__).resolve().parent
    raise SystemExit(launch(root, CONFIG, args.results) if args.scenes or args.results else launch_face_pilot(root))
