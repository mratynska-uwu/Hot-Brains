from pathlib import Path

from hotbrain.config import CONFIG
from hotbrain.gui import launch


if __name__ == "__main__":
    # Edit CONFIG in hotbrain/config.py (datasets, electrodes, filters, bands,
    # targets, classifiers) or pass a modified copy here.
    raise SystemExit(launch(Path(__file__).resolve().parent, CONFIG))
