from pathlib import Path

from hotbrain.gui import launch


if __name__ == "__main__":
    raise SystemExit(launch(Path(__file__).resolve().parent))
