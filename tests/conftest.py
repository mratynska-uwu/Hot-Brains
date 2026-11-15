"""Shared test helpers."""
import copy

from hotbrain.config import CONFIG


def classic_config() -> dict:
    """CONFIG without stimulus conditions: valence > 0 and median-split arousal, balanced by valence.

    For tests of the generic threshold / exclude_middle / folder mechanics and of
    small synthetic datasets whose CSVs have only valence and arousal columns.
    """
    config = copy.deepcopy(CONFIG)
    config.pop("conditions", None)
    config["balance_by"] = "valence"
    config["targets"] = {
        "valence": {"column": "valence", "threshold": 0, "classes": ["negative", "positive"]},
        "arousal": {"column": "arousal", "threshold": "median", "classes": ["low", "high"],
                    "exclude_middle": 1 / 3},
    }
    return config
