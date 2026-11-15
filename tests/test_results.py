from hotbrain.results import replay_curves


def test_saved_results_rebuild_training_and_holdout_curves():
    manifest = {"config": {"targets": {"valence": {"classes": ["negative", "positive"]}}},
                "trial_log": [
                    {"number": 1, "trial_role": "training", "labels": {"valence": "positive"},
                     "predictions": {"valence/logreg": {"p_class1": .8, "correct": True}}},
                    {"number": 2, "trial_role": "validation", "labels": {"valence": "negative"},
                     "predictions": {"valence/logreg": {"p_class1": .3, "correct": True}}},
                    {"number": 3, "trial_role": "training", "labels": {"valence": "negative"},
                     "predictions": {"valence/logreg": {"p_class1": .6, "correct": False}}},
                ]}
    curves = replay_curves(manifest, window=2)
    train = curves[("valence/logreg", "training")]
    heldout = curves[("valence/logreg", "validation")]
    assert train["trial"] == [1, 3]
    assert train["accuracy"] == [100, 50]
    assert abs(train["brier"][-1] - .2) < 1e-12
    assert heldout["trial"] == [2]
    assert heldout["accuracy"] == [100]
    assert abs(heldout["brier"][0] - .09) < 1e-12
