import pandas as pd
import pytest

from community_detection.synthetic import generate_interactions, split_interaction_weights


def test_generation_is_deterministic() -> None:
    first_interactions, first_truth = generate_interactions(seed=42, users=200)
    second_interactions, second_truth = generate_interactions(seed=42, users=200)
    pd.testing.assert_frame_equal(first_interactions, second_interactions)
    pd.testing.assert_frame_equal(first_truth, second_truth)


def test_interactions_do_not_expose_planted_labels() -> None:
    interactions, _ = generate_interactions(users=200)
    assert "planted_community" not in interactions.columns


def test_split_reconciles_and_retains_training_weight() -> None:
    interactions, _ = generate_interactions(users=200)
    split = split_interaction_weights(interactions)
    assert (split["train_weight"] >= 1).all()
    assert (split["train_weight"] + split["test_weight"] == split["interaction_weight"]).all()


def test_invalid_holdout_fraction_is_rejected() -> None:
    interactions, _ = generate_interactions(users=200)
    with pytest.raises(ValueError, match="holdout_fraction"):
        split_interaction_weights(interactions, holdout_fraction=0.8)
