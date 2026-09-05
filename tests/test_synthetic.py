import pandas as pd
import pytest

from community_detection.synthetic import generate_interactions, simulate_temporal_windows


def test_generation_is_deterministic() -> None:
    first_interactions, first_truth = generate_interactions(seed=42, users=200)
    second_interactions, second_truth = generate_interactions(seed=42, users=200)
    pd.testing.assert_frame_equal(first_interactions, second_interactions)
    pd.testing.assert_frame_equal(first_truth, second_truth)


def test_interactions_do_not_expose_planted_labels() -> None:
    interactions, _ = generate_interactions(users=200)
    assert "planted_community" not in interactions.columns


def test_temporal_windows_reconcile_and_retain_training_population() -> None:
    interactions, _ = generate_interactions(users=200)
    split = simulate_temporal_windows(interactions)
    assert (split.groupby("user_id")["train_weight"].sum() >= 1).all()
    assert (split["train_weight"] + split["test_weight"] == split["interaction_weight"]).all()


def test_temporal_windows_are_deterministic_and_include_unseen_edges() -> None:
    interactions, _ = generate_interactions(users=300)
    first = simulate_temporal_windows(interactions, seed=17)
    second = simulate_temporal_windows(interactions, seed=17)
    pd.testing.assert_frame_equal(first, second)
    assert (first.groupby("user_id")["train_weight"].sum() > 0).all()
    assert ((first["train_weight"] == 0) & (first["test_weight"] > 0)).any()


def test_invalid_temporal_windows_are_rejected() -> None:
    interactions, _ = generate_interactions(users=200)
    with pytest.raises(ValueError, match="Temporal windows"):
        simulate_temporal_windows(interactions, train_window_days=0)
