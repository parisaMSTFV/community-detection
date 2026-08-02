import pandas as pd
import pytest

from community_detection.schema import DataValidationError, validate_inputs
from community_detection.synthetic import generate_interactions, split_interaction_weights


def test_valid_synthetic_inputs_pass() -> None:
    interactions, truth = generate_interactions(users=200)
    summary = validate_inputs(split_interaction_weights(interactions), truth)
    assert summary.users == 200
    assert summary.categories == 30
    assert summary.checks_passed == 9


def test_duplicate_edges_are_rejected() -> None:
    interactions, truth = generate_interactions(users=200)
    split = split_interaction_weights(interactions)
    duplicate = split.iloc[[0]]
    split = pd.concat([split, duplicate], ignore_index=True)
    with pytest.raises(DataValidationError, match="edge must be unique"):
        validate_inputs(split, truth)


def test_missing_truth_nodes_are_rejected() -> None:
    interactions, truth = generate_interactions(users=200)
    with pytest.raises(DataValidationError, match="must match"):
        validate_inputs(split_interaction_weights(interactions), truth.iloc[:-1])
