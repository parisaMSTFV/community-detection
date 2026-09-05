import pandas as pd
import pytest

from community_detection.schema import (
    DataValidationError,
    load_reference_assignments,
    load_weighted_edge_list,
    pseudonymize_identifiers,
    validate_inputs,
)
from community_detection.synthetic import generate_interactions, simulate_temporal_windows


def _remove_user_from_training(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    selected = result["user_id"] == result.iloc[0]["user_id"]
    result.loc[selected, "train_weight"] = 0
    result.loc[selected, "interaction_weight"] = result.loc[selected, "test_weight"]
    return result[result["interaction_weight"] > 0]


def test_valid_synthetic_inputs_pass() -> None:
    interactions, truth = generate_interactions(users=200)
    summary = validate_inputs(simulate_temporal_windows(interactions), truth)
    assert summary.users == 200
    assert summary.categories == 30
    assert summary.checks_passed == 11


def test_duplicate_edges_are_rejected() -> None:
    interactions, truth = generate_interactions(users=200)
    split = simulate_temporal_windows(interactions)
    duplicate = split.iloc[[0]]
    split = pd.concat([split, duplicate], ignore_index=True)
    with pytest.raises(DataValidationError, match="edge must be unique"):
        validate_inputs(split, truth)


def test_missing_truth_nodes_are_rejected() -> None:
    interactions, truth = generate_interactions(users=200)
    with pytest.raises(DataValidationError, match="must match"):
        validate_inputs(simulate_temporal_windows(interactions), truth.iloc[:-1])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda frame: frame.drop(columns="train_weight"), "Missing interaction"),
        (lambda frame: frame.assign(train_weight=-1), "non-negative"),
        (lambda frame: frame.assign(interaction_weight=0), "reconcile"),
        (_remove_user_from_training, "Every synthetic user"),
    ],
)
def test_invalid_synthetic_invariants_are_rejected(change, message: str) -> None:
    interactions, truth = generate_interactions(users=200)
    temporal = simulate_temporal_windows(interactions)
    with pytest.raises(DataValidationError, match=message):
        validate_inputs(change(temporal), truth)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"user_id": ["U1"]}), "Missing edge-list"),
        (
            pd.DataFrame(
                {"user_id": ["U1", "U2"], "category_id": ["C1", "C2"], "weight": ["x", 1]}
            ),
            "numeric",
        ),
        (
            pd.DataFrame(
                {"user_id": ["U1", "U2"], "category_id": ["C1", "C2"], "weight": ["inf", 1]}
            ),
            "finite",
        ),
        (
            pd.DataFrame(
                {"user_id": ["U1", "U2"], "category_id": ["C1", "C2"], "weight": [1, 1]}
            ).assign(user_id=lambda value: value["user_id"].where(value.index != 0, "U\n1")),
            "control character",
        ),
        (
            pd.DataFrame(
                {"user_id": ["U1", "U2"], "category_id": ["C1", "C2"], "weight": [1, 1]}
            ).assign(user_id=lambda value: value["user_id"].where(value.index != 0, "U" * 129)),
            "128 characters",
        ),
    ],
)
def test_edge_list_contract_rejects_unsafe_values(
    tmp_path, frame: pd.DataFrame, message: str
) -> None:
    path = tmp_path / "edges.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match=message):
        load_weighted_edge_list(path)


def test_missing_and_empty_csv_files_are_rejected(tmp_path) -> None:
    with pytest.raises(DataValidationError, match="does not exist"):
        load_weighted_edge_list(tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DataValidationError, match="readable UTF-8"):
        load_weighted_edge_list(empty)


def test_short_salt_and_invalid_reference_are_rejected(tmp_path) -> None:
    edges = pd.DataFrame({"user_id": ["U1"], "category_id": ["C1"], "weight": [1]})
    assignments = pd.DataFrame({"node_id": ["U1"], "node_type": ["user"], "community": [0]})
    with pytest.raises(DataValidationError, match="at least 16"):
        pseudonymize_identifiers(edges, assignments, "short")

    reference = tmp_path / "reference.csv"
    pd.DataFrame({"node_id": ["U1"], "node_type": ["invalid"], "community": [0]}).to_csv(
        reference, index=False
    )
    with pytest.raises(DataValidationError, match="invalid node type"):
        load_reference_assignments(reference)
