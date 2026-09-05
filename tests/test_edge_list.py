import json
from pathlib import Path

import pandas as pd
import pytest

from community_detection.config import AnalysisConfig
from community_detection.pipeline import run_edge_list_pipeline
from community_detection.schema import (
    DataValidationError,
    load_reference_assignments,
    load_weighted_edge_list,
    pseudonymize_identifiers,
)


def _valid_edges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["U1", "U1", "U2", "U2", "U3", "U3", "U4", "U4"],
            "category_id": ["C1", "C2", "C1", "C2", "C2", "C3", "C3", "C4"],
            "weight": [8, 6, 7, 9, 1, 8, 7, 9],
        }
    )


def _small_config() -> AnalysisConfig:
    return AnalysisConfig(
        users=300,
        resolution_candidates=(0.8, 1.0),
        louvain_seeds=(11, 23, 37),
        null_permutations=20,
    )


def test_edge_list_contract_is_canonical_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "edges.csv"
    _valid_edges().sample(frac=1, random_state=7).to_csv(path, index=False)
    edges, summary = load_weighted_edge_list(path)
    assert list(edges.columns) == ["user_id", "category_id", "weight"]
    assert edges.equals(edges.sort_values(["user_id", "category_id"], ignore_index=True))
    assert summary.edge_rows == 8
    assert summary.total_weight == 55


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "edge must be unique"),
        (lambda frame: frame.assign(weight=0), "must be positive"),
        (lambda frame: frame.assign(user_id="=1+1"), "formula prefix"),
    ],
)
def test_invalid_edge_lists_are_rejected(tmp_path: Path, mutator, message: str) -> None:
    path = tmp_path / "edges.csv"
    mutator(_valid_edges()).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match=message):
        load_weighted_edge_list(path)


def test_edge_list_pipeline_writes_unlabeled_outputs(tmp_path: Path) -> None:
    edge_path = tmp_path / "edges.csv"
    output_root = tmp_path / "output"
    _valid_edges().to_csv(edge_path, index=False)

    metrics = run_edge_list_pipeline(
        edge_path,
        output_root,
        config=_small_config(),
        identifier_salt="test-only-salt-12345",
    )

    required = [
        "reports/community_assignments.csv",
        "reports/community_profiles.csv",
        "reports/component_diagnostics.csv",
        "reports/community_quality.csv",
        "reports/model_manifest.json",
        "reports/resolution_search.csv",
        "reports/stability_pairs.csv",
        "reports/metrics.json",
        "reports/run_summary.md",
        "reports/figures/community_sizes.png",
    ]
    assert all((output_root / path).exists() for path in required)
    assert metrics["mode"] == "weighted_bipartite_edge_list"
    assert metrics["graph"]["edges"] == 8
    assert "ground truth" in metrics["evaluation_boundary"].lower()
    assert metrics["quality_gate"]["status"] == "review_required"
    assignments = pd.read_csv(output_root / "reports/community_assignments.csv", dtype="string")
    assert assignments["node_id"].str.startswith(("usr_", "cat_")).all()
    assert json.loads((output_root / "reports/metrics.json").read_text()) == metrics


def test_identifiers_keep_leading_zeroes_and_typed_namespaces(tmp_path: Path) -> None:
    path = tmp_path / "edges.csv"
    frame = _valid_edges()
    frame.loc[0, "user_id"] = "001"
    frame.loc[0, "category_id"] = "001"
    frame.loc[1, "user_id"] = "کاربر-۰۰۲"
    frame.loc[1, "category_id"] = "دسته-کتاب"
    frame.to_csv(path, index=False)
    edges, _ = load_weighted_edge_list(path)
    assert "001" in set(edges["user_id"])
    assert "001" in set(edges["category_id"])
    assert "کاربر-۰۰۲" in set(edges["user_id"])
    assert "دسته-کتاب" in set(edges["category_id"])


def test_pseudonyms_are_stable_and_namespace_aware() -> None:
    edges = _valid_edges()
    assignments = pd.DataFrame(
        {
            "node_id": ["same", "same"],
            "node_type": ["user", "category"],
            "community": [0, 0],
        }
    )
    _, first = pseudonymize_identifiers(edges, assignments, "test-only-salt-12345")
    _, second = pseudonymize_identifiers(edges, assignments, "test-only-salt-12345")
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "node_id"] != first.loc[1, "node_id"]


def test_reference_assignment_contract(tmp_path: Path) -> None:
    path = tmp_path / "reference.csv"
    pd.DataFrame(
        {"node_id": ["001", "C1"], "node_type": ["user", "category"], "community": [2, 2]}
    ).to_csv(path, index=False)
    reference = load_reference_assignments(path)
    assert set(reference["node_id"]) == {"001", "C1"}


def test_temporal_comparison_and_reference_alignment(tmp_path: Path) -> None:
    current_path = tmp_path / "current.csv"
    future_path = tmp_path / "future.csv"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    current = _valid_edges()
    current.to_csv(current_path, index=False)
    future = current.copy()
    future["weight"] = [7, 5, 8, 8, 2, 9, 6, 10]
    future.to_csv(future_path, index=False)

    run_edge_list_pipeline(
        current_path,
        first_output,
        config=_small_config(),
        identifier_policy="raw",
    )
    metrics = run_edge_list_pipeline(
        current_path,
        second_output,
        config=_small_config(),
        future_edges_path=future_path,
        reference_assignments_path=first_output / "reports/community_assignments.csv",
        identifier_policy="raw",
    )
    assert metrics["temporal"]["partition_stability"]["common_users"] == 4
    assert metrics["label_alignment"]["overlapping_nodes"] == 8
    assert (second_output / "reports/temporal_null_distribution.csv").exists()
    assert (second_output / "reports/label_alignment.csv").exists()


def test_external_pipeline_requires_valid_privacy_policy(tmp_path: Path) -> None:
    edge_path = tmp_path / "edges.csv"
    _valid_edges().to_csv(edge_path, index=False)
    with pytest.raises(DataValidationError, match="salt is required"):
        run_edge_list_pipeline(edge_path, tmp_path / "missing-salt", config=_small_config())
    with pytest.raises(DataValidationError, match="identifier_policy"):
        run_edge_list_pipeline(
            edge_path,
            tmp_path / "bad-policy",
            config=_small_config(),
            identifier_policy="unsafe",
        )
