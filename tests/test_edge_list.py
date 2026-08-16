import json
from pathlib import Path

import pandas as pd
import pytest

from community_detection.config import AnalysisConfig
from community_detection.pipeline import run_edge_list_pipeline
from community_detection.schema import DataValidationError, load_weighted_edge_list


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
        seed=42,
        users=300,
        holdout_fraction=0.2,
        louvain_resolution=1.0,
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
        (
            lambda frame: frame.assign(
                category_id=lambda value: value["category_id"].where(value.index != 0, "U4")
            ),
            "disjoint namespaces",
        ),
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

    metrics = run_edge_list_pipeline(edge_path, output_root, config=_small_config())

    required = [
        "reports/community_assignments.csv",
        "reports/community_profiles.csv",
        "reports/stability_pairs.csv",
        "reports/metrics.json",
        "reports/run_summary.md",
        "reports/figures/community_sizes.png",
    ]
    assert all((output_root / path).exists() for path in required)
    assert metrics["mode"] == "weighted_bipartite_edge_list"
    assert metrics["graph"]["edges"] == 8
    assert "ground truth" in metrics["evaluation_boundary"].lower()
    assert json.loads((output_root / "reports/metrics.json").read_text()) == metrics
