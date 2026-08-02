import json
from pathlib import Path

from community_detection.config import AnalysisConfig
from community_detection.pipeline import run_pipeline


def _small_config() -> AnalysisConfig:
    return AnalysisConfig(
        seed=42,
        users=300,
        holdout_fraction=0.2,
        louvain_resolution=1.0,
        louvain_seeds=(11, 23, 37),
        null_permutations=20,
    )


def test_pipeline_writes_required_artifacts(tmp_path: Path) -> None:
    metrics = run_pipeline(tmp_path, config=_small_config())
    required = [
        "data/synthetic_interactions.csv",
        "data/synthetic_ground_truth.csv",
        "reports/metrics.json",
        "reports/community_assignments.csv",
        "reports/community_profiles.csv",
        "reports/figures/evaluation_summary.png",
        "reports/figures/community_profiles.png",
        "reports/figures/community_sizes.png",
        "reports/figures/category_projection.png",
    ]
    assert all((tmp_path / path).exists() for path in required)
    assert metrics["graph"]["detected_communities"] >= 2


def test_core_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_metrics = run_pipeline(first, config=_small_config())
    second_metrics = run_pipeline(second, config=_small_config())
    assert first_metrics["artifact_fingerprint"] == second_metrics["artifact_fingerprint"]
    assert json.loads((first / "reports/metrics.json").read_text()) == json.loads(
        (second / "reports/metrics.json").read_text()
    )
