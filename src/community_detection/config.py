"""Configuration loading for the community-detection pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int = 42
    users: int = 1200
    train_window_days: int = 90
    test_window_days: int = 30
    resolution_candidates: tuple[float, ...] = (0.8, 1.0, 1.2)
    louvain_seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    null_permutations: int = 100
    edge_weighting: str = "log_tfidf"
    min_component_users: int = 3
    min_component_categories: int = 2
    max_small_component_node_share: float = 0.25
    min_community_users: int = 5
    min_community_categories: int = 2
    min_eligible_user_coverage: float = 0.80
    max_category_weight_share: float = 0.35
    min_hub_removal_user_ari: float = 0.70
    min_temporal_user_ari: float = 0.60


def load_config(path: Path | None = None) -> AnalysisConfig:
    """Load a validated config from disk or the wheel-bundled default."""
    if path is None:
        bundled = resources.files("community_detection").joinpath("resources/analysis.json")
        raw = json.loads(bundled.read_text(encoding="utf-8"))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    raw["louvain_seeds"] = tuple(int(seed) for seed in raw["louvain_seeds"])
    raw["resolution_candidates"] = tuple(float(value) for value in raw["resolution_candidates"])
    config = AnalysisConfig(**raw)
    if config.users < 100:
        raise ValueError("Synthetic graph must contain at least 100 users")
    if config.train_window_days < 1 or config.test_window_days < 1:
        raise ValueError("Temporal windows must contain at least one day")
    if not config.resolution_candidates or any(
        value <= 0 for value in config.resolution_candidates
    ):
        raise ValueError("Resolution candidates must be positive")
    if len(set(config.resolution_candidates)) != len(config.resolution_candidates):
        raise ValueError("Resolution candidates must be unique")
    if len(config.louvain_seeds) < 2:
        raise ValueError("Louvain settings are invalid")
    if config.null_permutations < 10:
        raise ValueError("At least ten null permutations are required")
    if config.edge_weighting not in {"raw", "log_tfidf"}:
        raise ValueError("edge_weighting must be raw or log_tfidf")
    positive_integer_fields = (
        config.min_component_users,
        config.min_component_categories,
        config.min_community_users,
        config.min_community_categories,
    )
    if any(value < 1 for value in positive_integer_fields):
        raise ValueError("Graph size guardrails must be positive integers")
    fractions = (
        config.max_small_component_node_share,
        config.min_eligible_user_coverage,
        config.max_category_weight_share,
        config.min_hub_removal_user_ari,
        config.min_temporal_user_ari,
    )
    if any(not 0 <= value <= 1 for value in fractions):
        raise ValueError("Graph guardrail fractions must be between zero and one")
    return config
