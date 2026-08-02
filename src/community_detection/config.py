"""Configuration loading for the community-detection pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "analysis.json"


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int
    users: int
    holdout_fraction: float
    louvain_resolution: float
    louvain_seeds: tuple[int, ...]
    null_permutations: int


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AnalysisConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["louvain_seeds"] = tuple(int(seed) for seed in raw["louvain_seeds"])
    config = AnalysisConfig(**raw)
    if config.users < 100:
        raise ValueError("Synthetic graph must contain at least 100 users")
    if not 0 < config.holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between zero and 0.5")
    if config.louvain_resolution <= 0 or len(config.louvain_seeds) < 2:
        raise ValueError("Louvain settings are invalid")
    return config
