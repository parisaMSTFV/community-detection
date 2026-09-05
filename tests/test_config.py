import json
from dataclasses import asdict
from pathlib import Path

import pytest

from community_detection.config import AnalysisConfig, load_config


def _write_config(tmp_path: Path, **updates) -> Path:
    payload = asdict(AnalysisConfig())
    payload.update(updates)
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_bundled_config_loads() -> None:
    config = load_config()
    assert config.edge_weighting == "log_tfidf"
    assert config.resolution_candidates == (0.8, 1.0, 1.2)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"users": 99}, "at least 100"),
        ({"train_window_days": 0}, "Temporal windows"),
        ({"resolution_candidates": []}, "Resolution candidates"),
        ({"resolution_candidates": [1.0, 1.0]}, "must be unique"),
        ({"louvain_seeds": [11]}, "Louvain settings"),
        ({"null_permutations": 9}, "ten null"),
        ({"edge_weighting": "unsupported"}, "raw or log_tfidf"),
        ({"min_component_users": 0}, "positive integers"),
        ({"min_temporal_user_ari": 1.1}, "between zero and one"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, updates: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, **updates))
