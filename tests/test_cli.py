import sys
from pathlib import Path

import pytest

import community_detection.cli as cli
from community_detection.config import AnalysisConfig


def _config() -> AnalysisConfig:
    return AnalysisConfig(users=100, louvain_seeds=(11, 23), null_permutations=10)


def test_reproduce_and_smoke_commands(monkeypatch, tmp_path: Path, capsys) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(cli, "load_config", lambda path: _config())
    monkeypatch.setattr(
        cli,
        "run_pipeline",
        lambda output_root, config: (
            calls.append(output_root) or {"quality_gate": {"status": "pass"}}
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["community-detection", "reproduce", "--output-root", str(tmp_path)],
    )
    cli.main()
    monkeypatch.setattr(sys, "argv", ["community-detection", "smoke"])
    cli.main()
    assert calls[0] == tmp_path
    assert "quality_gate" in capsys.readouterr().out


def test_analyze_command_passes_privacy_and_temporal_options(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    monkeypatch.setenv("COMMUNITY_DETECTION_ID_SALT", "test-only-salt-12345")
    monkeypatch.setattr(cli, "load_config", lambda path: _config())

    def fake_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return {"quality_gate": {"status": "pass"}}

    monkeypatch.setattr(cli, "run_edge_list_pipeline", fake_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "community-detection",
            "analyze",
            "--edges",
            str(tmp_path / "current.csv"),
            "--future-edges",
            str(tmp_path / "future.csv"),
            "--reference-assignments",
            str(tmp_path / "reference.csv"),
            "--output-root",
            str(tmp_path),
        ],
    )
    cli.main()
    assert captured["identifier_policy"] == "pseudonymized"
    assert captured["identifier_salt"] == "test-only-salt-12345"
    assert captured["future_edges_path"].name == "future.csv"


def test_fail_on_review_exits_with_status_two(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "load_config", lambda path: _config())
    monkeypatch.setattr(
        cli,
        "run_edge_list_pipeline",
        lambda *args, **kwargs: {"quality_gate": {"status": "review_required"}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "community-detection",
            "analyze",
            "--edges",
            str(tmp_path / "edges.csv"),
            "--output-root",
            str(tmp_path),
            "--allow-raw-identifiers",
            "--fail-on-review",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        cli.main()
