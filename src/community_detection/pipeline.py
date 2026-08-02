"""End-to-end community-detection pipeline and artifact generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from community_detection.config import PROJECT_ROOT, AnalysisConfig, load_config
from community_detection.evaluation import (
    evaluate_seed_stability,
    holdout_agreement,
    modularity_score,
    recovery_metrics,
)
from community_detection.graph import build_bipartite_graph, detect_communities
from community_detection.profiles import build_community_profiles
from community_detection.reporting import (
    plot_category_projection,
    plot_community_sizes,
    plot_evaluation_summary,
    plot_family_profiles,
)
from community_detection.schema import validate_inputs
from community_detection.synthetic import generate_interactions, split_interaction_weights


def _fingerprint(frames: list[pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.to_csv(index=False, float_format="%.12g").encode("utf-8"))
    return digest.hexdigest()[:16]


def run_pipeline(
    output_root: Path = PROJECT_ROOT,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Generate data, detect graph communities, evaluate, and report."""
    config = config or load_config()
    data_dir = output_root / "data"
    reports_dir = output_root / "reports"
    figures_dir = reports_dir / "figures"
    for path in (data_dir, reports_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    interactions, truth = generate_interactions(seed=config.seed, users=config.users)
    interactions = split_interaction_weights(
        interactions,
        holdout_fraction=config.holdout_fraction,
        seed=config.seed + 1,
    )
    validation = validate_inputs(interactions, truth)
    graph = build_bipartite_graph(interactions)
    assignments, communities = detect_communities(
        graph,
        seed=config.louvain_seeds[0],
        resolution=config.louvain_resolution,
    )
    recovery = recovery_metrics(assignments, truth)
    stability_pairs, stability = evaluate_seed_stability(
        graph, seeds=config.louvain_seeds, resolution=config.louvain_resolution
    )
    null_distribution, holdout = holdout_agreement(
        interactions,
        assignments,
        null_permutations=config.null_permutations,
        seed=config.seed + 2,
    )
    profiles, family_matrix = build_community_profiles(interactions, assignments)
    fingerprint = _fingerprint(
        [interactions, truth, assignments, stability_pairs, profiles, family_matrix.reset_index()]
    )

    interactions.to_csv(data_dir / "synthetic_interactions.csv", index=False)
    truth.to_csv(data_dir / "synthetic_ground_truth.csv", index=False)
    assignments.to_csv(reports_dir / "community_assignments.csv", index=False)
    profiles.to_csv(reports_dir / "community_profiles.csv", index=False)
    family_matrix.to_csv(reports_dir / "family_profile_matrix.csv")
    stability_pairs.to_csv(reports_dir / "stability_pairs.csv", index=False)
    null_distribution.to_csv(reports_dir / "holdout_null_distribution.csv", index=False)

    metrics: dict[str, Any] = {
        "data": asdict(validation),
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "detected_communities": len(communities),
            "weighted_modularity": modularity_score(graph, communities),
        },
        "recovery": recovery,
        "stability": stability,
        "holdout": holdout,
        "artifact_fingerprint": fingerprint,
        "evaluation_boundary": "Planted synthetic communities only",
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary_scores = {**recovery, **stability, **holdout}
    plot_evaluation_summary(summary_scores, figures_dir / "evaluation_summary.png")
    plot_family_profiles(family_matrix, figures_dir / "community_profiles.png")
    plot_community_sizes(profiles, figures_dir / "community_sizes.png")
    plot_category_projection(interactions, assignments, figures_dir / "category_projection.png")
    _write_summary(metrics, reports_dir / "run_summary.md")
    return metrics


def _write_summary(metrics: dict[str, Any], output_path: Path) -> None:
    recovery = metrics["recovery"]
    stability = metrics["stability"]
    holdout = metrics["holdout"]
    text = f"""# Reproduction summary

- User ARI against planted labels: {recovery["user_ari"]:.3f}
- Category ARI against planted labels: {recovery["category_ari"]:.3f}
- Mean pairwise seed stability: {stability["mean_pairwise_ari"]:.3f}
- Held-out interaction agreement: {holdout["observed_holdout_agreement"]:.3f}
- Shuffled-category null agreement: {holdout["null_mean_holdout_agreement"]:.3f}
- Artifact fingerprint: `{metrics["artifact_fingerprint"]}`

These values describe the planted synthetic graph only and do not estimate production segmentation quality or campaign impact.
"""
    output_path.write_text(text, encoding="utf-8")
