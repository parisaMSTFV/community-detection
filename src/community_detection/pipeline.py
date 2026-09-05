"""End-to-end community-detection pipelines and audit artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from community_detection import __version__
from community_detection.config import AnalysisConfig, load_config
from community_detection.evaluation import (
    align_assignment_labels,
    evaluate_hub_sensitivity,
    evaluate_seed_stability,
    graph_guardrails,
    modularity_score,
    recovery_metrics,
    select_resolution,
    temporal_edge_agreement,
    temporal_partition_stability,
)
from community_detection.graph import (
    build_bipartite_graph,
    build_edge_list_graph,
    detect_communities,
)
from community_detection.profiles import build_community_profiles, build_edge_list_profiles
from community_detection.reporting import (
    plot_category_projection,
    plot_community_sizes,
    plot_evaluation_summary,
    plot_family_profiles,
)
from community_detection.schema import (
    DataValidationError,
    load_reference_assignments,
    load_weighted_edge_list,
    pseudonymize_identifiers,
    validate_inputs,
)
from community_detection.synthetic import generate_interactions, simulate_temporal_windows

MODEL_SCHEMA_VERSION = "2.0"


def _fingerprint(frames: list[pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.to_csv(index=False, float_format="%.12g").encode("utf-8"))
    return digest.hexdigest()[:16]


def _config_fingerprint(config: AnalysisConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _dependency_versions() -> dict[str, str]:
    return {
        "networkx": version("networkx"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _combined_gate(
    graph_gate: dict[str, Any],
    hub: dict[str, Any],
    temporal: dict[str, Any] | None,
    config: AnalysisConfig,
) -> dict[str, Any]:
    reasons = list(graph_gate["reasons"])
    hub_ari = hub.get("user_ari_after_removal")
    if hub_ari is not None and hub_ari < config.min_hub_removal_user_ari:
        reasons.append("hub_removal_instability")
    if temporal is None:
        reasons.append("temporal_validation_missing")
    else:
        temporal_ari = temporal.get("user_ari")
        if temporal_ari is None or temporal_ari < config.min_temporal_user_ari:
            reasons.append("temporal_partition_instability")
    return {
        **graph_gate,
        "status": "pass" if not reasons else "review_required",
        "reasons": reasons,
        "thresholds": {
            **graph_gate["thresholds"],
            "min_hub_removal_user_ari": config.min_hub_removal_user_ari,
            "min_temporal_user_ari": config.min_temporal_user_ari,
        },
    }


def _manifest(
    config: AnalysisConfig,
    snapshot_id: str,
    selected_resolution: float,
    source_fingerprint: str,
    identifier_policy: str,
) -> dict[str, Any]:
    return {
        "model_version": __version__,
        "schema_version": MODEL_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "source_fingerprint": source_fingerprint,
        "config_fingerprint": _config_fingerprint(config),
        "selected_resolution": selected_resolution,
        "edge_weighting": config.edge_weighting,
        "identifier_policy": identifier_policy,
        "dependency_versions": _dependency_versions(),
    }


def _with_model_metadata(assignments: pd.DataFrame, snapshot_id: str) -> pd.DataFrame:
    output = assignments.copy()
    output["snapshot_id"] = snapshot_id
    output["model_version"] = __version__
    return output


def run_pipeline(
    output_root: Path | None = None,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Generate temporal synthetic data, detect communities, and audit the result."""
    config = config or load_config()
    output_root = output_root or Path.cwd()
    data_dir = output_root / "data"
    reports_dir = output_root / "reports"
    figures_dir = reports_dir / "figures"
    for path in (data_dir, reports_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    base_interactions, truth = generate_interactions(seed=config.seed, users=config.users)
    interactions = simulate_temporal_windows(
        base_interactions,
        train_window_days=config.train_window_days,
        test_window_days=config.test_window_days,
        seed=config.seed + 1,
    )
    validation = validate_inputs(interactions, truth)
    graph = build_bipartite_graph(interactions, weighting=config.edge_weighting)
    selected_resolution, resolution_search = select_resolution(graph, config)
    assignments, communities = detect_communities(
        graph,
        seed=config.louvain_seeds[0],
        resolution=selected_resolution,
    )
    recovery = recovery_metrics(assignments, truth)
    stability_pairs, stability = evaluate_seed_stability(
        graph, seeds=config.louvain_seeds, resolution=selected_resolution
    )
    components, community_quality, graph_gate = graph_guardrails(
        graph, assignments, communities, config
    )
    hub = evaluate_hub_sensitivity(graph, assignments, config.louvain_seeds[0], selected_resolution)

    training_edges = interactions.loc[
        interactions["train_weight"] > 0,
        ["user_id", "category_id", "train_weight"],
    ].rename(columns={"train_weight": "weight"})
    future_edges = interactions.loc[
        interactions["test_weight"] > 0,
        ["user_id", "category_id", "test_weight"],
    ].rename(columns={"test_weight": "weight"})
    temporal_null, future_agreement = temporal_edge_agreement(
        future_edges,
        assignments,
        training_edges,
        null_permutations=config.null_permutations,
        seed=config.seed + 2,
    )
    future_graph = build_bipartite_graph(
        interactions,
        weighting=config.edge_weighting,
        weight_column="test_weight",
    )
    temporal_stability = temporal_partition_stability(
        assignments,
        future_graph,
        config.louvain_seeds[0],
        selected_resolution,
    )
    quality_gate = _combined_gate(graph_gate, hub, temporal_stability, config)
    profiles, family_matrix = build_community_profiles(interactions, assignments)

    source_fingerprint = _fingerprint([interactions, truth])
    snapshot_id = f"snap_{source_fingerprint}"
    output_assignments = _with_model_metadata(assignments, snapshot_id)
    artifact_fingerprint = _fingerprint(
        [
            interactions,
            truth,
            output_assignments,
            stability_pairs,
            resolution_search,
            profiles,
            family_matrix.reset_index(),
        ]
    )
    manifest = _manifest(
        config,
        snapshot_id,
        selected_resolution,
        source_fingerprint,
        identifier_policy="synthetic_public_ids",
    )

    interactions.to_csv(data_dir / "synthetic_interactions.csv", index=False)
    truth.to_csv(data_dir / "synthetic_ground_truth.csv", index=False)
    output_assignments.to_csv(reports_dir / "community_assignments.csv", index=False)
    profiles.to_csv(reports_dir / "community_profiles.csv", index=False)
    family_matrix.to_csv(reports_dir / "family_profile_matrix.csv")
    stability_pairs.to_csv(reports_dir / "stability_pairs.csv", index=False)
    temporal_null.to_csv(reports_dir / "temporal_null_distribution.csv", index=False)
    resolution_search.to_csv(reports_dir / "resolution_search.csv", index=False)
    components.to_csv(reports_dir / "component_diagnostics.csv", index=False)
    community_quality.to_csv(reports_dir / "community_quality.csv", index=False)
    _write_json(manifest, reports_dir / "model_manifest.json")

    metrics: dict[str, Any] = {
        "data": asdict(validation),
        "model": {
            "selected_resolution": selected_resolution,
            "edge_weighting": config.edge_weighting,
            "resolution_selection_boundary": (
                "Unsupervised stability, coverage, and modularity; planted labels are excluded."
            ),
        },
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "detected_communities": len(communities),
            "weighted_modularity": modularity_score(graph, communities),
        },
        "quality_gate": quality_gate,
        "recovery": recovery,
        "seed_stability": stability,
        "hub_sensitivity": hub,
        "temporal": {
            "future_edge_agreement": future_agreement,
            "partition_stability": temporal_stability,
            "train_window_days": config.train_window_days,
            "test_window_days": config.test_window_days,
        },
        "artifact_fingerprint": artifact_fingerprint,
        "audit": manifest,
        "evaluation_boundary": (
            "Controlled synthetic temporal windows; no production validity or business lift."
        ),
    }
    _write_json(metrics, reports_dir / "metrics.json")

    summary_scores = {
        **recovery,
        **stability,
        **future_agreement,
        "temporal_user_ari": temporal_stability["user_ari"],
    }
    plot_evaluation_summary(summary_scores, figures_dir / "evaluation_summary.png")
    plot_family_profiles(family_matrix, figures_dir / "community_profiles.png")
    plot_community_sizes(profiles, figures_dir / "community_sizes.png")
    plot_category_projection(interactions, assignments, figures_dir / "category_projection.png")
    _write_summary(metrics, reports_dir / "run_summary.md")
    return metrics


def run_edge_list_pipeline(
    edges_path: Path,
    output_root: Path,
    config: AnalysisConfig | None = None,
    *,
    future_edges_path: Path | None = None,
    reference_assignments_path: Path | None = None,
    identifier_policy: str = "pseudonymized",
    identifier_salt: str | None = None,
) -> dict[str, Any]:
    """Detect communities in an external edge list with guarded exports."""
    config = config or load_config()
    if identifier_policy not in {"pseudonymized", "raw"}:
        raise DataValidationError("identifier_policy must be pseudonymized or raw")
    if identifier_policy == "pseudonymized" and identifier_salt is None:
        raise DataValidationError(
            "A salt is required for pseudonymized exports; set COMMUNITY_DETECTION_ID_SALT"
        )

    reports_dir = output_root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    edges, validation = load_weighted_edge_list(edges_path)
    graph = build_edge_list_graph(edges, weighting=config.edge_weighting)
    selected_resolution, resolution_search = select_resolution(graph, config)
    assignments, communities = detect_communities(
        graph,
        seed=config.louvain_seeds[0],
        resolution=selected_resolution,
    )
    stability_pairs, stability = evaluate_seed_stability(
        graph,
        seeds=config.louvain_seeds,
        resolution=selected_resolution,
    )
    components, community_quality, graph_gate = graph_guardrails(
        graph, assignments, communities, config
    )
    hub = evaluate_hub_sensitivity(graph, assignments, config.louvain_seeds[0], selected_resolution)

    temporal: dict[str, Any] | None = None
    temporal_null: pd.DataFrame | None = None
    future_validation: dict[str, Any] | None = None
    if future_edges_path is not None:
        future_edges, future_summary = load_weighted_edge_list(future_edges_path)
        future_validation = asdict(future_summary)
        temporal_null, future_agreement = temporal_edge_agreement(
            future_edges,
            assignments,
            edges,
            null_permutations=config.null_permutations,
            seed=config.seed + 2,
        )
        future_graph = build_edge_list_graph(future_edges, weighting=config.edge_weighting)
        partition_stability = temporal_partition_stability(
            assignments,
            future_graph,
            config.louvain_seeds[0],
            selected_resolution,
        )
        temporal = {
            "future_edge_agreement": future_agreement,
            "partition_stability": partition_stability,
        }
    quality_gate = _combined_gate(
        graph_gate,
        hub,
        temporal["partition_stability"] if temporal else None,
        config,
    )

    if identifier_policy == "pseudonymized":
        output_edges, output_assignments = pseudonymize_identifiers(
            edges, assignments, identifier_salt or ""
        )
    else:
        output_edges, output_assignments = edges.copy(), assignments.copy()

    alignment: dict[str, Any] | None = None
    alignment_frame: pd.DataFrame | None = None
    if reference_assignments_path is not None:
        reference = load_reference_assignments(reference_assignments_path)
        output_assignments, alignment_frame, alignment = align_assignment_labels(
            output_assignments, reference
        )

    source_fingerprint = _fingerprint([edges])
    snapshot_id = f"snap_{source_fingerprint}"
    output_assignments = _with_model_metadata(output_assignments, snapshot_id)
    profiles = build_edge_list_profiles(output_edges, output_assignments)
    artifact_fingerprint = _fingerprint(
        [output_edges, output_assignments, stability_pairs, resolution_search, profiles]
    )
    manifest = _manifest(
        config,
        snapshot_id,
        selected_resolution,
        source_fingerprint,
        identifier_policy=identifier_policy,
    )

    output_assignments.to_csv(reports_dir / "community_assignments.csv", index=False)
    profiles.to_csv(reports_dir / "community_profiles.csv", index=False)
    stability_pairs.to_csv(reports_dir / "stability_pairs.csv", index=False)
    resolution_search.to_csv(reports_dir / "resolution_search.csv", index=False)
    components.to_csv(reports_dir / "component_diagnostics.csv", index=False)
    community_quality.to_csv(reports_dir / "community_quality.csv", index=False)
    if temporal_null is not None:
        temporal_null.to_csv(reports_dir / "temporal_null_distribution.csv", index=False)
    if alignment_frame is not None:
        alignment_frame.to_csv(reports_dir / "label_alignment.csv", index=False)
    _write_json(manifest, reports_dir / "model_manifest.json")
    plot_community_sizes(profiles, figures_dir / "community_sizes.png")

    metrics: dict[str, Any] = {
        "mode": "weighted_bipartite_edge_list",
        "data": asdict(validation),
        "future_data": future_validation,
        "model": {
            "selected_resolution": selected_resolution,
            "edge_weighting": config.edge_weighting,
            "resolution_selection_boundary": (
                "Unsupervised stability, coverage, and modularity; outcomes are excluded."
            ),
        },
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "detected_communities": len(communities),
            "weighted_modularity": modularity_score(graph, communities),
        },
        "quality_gate": quality_gate,
        "seed_stability": stability,
        "hub_sensitivity": hub,
        "temporal": temporal,
        "label_alignment": alignment,
        "artifact_fingerprint": artifact_fingerprint,
        "audit": manifest,
        "evaluation_boundary": (
            "Unsupervised diagnostics only; no ground truth or business outcome was supplied."
        ),
    }
    _write_json(metrics, reports_dir / "metrics.json")
    _write_edge_list_summary(metrics, reports_dir / "run_summary.md")
    return metrics


def _write_summary(metrics: dict[str, Any], output_path: Path) -> None:
    recovery = metrics["recovery"]
    stability = metrics["seed_stability"]
    temporal = metrics["temporal"]
    future = temporal["future_edge_agreement"]
    partition = temporal["partition_stability"]
    text = f"""# Reproduction summary

- Quality gate: {metrics["quality_gate"]["status"]}
- Selected resolution: {metrics["model"]["selected_resolution"]:.2f}
- User ARI against planted labels: {recovery["user_ari"]:.3f}
- Minimum user seed stability: {stability["minimum_user_ari"]:.3f}
- Future-window agreement: {future["observed_future_agreement"]:.3f}
- Future unseen-edge agreement: {future["unseen_edge_agreement"]:.3f}
- Train-to-future user ARI: {partition["user_ari"]:.3f}
- Artifact fingerprint: `{metrics["artifact_fingerprint"]}`

These values describe controlled synthetic temporal windows only. They do not estimate production
segmentation quality, campaign adoption, incrementality, or business impact.
"""
    output_path.write_text(text, encoding="utf-8")


def _write_edge_list_summary(metrics: dict[str, Any], output_path: Path) -> None:
    graph = metrics["graph"]
    stability = metrics["seed_stability"]
    reasons = ", ".join(metrics["quality_gate"]["reasons"]) or "none"
    text = f"""# Edge-list analysis summary

- Quality gate: {metrics["quality_gate"]["status"]}
- Review reasons: {reasons}
- Nodes: {graph["nodes"]}
- Edges: {graph["edges"]}
- Detected communities: {graph["detected_communities"]}
- Selected resolution: {metrics["model"]["selected_resolution"]:.2f}
- Weighted modularity: {graph["weighted_modularity"]:.3f}
- Minimum user seed stability: {stability["minimum_user_ari"]:.3f}
- Identifier policy: {metrics["audit"]["identifier_policy"]}
- Snapshot: `{metrics["audit"]["snapshot_id"]}`
- Artifact fingerprint: `{metrics["artifact_fingerprint"]}`

These are unsupervised graph diagnostics. A pass means the configured technical review floor was
met; it does not establish behavioral validity, adoption, future performance, or business impact.
"""
    output_path.write_text(text, encoding="utf-8")
