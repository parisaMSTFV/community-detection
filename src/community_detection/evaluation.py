"""Recovery, stability, temporal, hub, and graph-quality evaluation."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from community_detection.config import AnalysisConfig
from community_detection.graph import detect_communities


def recovery_metrics(assignments: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Measure detected-label recovery without using truth during detection."""
    merged = truth.merge(
        assignments[["node_id", "node_type", "community"]],
        on=["node_id", "node_type"],
        how="left",
    )
    if merged["community"].isna().any():
        raise ValueError("Every truth node must have a detected assignment")
    metrics: dict[str, float] = {}
    for node_type in ("user", "category"):
        part = merged[merged["node_type"] == node_type]
        metrics[f"{node_type}_ari"] = float(
            adjusted_rand_score(part["planted_community"], part["community"])
        )
    metrics["overall_ari"] = float(
        adjusted_rand_score(merged["planted_community"], merged["community"])
    )
    return metrics


def modularity_score(graph: nx.Graph, communities: list[set[str]]) -> float:
    """Calculate weighted modularity of a detected partition."""
    return float(nx.community.modularity(graph, communities, weight="weight"))


def evaluate_seed_stability(
    graph: nx.Graph,
    seeds: tuple[int, ...],
    resolution: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare all Louvain seed pairs with label-invariant ARI."""
    assignments_by_seed: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        assignments, _ = detect_communities(graph, seed=seed, resolution=resolution)
        assignments_by_seed[seed] = assignments[["node_id", "community", "node_type"]]

    rows: list[dict[str, object]] = []
    for seed_a, seed_b in combinations(seeds, 2):
        merged = assignments_by_seed[seed_a].merge(
            assignments_by_seed[seed_b],
            on=["node_id", "node_type"],
            suffixes=("_a", "_b"),
        )
        rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "overall_ari": adjusted_rand_score(merged["community_a"], merged["community_b"]),
                "user_ari": adjusted_rand_score(
                    merged.loc[merged["node_type"] == "user", "community_a"],
                    merged.loc[merged["node_type"] == "user", "community_b"],
                ),
                "category_ari": adjusted_rand_score(
                    merged.loc[merged["node_type"] == "category", "community_a"],
                    merged.loc[merged["node_type"] == "category", "community_b"],
                ),
            }
        )
    pairwise = pd.DataFrame.from_records(rows)
    summary = {
        "mean_pairwise_ari": float(pairwise["overall_ari"].mean()),
        "minimum_pairwise_ari": float(pairwise["overall_ari"].min()),
        "mean_user_ari": float(pairwise["user_ari"].mean()),
        "minimum_user_ari": float(pairwise["user_ari"].min()),
        "mean_category_ari": float(pairwise["category_ari"].mean()),
    }
    return pairwise, summary


def graph_guardrails(
    graph: nx.Graph,
    assignments: pd.DataFrame,
    communities: list[set[str]],
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Diagnose component explosion, tiny communities, coverage, and category hubs."""
    component_rows: list[dict[str, Any]] = []
    for component_id, nodes in enumerate(
        sorted(nx.connected_components(graph), key=lambda values: (-len(values), min(values)))
    ):
        users = sum(graph.nodes[node]["node_type"] == "user" for node in nodes)
        categories = len(nodes) - users
        component_rows.append(
            {
                "component": component_id,
                "nodes": len(nodes),
                "users": users,
                "categories": categories,
                "raw_weight": float(graph.subgraph(nodes).size(weight="raw_weight")),
                "is_small": users < config.min_component_users
                or categories < config.min_component_categories,
            }
        )
    components = pd.DataFrame.from_records(component_rows)

    counts = (
        assignments.groupby(["community", "node_type"])["node_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=["user", "category"], fill_value=0)
        .rename(columns={"user": "users", "category": "categories"})
        .reset_index()
    )
    counts["eligible"] = (counts["users"] >= config.min_community_users) & (
        counts["categories"] >= config.min_community_categories
    )
    total_users = int((assignments["node_type"] == "user").sum())
    eligible_users = int(counts.loc[counts["eligible"], "users"].sum())
    eligible_coverage = eligible_users / total_users if total_users else 0.0

    small_nodes = int(components.loc[components["is_small"], "nodes"].sum())
    small_component_share = small_nodes / graph.number_of_nodes()
    largest_component_share = float(components["nodes"].max() / graph.number_of_nodes())

    category_strengths = [
        float(graph.degree(node, weight="raw_weight"))
        for node, attributes in graph.nodes(data=True)
        if attributes["node_type"] == "category"
    ]
    total_raw_weight = float(graph.size(weight="raw_weight"))
    max_category_share = max(category_strengths) / total_raw_weight if total_raw_weight else 0.0

    reasons: list[str] = []
    if small_component_share > config.max_small_component_node_share:
        reasons.append("small_component_node_share")
    if eligible_coverage < config.min_eligible_user_coverage:
        reasons.append("eligible_user_coverage")
    if max_category_share > config.max_category_weight_share:
        reasons.append("category_weight_concentration")
    summary: dict[str, Any] = {
        "status": "pass" if not reasons else "review_required",
        "reasons": reasons,
        "connected_components": int(len(components)),
        "largest_component_node_share": largest_component_share,
        "small_component_node_share": float(small_component_share),
        "eligible_user_coverage": float(eligible_coverage),
        "maximum_category_weight_share": float(max_category_share),
        "thresholds": {
            "max_small_component_node_share": config.max_small_component_node_share,
            "min_eligible_user_coverage": config.min_eligible_user_coverage,
            "max_category_weight_share": config.max_category_weight_share,
        },
    }
    return components, counts, summary


def select_resolution(
    graph: nx.Graph,
    config: AnalysisConfig,
) -> tuple[float, pd.DataFrame]:
    """Choose a diagnostic resolution without using planted labels or outcomes."""
    rows: list[dict[str, Any]] = []
    for resolution in config.resolution_candidates:
        assignments, communities = detect_communities(
            graph, seed=config.louvain_seeds[0], resolution=resolution
        )
        _, stability = evaluate_seed_stability(graph, config.louvain_seeds, resolution)
        _, _, guardrails = graph_guardrails(graph, assignments, communities, config)
        modularity = modularity_score(graph, communities)
        score = (
            2.0 * (guardrails["status"] == "pass")
            + stability["minimum_user_ari"]
            + guardrails["eligible_user_coverage"]
            + modularity
            - 0.002 * len(communities)
        )
        rows.append(
            {
                "resolution": resolution,
                "communities": len(communities),
                "weighted_modularity": modularity,
                "minimum_user_ari": stability["minimum_user_ari"],
                "eligible_user_coverage": guardrails["eligible_user_coverage"],
                "guardrail_status": guardrails["status"],
                "selection_score": score,
            }
        )
    search = pd.DataFrame.from_records(rows).sort_values("resolution", ignore_index=True)
    selected = search.sort_values(["selection_score", "resolution"], ascending=[False, True]).iloc[
        0
    ]
    return float(selected["resolution"]), search


def evaluate_hub_sensitivity(
    graph: nx.Graph,
    assignments: pd.DataFrame,
    seed: int,
    resolution: float,
) -> dict[str, Any]:
    """Remove the highest-weight category and compare the remaining user partition."""
    category_nodes = [
        node for node, values in graph.nodes(data=True) if values["node_type"] == "category"
    ]
    top_category = max(category_nodes, key=lambda node: graph.degree(node, weight="raw_weight"))
    top_share = float(
        graph.degree(top_category, weight="raw_weight") / graph.size(weight="raw_weight")
    )
    reduced = graph.copy()
    reduced.remove_node(top_category)
    reduced.remove_nodes_from(list(nx.isolates(reduced)))
    if reduced.number_of_edges() == 0:
        return {"top_category_weight_share": top_share, "user_ari_after_removal": None}
    reduced_assignments, _ = detect_communities(reduced, seed=seed, resolution=resolution)
    base_users = assignments[assignments["node_type"] == "user"]
    reduced_users = reduced_assignments[reduced_assignments["node_type"] == "user"]
    merged = base_users.merge(reduced_users, on=["node_id", "node_type"], suffixes=("_a", "_b"))
    ari = (
        float(adjusted_rand_score(merged["community_a"], merged["community_b"]))
        if len(merged) >= 2
        else None
    )
    return {
        "top_category_weight_share": top_share,
        "users_compared": int(len(merged)),
        "user_ari_after_removal": ari,
    }


def temporal_edge_agreement(
    future_edges: pd.DataFrame,
    assignments: pd.DataFrame,
    training_edges: pd.DataFrame,
    null_permutations: int = 100,
    seed: int = 99,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate later-window edges, including genuinely unseen user-category pairs."""
    if null_permutations < 10:
        raise ValueError("At least ten null permutations are required")
    users = assignments[assignments["node_type"] == "user"].rename(
        columns={"node_id": "user_id", "community": "user_community"}
    )[["user_id", "user_community"]]
    categories = assignments[assignments["node_type"] == "category"].rename(
        columns={"node_id": "category_id", "community": "category_community"}
    )[["category_id", "category_community"]]
    evaluated = future_edges.merge(users, on="user_id", how="left").merge(
        categories, on="category_id", how="left"
    )
    if evaluated.empty or evaluated["weight"].sum() <= 0:
        raise ValueError("Future window contains no positive event weight")
    evaluated["covered"] = evaluated[["user_community", "category_community"]].notna().all(axis=1)
    training_keys = set(zip(training_edges["user_id"], training_edges["category_id"], strict=True))
    evaluated["known_edge"] = [
        (user_id, category_id) in training_keys
        for user_id, category_id in zip(evaluated["user_id"], evaluated["category_id"], strict=True)
    ]
    covered = evaluated[evaluated["covered"]].copy()
    if covered.empty:
        raise ValueError("No future edge has both nodes represented in training")
    covered["same_community"] = covered["user_community"] == covered["category_community"]

    def agreement(frame: pd.DataFrame) -> float | None:
        total = float(frame["weight"].sum())
        if total == 0:
            return None
        return float(frame.loc[frame["same_community"], "weight"].sum() / total)

    observed = agreement(covered)
    rng = np.random.default_rng(seed)
    category_labels = categories["category_community"].to_numpy()
    null_scores: list[float] = []
    for _ in range(null_permutations):
        shuffled = categories[["category_id"]].copy()
        shuffled["shuffled_community"] = rng.permutation(category_labels)
        null_frame = covered.drop(columns="category_community").merge(
            shuffled, on="category_id", how="left"
        )
        match = null_frame["user_community"] == null_frame["shuffled_community"]
        null_scores.append(float(null_frame.loc[match, "weight"].sum() / covered["weight"].sum()))

    null_distribution = pd.DataFrame(
        {"permutation": np.arange(1, null_permutations + 1), "future_agreement": null_scores}
    )
    null_mean = float(np.mean(null_scores))
    total_weight = float(evaluated["weight"].sum())
    covered_weight = float(covered["weight"].sum())
    summary: dict[str, Any] = {
        "future_node_coverage_by_weight": covered_weight / total_weight,
        "observed_future_agreement": observed,
        "known_edge_agreement": agreement(covered[covered["known_edge"]]),
        "unseen_edge_agreement": agreement(covered[~covered["known_edge"]]),
        "unseen_edge_weight_share": float(
            covered.loc[~covered["known_edge"], "weight"].sum() / covered_weight
        ),
        "null_mean_future_agreement": null_mean,
        "agreement_lift": observed / null_mean if observed is not None and null_mean else None,
        "evaluated_future_weight": covered_weight,
    }
    return null_distribution, summary


def temporal_partition_stability(
    training_assignments: pd.DataFrame,
    future_graph: nx.Graph,
    seed: int,
    resolution: float,
) -> dict[str, Any]:
    """Compare training and future partitions on nodes present in both snapshots."""
    future_assignments, _ = detect_communities(future_graph, seed=seed, resolution=resolution)
    merged = training_assignments[["node_id", "node_type", "community"]].merge(
        future_assignments[["node_id", "node_type", "community"]],
        on=["node_id", "node_type"],
        suffixes=("_train", "_future"),
    )
    result: dict[str, Any] = {"common_nodes": int(len(merged))}
    for node_type in ("user", "category"):
        part = merged[merged["node_type"] == node_type]
        count_key = "common_users" if node_type == "user" else "common_categories"
        result[count_key] = int(len(part))
        result[f"{node_type}_ari"] = (
            float(adjusted_rand_score(part["community_train"], part["community_future"]))
            if len(part) >= 2
            else None
        )
    result["overall_ari"] = (
        float(adjusted_rand_score(merged["community_train"], merged["community_future"]))
        if len(merged) >= 2
        else None
    )
    return result


def align_assignment_labels(
    assignments: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Align numeric labels to an earlier snapshot by maximum typed-node overlap."""
    overlap = assignments[["node_id", "node_type", "community"]].merge(
        reference,
        on=["node_id", "node_type"],
        suffixes=("_current", "_reference"),
    )
    counts = (
        overlap.groupby(["community_current", "community_reference"])
        .size()
        .reset_index(name="overlap_nodes")
        .sort_values(
            ["overlap_nodes", "community_current", "community_reference"],
            ascending=[False, True, True],
        )
    )
    mapping: dict[int, int] = {}
    used_reference: set[int] = set()
    for row in counts.itertuples(index=False):
        current = int(row.community_current)
        previous = int(row.community_reference)
        if current not in mapping and previous not in used_reference:
            mapping[current] = previous
            used_reference.add(previous)
    next_label = int(reference["community"].max()) + 1 if not reference.empty else 0
    for current in sorted(assignments["community"].unique()):
        if int(current) not in mapping:
            while next_label in used_reference:
                next_label += 1
            mapping[int(current)] = next_label
            used_reference.add(next_label)
            next_label += 1

    aligned = assignments.copy()
    aligned["community"] = aligned["community"].map(mapping).astype(int)
    mapping_frame = pd.DataFrame(
        [
            {"detected_community": current, "aligned_community": aligned_label}
            for current, aligned_label in sorted(mapping.items())
        ]
    )
    post = aligned[["node_id", "node_type", "community"]].merge(
        reference,
        on=["node_id", "node_type"],
        suffixes=("_current", "_reference"),
    )
    summary = {
        "reference_nodes": int(len(reference)),
        "overlapping_nodes": int(len(post)),
        "overlap_share": float(len(post) / len(reference)) if len(reference) else 0.0,
        "assignment_change_rate": float(
            (post["community_current"] != post["community_reference"]).mean()
        )
        if len(post)
        else None,
    }
    return aligned.sort_values(["node_type", "node_id"], ignore_index=True), mapping_frame, summary
