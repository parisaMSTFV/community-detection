"""Ground-truth, holdout, modularity, and seed-stability evaluation."""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from community_detection.graph import detect_communities


def recovery_metrics(assignments: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Measure detected-label recovery without using truth during detection."""
    merged = truth.merge(assignments[["node_id", "community"]], on="node_id", how="left")
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
        "mean_category_ari": float(pairwise["category_ari"].mean()),
    }
    return pairwise, summary


def holdout_agreement(
    interactions: pd.DataFrame,
    assignments: pd.DataFrame,
    null_permutations: int = 100,
    seed: int = 99,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare held-out within-community weight with a shuffled-category null."""
    if null_permutations < 10:
        raise ValueError("At least ten null permutations are required")
    user_assignment = assignments[assignments["node_type"] == "user"].rename(
        columns={"node_id": "user_id", "community": "user_community"}
    )[["user_id", "user_community"]]
    category_assignment = assignments[assignments["node_type"] == "category"].rename(
        columns={"node_id": "category_id", "community": "category_community"}
    )[["category_id", "category_community"]]
    evaluated = interactions.merge(user_assignment, on="user_id", how="left").merge(
        category_assignment, on="category_id", how="left"
    )
    evaluated = evaluated[evaluated["test_weight"] > 0].copy()
    if evaluated.empty:
        raise ValueError("Holdout contains no positive event weight")
    evaluated["same_community"] = evaluated["user_community"] == evaluated["category_community"]
    total_weight = float(evaluated["test_weight"].sum())
    observed = float(evaluated.loc[evaluated["same_community"], "test_weight"].sum() / total_weight)

    rng = np.random.default_rng(seed)
    category_labels = category_assignment["category_community"].to_numpy()
    null_scores: list[float] = []
    for _ in range(null_permutations):
        shuffled = category_assignment[["category_id"]].copy()
        shuffled["shuffled_community"] = rng.permutation(category_labels)
        null_frame = evaluated.drop(columns="category_community").merge(
            shuffled, on="category_id", how="left"
        )
        match = null_frame["user_community"] == null_frame["shuffled_community"]
        null_scores.append(float(null_frame.loc[match, "test_weight"].sum() / total_weight))

    null_frame = pd.DataFrame(
        {"permutation": np.arange(1, null_permutations + 1), "holdout_agreement": null_scores}
    )
    null_mean = float(np.mean(null_scores))
    summary = {
        "observed_holdout_agreement": observed,
        "null_mean_holdout_agreement": null_mean,
        "null_std_holdout_agreement": float(np.std(null_scores, ddof=0)),
        "agreement_lift": observed / null_mean if null_mean else float("inf"),
        "evaluated_test_weight": total_weight,
    }
    return null_frame, summary
