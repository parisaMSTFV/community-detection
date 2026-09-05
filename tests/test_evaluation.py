import networkx as nx
import pandas as pd
import pytest

from community_detection.config import AnalysisConfig
from community_detection.evaluation import (
    align_assignment_labels,
    evaluate_hub_sensitivity,
    evaluate_seed_stability,
    graph_guardrails,
    modularity_score,
    recovery_metrics,
    temporal_edge_agreement,
    temporal_partition_stability,
)
from community_detection.graph import build_bipartite_graph, detect_communities
from community_detection.synthetic import generate_interactions, simulate_temporal_windows


def _case():
    interactions, truth = generate_interactions(users=500)
    split = simulate_temporal_windows(interactions)
    graph = build_bipartite_graph(split)
    assignments, communities = detect_communities(graph, seed=11)
    return split, truth, graph, assignments, communities


def test_detected_partition_recovers_planted_structure() -> None:
    _, truth, graph, assignments, communities = _case()
    metrics = recovery_metrics(assignments, truth)
    assert metrics["user_ari"] >= 0.80
    assert metrics["category_ari"] >= 0.80
    assert modularity_score(graph, communities) > 0.30


def test_detected_partition_is_stable_across_seeds() -> None:
    _, _, graph, _, _ = _case()
    pairs, summary = evaluate_seed_stability(graph, seeds=(11, 23, 37), resolution=1.0)
    assert len(pairs) == 3
    assert summary["mean_pairwise_ari"] >= 0.80


def test_future_agreement_beats_shuffled_category_null() -> None:
    interactions, _, _, assignments, _ = _case()
    future = interactions.loc[
        interactions["test_weight"] > 0,
        ["user_id", "category_id", "test_weight"],
    ].rename(columns={"test_weight": "weight"})
    training = interactions.loc[
        interactions["train_weight"] > 0,
        ["user_id", "category_id", "train_weight"],
    ].rename(columns={"train_weight": "weight"})
    null_scores, summary = temporal_edge_agreement(
        future, assignments, training, null_permutations=20
    )
    assert len(null_scores) == 20
    assert summary["observed_future_agreement"] > summary["null_mean_future_agreement"]
    assert summary["agreement_lift"] > 2


def test_tiny_disconnected_components_require_review() -> None:
    interactions, _, graph, assignments, communities = _case()
    del interactions
    _, _, summary = graph_guardrails(graph, assignments, communities, AnalysisConfig())
    assert summary["status"] == "pass"

    tiny = nx.Graph()
    for index in range(4):
        user = f"user::{index}"
        category = f"category::{index}"
        tiny.add_node(user, node_type="user", source_id=str(index), bipartite=0)
        tiny.add_node(category, node_type="category", source_id=str(index), bipartite=1)
        tiny.add_edge(user, category, weight=1.0, raw_weight=1.0)
    tiny_assignments, tiny_communities = detect_communities(tiny, seed=11)
    _, _, tiny_summary = graph_guardrails(
        tiny, tiny_assignments, tiny_communities, AnalysisConfig()
    )
    assert tiny_summary["status"] == "review_required"
    assert "small_component_node_share" in tiny_summary["reasons"]
    assert tiny_summary["eligible_user_coverage"] == 0


def test_temporal_partition_stability_reports_common_users() -> None:
    interactions, _, graph, assignments, _ = _case()
    future_graph = build_bipartite_graph(interactions, weight_column="test_weight")
    summary = temporal_partition_stability(assignments, future_graph, seed=11, resolution=1.0)
    assert summary["common_users"] / interactions["user_id"].nunique() > 0.95
    assert summary["common_categories"] == interactions["category_id"].nunique()
    assert -1 <= summary["user_ari"] <= 1


def test_hub_sensitivity_handles_single_category_graph() -> None:
    graph = nx.Graph()
    graph.add_node("user::1", node_type="user", source_id="1")
    graph.add_node("user::2", node_type="user", source_id="2")
    graph.add_node("category::1", node_type="category", source_id="1")
    graph.add_edge("user::1", "category::1", weight=1.0, raw_weight=1.0)
    graph.add_edge("user::2", "category::1", weight=1.0, raw_weight=1.0)
    assignments, _ = detect_communities(graph, seed=11)
    result = evaluate_hub_sensitivity(graph, assignments, seed=11, resolution=1.0)
    assert result["user_ari_after_removal"] is None


def test_assignment_labels_align_to_reference_and_allocate_new_labels() -> None:
    current = pd.DataFrame(
        {
            "node_id": ["U1", "U2", "U3"],
            "node_type": ["user", "user", "user"],
            "community": [0, 0, 1],
        }
    )
    reference = pd.DataFrame(
        {
            "node_id": ["U1", "U2", "OLD"],
            "node_type": ["user", "user", "user"],
            "community": [7, 7, 8],
        }
    )
    aligned, mapping, summary = align_assignment_labels(current, reference)
    assert set(aligned.loc[aligned["node_id"].isin(["U1", "U2"]), "community"]) == {7}
    assert mapping["aligned_community"].nunique() == 2
    assert summary["overlapping_nodes"] == 2
    assert summary["assignment_change_rate"] == 0


def test_recovery_rejects_missing_assignment() -> None:
    _, truth, _, assignments, _ = _case()
    with pytest.raises(ValueError, match="Every truth node"):
        recovery_metrics(assignments.iloc[:-1], truth)
