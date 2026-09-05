import pandas as pd

from community_detection.graph import build_bipartite_graph, detect_communities
from community_detection.synthetic import generate_interactions, simulate_temporal_windows


def test_graph_contains_only_user_category_edges() -> None:
    interactions, _ = generate_interactions(users=200)
    graph = build_bipartite_graph(simulate_temporal_windows(interactions))
    assert all(
        graph.nodes[left]["node_type"] != graph.nodes[right]["node_type"]
        for left, right in graph.edges
    )


def test_detection_assigns_every_node_once() -> None:
    interactions, _ = generate_interactions(users=200)
    graph = build_bipartite_graph(simulate_temporal_windows(interactions))
    assignments, communities = detect_communities(graph, seed=11)
    assert assignments["node_id"].nunique() == graph.number_of_nodes()
    assert len(assignments) == graph.number_of_nodes()
    assert len(communities) >= 2


def test_log_tfidf_reduces_a_dominant_category_hub_weight_share() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "user_id": f"U{index}",
                "category_id": f"C{index % 2}",
                "category_name": f"C{index % 2}",
                "category_family": "local",
                "train_weight": 5,
            }
        )
        rows.append(
            {
                "user_id": f"U{index}",
                "category_id": "HUB",
                "category_name": "HUB",
                "category_family": "global",
                "train_weight": 100,
            }
        )
    interactions = pd.DataFrame(rows)
    raw = build_bipartite_graph(interactions, weighting="raw")
    guarded = build_bipartite_graph(interactions, weighting="log_tfidf")
    raw_share = raw.degree("category::HUB", weight="weight") / raw.size(weight="weight")
    guarded_share = guarded.degree("category::HUB", weight="weight") / guarded.size(weight="weight")
    assert guarded_share < raw_share
