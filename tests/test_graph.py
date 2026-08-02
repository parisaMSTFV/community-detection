from community_detection.graph import build_bipartite_graph, detect_communities
from community_detection.synthetic import generate_interactions, split_interaction_weights


def test_graph_contains_only_user_category_edges() -> None:
    interactions, _ = generate_interactions(users=200)
    graph = build_bipartite_graph(split_interaction_weights(interactions))
    assert all(
        graph.nodes[left]["node_type"] != graph.nodes[right]["node_type"]
        for left, right in graph.edges
    )


def test_detection_assigns_every_node_once() -> None:
    interactions, _ = generate_interactions(users=200)
    graph = build_bipartite_graph(split_interaction_weights(interactions))
    assignments, communities = detect_communities(graph, seed=11)
    assert assignments["node_id"].nunique() == graph.number_of_nodes()
    assert len(assignments) == graph.number_of_nodes()
    assert len(communities) >= 2
