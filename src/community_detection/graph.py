"""Bipartite graph construction and Louvain community detection."""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_bipartite_graph(interactions: pd.DataFrame) -> nx.Graph:
    """Build a weighted user-category graph from training interaction counts."""
    graph = nx.Graph()
    for row in interactions.itertuples(index=False):
        graph.add_node(row.user_id, node_type="user", bipartite=0)
        graph.add_node(
            row.category_id,
            node_type="category",
            category_name=row.category_name,
            category_family=row.category_family,
            bipartite=1,
        )
        if row.train_weight > 0:
            graph.add_edge(row.user_id, row.category_id, weight=float(row.train_weight))
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("Graph must contain nodes and weighted edges")
    return graph


def build_edge_list_graph(edges: pd.DataFrame) -> nx.Graph:
    """Build a weighted bipartite graph from the external edge-list contract."""
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_node(row.user_id, node_type="user", bipartite=0)
        graph.add_node(
            row.category_id,
            node_type="category",
            category_name=row.category_id,
            category_family="Input edge list",
            bipartite=1,
        )
        graph.add_edge(row.user_id, row.category_id, weight=float(row.weight))
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("Graph must contain nodes and weighted edges")
    return graph


def detect_communities(
    graph: nx.Graph,
    seed: int,
    resolution: float = 1.0,
) -> tuple[pd.DataFrame, list[set[str]]]:
    """Detect and deterministically relabel weighted Louvain communities."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    raw_communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=resolution, seed=seed
    )
    ranked = sorted(
        (set(map(str, community)) for community in raw_communities),
        key=lambda nodes: (
            -sum(graph.nodes[node]["node_type"] == "user" for node in nodes),
            min(nodes),
        ),
    )
    records: list[dict[str, object]] = []
    for community_id, nodes in enumerate(ranked):
        for node in sorted(nodes):
            records.append(
                {
                    "node_id": node,
                    "node_type": graph.nodes[node]["node_type"],
                    "community": community_id,
                    "weighted_degree": float(graph.degree(node, weight="weight")),
                }
            )
    assignments = pd.DataFrame.from_records(records).sort_values("node_id", ignore_index=True)
    return assignments, ranked
