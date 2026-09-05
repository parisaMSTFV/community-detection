"""Bipartite graph construction and weighted Louvain detection."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd


def _node_key(node_type: str, source_id: str) -> str:
    """Keep user and category namespaces separate inside NetworkX."""
    return f"{node_type}::{source_id}"


def _transformed_weights(
    edges: pd.DataFrame,
    weight_column: str,
    weighting: str,
) -> np.ndarray:
    raw = edges[weight_column].to_numpy(dtype=float)
    if weighting == "raw":
        return raw
    if weighting != "log_tfidf":
        raise ValueError("weighting must be raw or log_tfidf")

    user_count = edges["user_id"].nunique()
    category_degree = edges.groupby("category_id")["user_id"].nunique()
    inverse_popularity = edges["category_id"].map(
        lambda value: math.log((1 + user_count) / (1 + category_degree[value])) + 1
    )
    return np.log1p(raw) * inverse_popularity.to_numpy(dtype=float)


def _build_graph(
    edges: pd.DataFrame,
    weight_column: str,
    weighting: str,
    category_metadata: bool,
) -> nx.Graph:
    positive = edges.loc[edges[weight_column] > 0].copy()
    if positive.empty:
        raise ValueError("Graph must contain positive weighted edges")
    positive["model_weight"] = _transformed_weights(positive, weight_column, weighting)

    graph = nx.Graph(weighting=weighting)
    for row in positive.itertuples(index=False):
        user_key = _node_key("user", str(row.user_id))
        category_key = _node_key("category", str(row.category_id))
        graph.add_node(
            user_key,
            node_type="user",
            source_id=str(row.user_id),
            bipartite=0,
        )
        category_attributes = {
            "node_type": "category",
            "source_id": str(row.category_id),
            "bipartite": 1,
        }
        if category_metadata:
            category_attributes.update(
                category_name=str(row.category_name),
                category_family=str(row.category_family),
            )
        else:
            category_attributes.update(
                category_name=str(row.category_id),
                category_family="Input edge list",
            )
        graph.add_node(category_key, **category_attributes)
        graph.add_edge(
            user_key,
            category_key,
            weight=float(row.model_weight),
            raw_weight=float(getattr(row, weight_column)),
        )
    return graph


def build_bipartite_graph(
    interactions: pd.DataFrame,
    weighting: str = "log_tfidf",
    weight_column: str = "train_weight",
) -> nx.Graph:
    """Build a weighted user-category graph from one temporal window."""
    return _build_graph(
        interactions,
        weight_column=weight_column,
        weighting=weighting,
        category_metadata=True,
    )


def build_edge_list_graph(
    edges: pd.DataFrame,
    weighting: str = "log_tfidf",
) -> nx.Graph:
    """Build a weighted bipartite graph from the external edge-list contract."""
    return _build_graph(
        edges,
        weight_column="weight",
        weighting=weighting,
        category_metadata=False,
    )


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
        (set(community) for community in raw_communities),
        key=lambda nodes: (
            -sum(graph.nodes[node]["node_type"] == "user" for node in nodes),
            min(nodes),
        ),
    )
    records: list[dict[str, object]] = []
    for community_id, nodes in enumerate(ranked):
        for node in sorted(nodes):
            attributes = graph.nodes[node]
            records.append(
                {
                    "node_id": attributes["source_id"],
                    "node_type": attributes["node_type"],
                    "community": community_id,
                    "weighted_degree": float(graph.degree(node, weight="weight")),
                    "raw_weighted_degree": float(graph.degree(node, weight="raw_weight")),
                }
            )
    assignments = pd.DataFrame.from_records(records).sort_values(
        ["node_type", "node_id"], ignore_index=True
    )
    return assignments, ranked
