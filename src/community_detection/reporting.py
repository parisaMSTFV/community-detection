"""Reproducible network, profile, and evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

COLORS = {
    "navy": "#17324D",
    "teal": "#4A9D8F",
    "gray": "#78808E",
    "amber": "#D69E3D",
    "ivory": "#F7F3EA",
}


def plot_evaluation_summary(metrics: dict[str, float], output_path: Path) -> None:
    """Plot recovery, stability, and holdout scores on a common zero-one scale."""
    labels = [
        "User ARI",
        "Category ARI",
        "Seed stability",
        "Holdout agreement",
        "Shuffled null",
    ]
    values = [
        metrics["user_ari"],
        metrics["category_ari"],
        metrics["mean_pairwise_ari"],
        metrics["observed_holdout_agreement"],
        metrics["null_mean_holdout_agreement"],
    ]
    colors = [COLORS["teal"]] * 4 + [COLORS["gray"]]
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    bars = axis.bar(labels, values, color=colors, width=0.68)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title("Synthetic community evaluation", loc="left", weight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", rotation=18)
    axis.bar_label(bars, fmt="%.2f", padding=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_family_profiles(family_matrix: pd.DataFrame, output_path: Path) -> None:
    """Plot normalized family affinity for each detected user community."""
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    image = axis.imshow(family_matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(family_matrix.columns)), family_matrix.columns, rotation=25)
    axis.set_yticks(
        np.arange(len(family_matrix.index)),
        [f"Community {value}" for value in family_matrix.index],
    )
    axis.set_title("Detected community interaction profile", loc="left", weight="bold")
    for row in range(family_matrix.shape[0]):
        for column in range(family_matrix.shape[1]):
            value = family_matrix.iloc[row, column]
            axis.text(
                column,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else COLORS["navy"],
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, label="Share of training interaction weight")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_community_sizes(profiles: pd.DataFrame, output_path: Path) -> None:
    """Plot user and category counts with separate axes."""
    labels = [f"C{value}" for value in profiles["community"]]
    positions = np.arange(len(labels))
    fig, user_axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    category_axis = user_axis.twinx()
    user_bars = user_axis.bar(
        positions - 0.18,
        profiles["users"],
        width=0.36,
        color=COLORS["teal"],
        label="Users",
    )
    category_bars = category_axis.bar(
        positions + 0.18,
        profiles["categories"],
        width=0.36,
        color=COLORS["amber"],
        label="Categories",
    )
    user_axis.set_xticks(positions, labels)
    user_axis.set_ylabel("Users", color=COLORS["teal"])
    category_axis.set_ylabel("Categories", color=COLORS["amber"])
    user_axis.set_title("Detected community composition", loc="left", weight="bold")
    user_axis.grid(axis="y", alpha=0.2)
    user_axis.spines["top"].set_visible(False)
    category_axis.spines["top"].set_visible(False)
    user_axis.bar_label(user_bars, padding=2)
    category_axis.bar_label(category_bars, padding=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_category_projection(
    interactions: pd.DataFrame,
    assignments: pd.DataFrame,
    output_path: Path,
    max_edges: int = 130,
) -> None:
    """Plot a category projection weighted by shared synthetic users."""
    incidence = pd.crosstab(interactions["user_id"], interactions["category_id"])
    incidence = incidence.gt(0).astype(int)
    co_users = incidence.T.dot(incidence)
    for category_id in co_users.columns:
        co_users.loc[category_id, category_id] = 0
    edge_rows: list[tuple[str, str, int]] = []
    categories = list(co_users.columns)
    for left_position, left in enumerate(categories):
        for right in categories[left_position + 1 :]:
            weight = int(co_users.loc[left, right])
            if weight > 0:
                edge_rows.append((left, right, weight))
    edge_rows = sorted(edge_rows, key=lambda item: item[2], reverse=True)[:max_edges]
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_rows)
    assignment_map = assignments.set_index("node_id")["community"].to_dict()
    name_map = (
        interactions[["category_id", "category_name"]]
        .drop_duplicates()
        .set_index("category_id")["category_name"]
        .to_dict()
    )
    position = nx.spring_layout(graph, seed=42, weight="weight", k=0.5, iterations=200)
    node_values = [assignment_map[node] for node in graph.nodes]
    node_sizes = [110 + 24 * graph.degree(node) for node in graph.nodes]
    edge_widths = [0.35 + graph.edges[edge]["weight"] / 50 for edge in graph.edges]
    fig, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    axis.set_facecolor("#FCFAF5")
    nx.draw_networkx_edges(graph, position, ax=axis, width=edge_widths, alpha=0.22)
    nx.draw_networkx_nodes(
        graph,
        position,
        ax=axis,
        node_color=node_values,
        cmap="tab10",
        node_size=node_sizes,
        edgecolors="white",
        linewidths=0.8,
    )
    leading: list[str] = []
    for community in sorted(set(node_values)):
        candidates = [node for node in graph.nodes if assignment_map[node] == community]
        leading.append(max(candidates, key=graph.degree))
    label_nodes = {node: name_map[node] for node in leading}
    nx.draw_networkx_labels(graph, position, labels=label_nodes, font_size=8, ax=axis)
    axis.set_title(
        "Category projection: shared-user topology",
        loc="left",
        weight="bold",
        color=COLORS["navy"],
    )
    axis.text(
        0,
        -0.03,
        f"Color = detected community · edge width = shared synthetic users · top {max_edges} edges",
        transform=axis.transAxes,
        color=COLORS["gray"],
    )
    axis.margins(0.15)
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
