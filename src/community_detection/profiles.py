"""Business-readable community profiles from detected assignments."""

from __future__ import annotations

import pandas as pd


def build_community_profiles(
    interactions: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize member counts, internal weight, and leading categories."""
    users = assignments[assignments["node_type"] == "user"].rename(
        columns={"node_id": "user_id", "community": "user_community"}
    )[["user_id", "user_community"]]
    categories = assignments[assignments["node_type"] == "category"].rename(
        columns={"node_id": "category_id", "community": "category_community"}
    )[["category_id", "category_community"]]
    enriched = interactions.merge(users, on="user_id", how="left").merge(
        categories, on="category_id", how="left"
    )
    enriched["is_internal"] = enriched["user_community"] == enriched["category_community"]

    user_counts = users.groupby("user_community")["user_id"].nunique()
    category_counts = categories.groupby("category_community")["category_id"].nunique()
    internal_weight = (
        enriched[enriched["is_internal"]].groupby("user_community")["train_weight"].sum()
    )
    total_weight = enriched.groupby("user_community")["train_weight"].sum()

    top_categories = (
        enriched[enriched["is_internal"]]
        .groupby(["user_community", "category_name"], as_index=False)["train_weight"]
        .sum()
        .sort_values(["user_community", "train_weight"], ascending=[True, False])
        .groupby("user_community")
        .head(3)
        .groupby("user_community")["category_name"]
        .agg(lambda values: " | ".join(values))
    )
    communities = sorted(assignments["community"].unique())
    profile = pd.DataFrame({"community": communities})
    profile["users"] = profile["community"].map(user_counts).fillna(0).astype(int)
    profile["categories"] = profile["community"].map(category_counts).fillna(0).astype(int)
    profile["train_interaction_weight"] = (
        profile["community"].map(total_weight).fillna(0).astype(int)
    )
    profile["internal_weight_share"] = profile["community"].map(
        (internal_weight / total_weight).fillna(0)
    )
    profile["top_categories"] = profile["community"].map(top_categories).fillna("")

    family_matrix = (
        enriched.groupby(["user_community", "category_family"])["train_weight"]
        .sum()
        .unstack(fill_value=0)
    )
    family_matrix = family_matrix.div(family_matrix.sum(axis=1), axis=0)
    family_matrix.index.name = "community"
    return profile, family_matrix
