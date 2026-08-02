"""Generate planted user-category interaction communities."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORY_FAMILIES = {
    "Digital": ["Smartphones", "Laptops", "Headphones", "Monitors", "Storage"],
    "Home": ["Kitchen appliances", "Cleaning", "Coffee", "Lighting", "Bedding"],
    "Style": ["Clothing", "Shoes", "Bags", "Accessories", "Jewelry"],
    "Wellness": ["Skincare", "Haircare", "Supplements", "Fitness", "Personal care"],
    "Family": ["Toys", "Baby care", "Books", "Stationery", "Board games"],
    "Outdoor": ["Cycling", "Camping", "Garden", "Automotive", "Sports"],
}


def generate_interactions(
    seed: int = 42,
    users: int = 1200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create weighted interactions and planted labels used only for evaluation."""
    if users < 100:
        raise ValueError("users must be at least 100")
    rng = np.random.default_rng(seed)
    families = list(CATEGORY_FAMILIES)
    categories: list[dict[str, object]] = []
    for family_id, family in enumerate(families):
        for name in CATEGORY_FAMILIES[family]:
            categories.append(
                {
                    "category_id": f"CAT-{len(categories) + 1:03d}",
                    "category_name": name,
                    "category_family": family,
                    "planted_community": family_id,
                }
            )
    category_frame = pd.DataFrame(categories)

    user_labels = rng.choice(len(families), size=users, p=[0.20, 0.18, 0.17, 0.16, 0.15, 0.14])
    rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    for position, planted in enumerate(user_labels, start=1):
        user_id = f"USR-{position:05d}"
        truth_rows.append(
            {"node_id": user_id, "node_type": "user", "planted_community": int(planted)}
        )
        home = category_frame[category_frame["planted_community"] == planted]
        home_count = int(rng.integers(3, 6))
        chosen_home = rng.choice(home.index.to_numpy(), size=home_count, replace=False)
        if rng.random() < 0.25:
            secondary = int(
                rng.choice([value for value in range(len(families)) if value != planted])
            )
            other = category_frame[category_frame["planted_community"] == secondary]
            cross_count = 3
            cross_lambda = 5
            cross_base = 2
        else:
            other = category_frame[category_frame["planted_community"] != planted]
            cross_count = int(rng.choice([0, 1, 2], p=[0.20, 0.55, 0.25]))
            cross_lambda = 2
            cross_base = 1
        chosen_cross = (
            rng.choice(other.index.to_numpy(), size=cross_count, replace=False)
            if cross_count
            else np.array([], dtype=int)
        )
        for category_index in chosen_home:
            category = category_frame.loc[category_index]
            rows.append(
                {
                    "user_id": user_id,
                    "category_id": category["category_id"],
                    "category_name": category["category_name"],
                    "category_family": category["category_family"],
                    "interaction_weight": int(rng.poisson(7) + 2),
                }
            )
        for category_index in chosen_cross:
            category = category_frame.loc[category_index]
            rows.append(
                {
                    "user_id": user_id,
                    "category_id": category["category_id"],
                    "category_name": category["category_name"],
                    "category_family": category["category_family"],
                    "interaction_weight": int(rng.poisson(cross_lambda) + cross_base),
                }
            )

    for row in category_frame.itertuples(index=False):
        truth_rows.append(
            {
                "node_id": row.category_id,
                "node_type": "category",
                "planted_community": int(row.planted_community),
            }
        )
    interactions = pd.DataFrame.from_records(rows).sort_values(
        ["user_id", "category_id"], ignore_index=True
    )
    truth = pd.DataFrame.from_records(truth_rows).sort_values("node_id", ignore_index=True)
    return interactions, truth


def split_interaction_weights(
    interactions: pd.DataFrame,
    holdout_fraction: float = 0.2,
    seed: int = 43,
) -> pd.DataFrame:
    """Split event counts per observed edge while keeping at least one train event."""
    if not 0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between zero and 0.5")
    rng = np.random.default_rng(seed)
    result = interactions.copy()
    total = result["interaction_weight"].astype(int).to_numpy()
    test = rng.binomial(total, holdout_fraction)
    test = np.minimum(test, np.maximum(total - 1, 0))
    result["train_weight"] = total - test
    result["test_weight"] = test
    return result
