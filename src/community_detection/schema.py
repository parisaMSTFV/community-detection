"""Input validation for synthetic interactions and planted labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when an interaction dataset violates a required invariant."""


@dataclass(frozen=True)
class ValidationSummary:
    interaction_rows: int
    users: int
    categories: int
    total_weight: int
    checks_passed: int


@dataclass(frozen=True)
class EdgeListValidationSummary:
    """Validated shape of a user-category weighted edge list."""

    edge_rows: int
    users: int
    categories: int
    total_weight: float
    checks_passed: int


def validate_inputs(interactions: pd.DataFrame, truth: pd.DataFrame) -> ValidationSummary:
    required_interactions = {
        "user_id",
        "category_id",
        "category_name",
        "category_family",
        "interaction_weight",
        "train_weight",
        "test_weight",
    }
    required_truth = {"node_id", "node_type", "planted_community"}
    missing = required_interactions.difference(interactions.columns)
    if missing:
        raise DataValidationError(f"Missing interaction columns: {sorted(missing)}")
    missing_truth = required_truth.difference(truth.columns)
    if missing_truth:
        raise DataValidationError(f"Missing truth columns: {sorted(missing_truth)}")
    if interactions.empty or truth.empty:
        raise DataValidationError("Inputs must not be empty")
    if interactions.duplicated(["user_id", "category_id"]).any():
        raise DataValidationError("Each user-category edge must be unique")
    if truth["node_id"].duplicated().any():
        raise DataValidationError("Ground-truth node IDs must be unique")
    weight_columns = ["interaction_weight", "train_weight", "test_weight"]
    if interactions[weight_columns].isna().any().any():
        raise DataValidationError("Weights must not contain nulls")
    if (interactions[weight_columns] < 0).any().any():
        raise DataValidationError("Weights must be non-negative")
    if not (
        interactions["train_weight"] + interactions["test_weight"]
        == interactions["interaction_weight"]
    ).all():
        raise DataValidationError("Train and test weights must reconcile to total weight")
    if (interactions["train_weight"] < 1).any():
        raise DataValidationError("Every observed edge must retain at least one train event")
    interaction_nodes = set(interactions["user_id"]) | set(interactions["category_id"])
    if interaction_nodes != set(truth["node_id"]):
        raise DataValidationError("Ground truth and interaction nodes must match")
    if not truth["node_type"].isin(["user", "category"]).all():
        raise DataValidationError("Unsupported node type")
    return ValidationSummary(
        interaction_rows=len(interactions),
        users=interactions["user_id"].nunique(),
        categories=interactions["category_id"].nunique(),
        total_weight=int(interactions["interaction_weight"].sum()),
        checks_passed=9,
    )


def load_weighted_edge_list(path: Path) -> tuple[pd.DataFrame, EdgeListValidationSummary]:
    """Load and validate a minimal external bipartite edge-list contract."""
    if not path.is_file():
        raise DataValidationError(f"Edge-list file does not exist: {path}")
    try:
        edges = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise DataValidationError("Edge-list file must be a readable CSV with a header") from error
    required = ["user_id", "category_id", "weight"]
    missing = set(required).difference(edges.columns)
    if missing:
        raise DataValidationError(f"Missing edge-list columns: {sorted(missing)}")
    if edges.empty:
        raise DataValidationError("Edge list must not be empty")

    canonical = edges[required].copy()
    if canonical.isna().any().any():
        raise DataValidationError("Edge-list fields must not contain nulls")
    canonical["user_id"] = canonical["user_id"].astype("string").str.strip()
    canonical["category_id"] = canonical["category_id"].astype("string").str.strip()
    if canonical[["user_id", "category_id"]].eq("").any().any():
        raise DataValidationError("Node identifiers must not be blank")
    try:
        canonical["weight"] = pd.to_numeric(canonical["weight"], errors="raise")
    except (TypeError, ValueError) as error:
        raise DataValidationError("Edge weights must be numeric") from error
    if not np.isfinite(canonical["weight"].to_numpy(dtype=float)).all():
        raise DataValidationError("Edge weights must be finite")
    if (canonical["weight"] <= 0).any():
        raise DataValidationError("Edge weights must be positive")
    if canonical.duplicated(["user_id", "category_id"]).any():
        raise DataValidationError("Each user-category edge must be unique")

    users = set(canonical["user_id"])
    categories = set(canonical["category_id"])
    if users.intersection(categories):
        raise DataValidationError("User and category identifiers must use disjoint namespaces")
    if len(users) < 2 or len(categories) < 2:
        raise DataValidationError("Edge list must contain at least two users and two categories")

    canonical = canonical.sort_values(["user_id", "category_id"], ignore_index=True)
    summary = EdgeListValidationSummary(
        edge_rows=len(canonical),
        users=len(users),
        categories=len(categories),
        total_weight=float(canonical["weight"].sum()),
        checks_passed=8,
    )
    return canonical, summary
