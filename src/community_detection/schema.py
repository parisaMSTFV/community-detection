"""Input validation for synthetic interactions and planted labels."""

from __future__ import annotations

from dataclasses import dataclass

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
