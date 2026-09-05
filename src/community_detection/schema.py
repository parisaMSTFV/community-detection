"""Input validation and identifier-safety controls."""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MAX_INPUT_BYTES = 250 * 1024 * 1024
MAX_IDENTIFIER_LENGTH = 128
FORMULA_PREFIXES = ("=", "+", "-", "@")
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


class DataValidationError(ValueError):
    """Raised when an input dataset violates a required invariant."""


@dataclass(frozen=True)
class ValidationSummary:
    interaction_rows: int
    users: int
    categories: int
    total_weight: int
    future_only_edges: int
    checks_passed: int


@dataclass(frozen=True)
class EdgeListValidationSummary:
    """Validated shape of a user-category weighted edge list."""

    edge_rows: int
    users: int
    categories: int
    total_weight: float
    source_sha256: str
    checks_passed: int


def _validate_identifiers(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = frame[column]
        if values.str.len().gt(MAX_IDENTIFIER_LENGTH).any():
            raise DataValidationError(
                f"{column} values must not exceed {MAX_IDENTIFIER_LENGTH} characters"
            )
        if values.str.startswith(FORMULA_PREFIXES).any():
            raise DataValidationError(f"{column} contains a spreadsheet-formula prefix")
        if values.str.contains(CONTROL_CHARACTERS).any():
            raise DataValidationError(f"{column} contains a control character")


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
    if truth.duplicated(["node_id", "node_type"]).any():
        raise DataValidationError("Ground-truth typed node IDs must be unique")

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
    if (interactions["interaction_weight"] <= 0).any():
        raise DataValidationError("Every retained edge must contain at least one event")

    train = interactions[interactions["train_weight"] > 0]
    if train["user_id"].nunique() != interactions["user_id"].nunique():
        raise DataValidationError("Every synthetic user must occur in the training window")
    if train["category_id"].nunique() != interactions["category_id"].nunique():
        raise DataValidationError("Every synthetic category must occur in the training window")

    interaction_nodes = set(
        zip(interactions["user_id"], ["user"] * len(interactions), strict=True)
    ) | set(
        zip(
            interactions["category_id"],
            ["category"] * len(interactions),
            strict=True,
        )
    )
    truth_nodes = set(zip(truth["node_id"], truth["node_type"], strict=True))
    if interaction_nodes != truth_nodes:
        raise DataValidationError("Ground truth and interaction typed nodes must match")
    if not truth["node_type"].isin(["user", "category"]).all():
        raise DataValidationError("Unsupported node type")
    return ValidationSummary(
        interaction_rows=len(interactions),
        users=interactions["user_id"].nunique(),
        categories=interactions["category_id"].nunique(),
        total_weight=int(interactions["interaction_weight"].sum()),
        future_only_edges=int(
            ((interactions["train_weight"] == 0) & (interactions["test_weight"] > 0)).sum()
        ),
        checks_passed=11,
    )


def _read_string_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataValidationError(f"CSV file does not exist: {path}")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise DataValidationError("CSV exceeds the 250 MB input limit")
    try:
        return pd.read_csv(path, dtype="string", keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DataValidationError("Input must be a readable UTF-8 CSV with a header") from error


def load_weighted_edge_list(path: Path) -> tuple[pd.DataFrame, EdgeListValidationSummary]:
    """Load a minimal external edge list without coercing identifiers."""
    edges = _read_string_csv(path)
    required = ["user_id", "category_id", "weight"]
    missing = set(required).difference(edges.columns)
    if missing:
        raise DataValidationError(f"Missing edge-list columns: {sorted(missing)}")
    if edges.empty:
        raise DataValidationError("Edge list must not be empty")

    canonical = edges[required].copy()
    canonical["user_id"] = canonical["user_id"].str.strip()
    canonical["category_id"] = canonical["category_id"].str.strip()
    if canonical[["user_id", "category_id"]].eq("").any().any():
        raise DataValidationError("Node identifiers must not be blank")
    _validate_identifiers(canonical, ("user_id", "category_id"))
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
    if len(users) < 2 or len(categories) < 2:
        raise DataValidationError("Edge list must contain at least two users and two categories")

    canonical = canonical.sort_values(["user_id", "category_id"], ignore_index=True)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = EdgeListValidationSummary(
        edge_rows=len(canonical),
        users=len(users),
        categories=len(categories),
        total_weight=float(canonical["weight"].sum()),
        source_sha256=source_sha256,
        checks_passed=10,
    )
    return canonical, summary


def pseudonymize_identifiers(
    edges: pd.DataFrame,
    assignments: pd.DataFrame,
    salt: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic, namespace-aware HMAC identifiers for exported artifacts."""
    if len(salt) < 16:
        raise DataValidationError("Identifier salt must contain at least 16 characters")

    def token(node_type: str, value: str) -> str:
        digest = hmac.new(
            salt.encode("utf-8"),
            f"{node_type}:{value}".encode(),
            hashlib.sha256,
        ).hexdigest()[:20]
        prefix = "usr" if node_type == "user" else "cat"
        return f"{prefix}_{digest}"

    safe_edges = edges.copy()
    safe_edges["user_id"] = safe_edges["user_id"].map(lambda value: token("user", str(value)))
    safe_edges["category_id"] = safe_edges["category_id"].map(
        lambda value: token("category", str(value))
    )
    safe_assignments = assignments.copy()
    safe_assignments["node_id"] = safe_assignments.apply(
        lambda row: token(str(row["node_type"]), str(row["node_id"])), axis=1
    )
    return safe_edges, safe_assignments


def load_reference_assignments(path: Path) -> pd.DataFrame:
    """Validate an earlier assignment artifact used only for label alignment."""
    frame = _read_string_csv(path)
    required = ["node_id", "node_type", "community"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise DataValidationError(f"Missing reference columns: {sorted(missing)}")
    reference = frame[required].copy()
    if reference.empty or reference[required].eq("").any().any():
        raise DataValidationError("Reference assignments must not contain blank fields")
    if not reference["node_type"].isin(["user", "category"]).all():
        raise DataValidationError("Reference assignments contain an invalid node type")
    _validate_identifiers(reference, ("node_id",))
    try:
        reference["community"] = pd.to_numeric(
            reference["community"], errors="raise", downcast="integer"
        )
    except (TypeError, ValueError) as error:
        raise DataValidationError("Reference community labels must be integers") from error
    if reference.duplicated(["node_id", "node_type"]).any():
        raise DataValidationError("Reference typed node IDs must be unique")
    return reference.sort_values(["node_type", "node_id"], ignore_index=True)
