"""Command-line interface for the community-detection case study."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from community_detection.config import load_config
from community_detection.pipeline import run_edge_list_pipeline, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reproduce = subparsers.add_parser("reproduce", help="regenerate all artifacts")
    reproduce.add_argument("--output-root", type=Path, default=Path.cwd())
    reproduce.add_argument("--config", type=Path)
    analyze = subparsers.add_parser(
        "analyze", help="analyze a validated user-category weighted edge list"
    )
    analyze.add_argument("--edges", type=Path, required=True)
    analyze.add_argument("--future-edges", type=Path)
    analyze.add_argument("--reference-assignments", type=Path)
    analyze.add_argument("--output-root", type=Path, required=True)
    analyze.add_argument("--config", type=Path)
    analyze.add_argument(
        "--allow-raw-identifiers",
        action="store_true",
        help="export raw IDs only when the input is explicitly approved for that use",
    )
    analyze.add_argument(
        "--fail-on-review",
        action="store_true",
        help="exit with status 2 when a quality guardrail requires review",
    )
    smoke = subparsers.add_parser("smoke", help="run the pipeline in a temporary directory")
    smoke.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "analyze":
        policy = "raw" if args.allow_raw_identifiers else "pseudonymized"
        metrics = run_edge_list_pipeline(
            args.edges,
            args.output_root,
            config=config,
            future_edges_path=args.future_edges,
            reference_assignments_path=args.reference_assignments,
            identifier_policy=policy,
            identifier_salt=os.environ.get("COMMUNITY_DETECTION_ID_SALT"),
        )
    elif args.command == "smoke":
        with tempfile.TemporaryDirectory(prefix="community-detection-") as directory:
            metrics = run_pipeline(Path(directory), config=config)
    else:
        metrics = run_pipeline(args.output_root, config=config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.command == "analyze" and args.fail_on_review:
        if metrics["quality_gate"]["status"] != "pass":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
