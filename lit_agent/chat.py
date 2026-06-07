"""Conversational command-line entry point for the LangGraph literature agent."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .agent_graph import run_literature_graph_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat-oriented legal literature LangGraph agent.")
    parser.add_argument("request", nargs="*", help="Natural-language literature task")
    parser.add_argument("--allow-network-metadata", action="store_true", help="Allow public metadata API calls")
    parser.add_argument("--allow-network-rank", action="store_true", help="Allow journal-rank network lookup")
    parser.add_argument("--download", action="store_true", help="Attempt real legal-OA downloads after checks")
    parser.add_argument("--allow-download", action="store_true", help="Permit actual download execution after confirmation")
    parser.add_argument("--yes", action="store_true", help="Confirm real legal-OA download execution")
    parser.add_argument("--max-results", type=int, default=30, help="Maximum metadata results")
    parser.add_argument("--max-downloads", type=int, default=5, help="Maximum download attempts")
    parser.add_argument("--no-llm-diagnosis", action="store_true", help="Use rule-based diagnosis")
    parser.add_argument("--output-dir", default="agent_runs", help="Directory for generated reports")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    user_request = " ".join(args.request).strip()
    if not user_request:
        parser.error("Please provide a literature task.")
    summary = run_literature_graph_agent(
        user_request,
        allow_network_metadata=args.allow_network_metadata,
        allow_network_rank=args.allow_network_rank,
        download=args.download,
        allow_download=args.allow_download,
        yes=args.yes,
        max_results=args.max_results,
        max_downloads=args.max_downloads,
        use_llm_diagnosis=not args.no_llm_diagnosis,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
