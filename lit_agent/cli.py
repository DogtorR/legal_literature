"""Command line interface for local dry-run query normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .agent_graph import run_literature_graph_agent
from .config import load_config, normalize_config
from .downloader import execute_download_plan, plan_downloads_from_legality_audit
from .failure_analysis import summarize_dedupe, summarize_failures
from .filters import apply_query_filters, apply_record_scope_filters, record_matches_scope
from .legality import assess_legal_oa_candidate
from .landing_page import resolve_landing_page, select_landing_page_records
from .journal_rank import enrich_records_with_rank
from .manifest import read_jsonl, write_candidates, write_download_plan, write_failure, write_legality_audits, write_metadata_results, write_search_log
from .metadata import deduplicate_records
from .query import QuerySpec, load_queries_csv, merge_cli_overrides, query_specs_from_config
from .scope import filter_records_for_scope, record_in_scope, scope_tokens
from .sources import crossref, europe_pmc, openalex, pubmed, unpaywall
from .topic_guard import apply_topic_guard


def _defaults_from_config(config: dict) -> dict:
    return {
        "keywords": config.get("keywords"),
        "year_from": config.get("year_from"),
        "year_to": config.get("year_to"),
        "publication_date_from": config.get("publication_date_from"),
        "publication_date_to": config.get("publication_date_to"),
        "journal": (config.get("journals") or [""])[0],
        "publication_type": ";".join(config.get("publication_types") or []),
        "max_results": config.get("max_results_per_source"),
        "include_terms": config.get("include_terms"),
        "exclude_terms": config.get("exclude_terms"),
        "source_preference": config.get("sources"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare legal OA literature search queries.")
    parser.add_argument("request", nargs="*", help="Natural-language literature task for the high-level LangGraph agent")
    parser.add_argument("--config", help="Path to search_config.yaml")
    parser.add_argument("--input", help="Path to queries.csv")
    parser.add_argument("--year-from", type=int, dest="year_from")
    parser.add_argument("--year-to", type=int, dest="year_to")
    parser.add_argument("--keywords")
    parser.add_argument("--include-terms", dest="include_terms")
    parser.add_argument("--exclude-terms", dest="exclude_terms")
    parser.add_argument("--max-results", type=int, dest="max_results")
    parser.add_argument("--source", action="append", choices=["openalex", "crossref", "pubmed", "europe_pmc", "unpaywall", "arxiv", "doaj", "all"])
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--oa-check", action="store_true")
    parser.add_argument("--legality-check", action="store_true")
    parser.add_argument("--allow-network-metadata", action="store_true")
    parser.add_argument("--rank-check", action="store_true", dest="rank_check")
    parser.add_argument("--allow-network-rank", action="store_true", dest="allow_network_rank")
    parser.add_argument("--min-impact-factor", type=float, dest="min_impact_factor")
    parser.add_argument("--require-jcr-q1", action="store_true", dest="require_jcr_q1")
    parser.add_argument("--journal-rank-cache", dest="journal_rank_cache")
    parser.add_argument("--dry-run", action="store_true", help="Prepare queries without network calls")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--discover-pdf-urls", action="store_true", dest="discover_pdf_urls")
    parser.add_argument("--discover-landing-pages", action="store_true", dest="discover_landing_pages")
    parser.add_argument("--plan-downloads", action="store_true", dest="plan_downloads")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Confirm real legal-OA download execution for the high-level graph agent")
    parser.add_argument("--max-downloads", type=int, dest="max_downloads")
    parser.add_argument("--output-dir", default="agent_runs", help="Directory for high-level graph reports")
    parser.add_argument("--analyze-failures", action="store_true", dest="analyze_failures")
    parser.add_argument("--summarize-dedupe", action="store_true", dest="summarize_dedupe")
    parser.add_argument("--round-scope", choices=["all", "current"], default="all", dest="round_scope")
    parser.add_argument("--query-name", dest="query_name")
    parser.add_argument("--since-round", type=int, dest="since_round")
    return parser


def _load_specs(args: argparse.Namespace) -> tuple[list[QuerySpec], dict]:
    config = load_config(args.config) if args.config else normalize_config({})
    specs: list[QuerySpec] = []
    if args.input:
        specs.extend(load_queries_csv(Path(args.input), defaults=_defaults_from_config(config)))
    elif args.config:
        queries_file = config.get("queries_file") or config.get("queries_csv")
        if queries_file:
            queries_path = Path(str(queries_file))
            if not queries_path.is_absolute():
                queries_path = Path(args.config).resolve().parent / queries_path
            specs.extend(load_queries_csv(queries_path, defaults=_defaults_from_config(config)))
        else:
            specs.extend(query_specs_from_config(config))
    if not specs:
        specs.append(QuerySpec(input_id="cli_query"))

    return merge_cli_overrides(specs, args), config


def _mock_records(path: Path = Path("examples/mock_metadata.jsonl")) -> list[dict]:
    return read_jsonl(path)


SOURCE_MODULES = {
    "openalex": openalex,
    "crossref": crossref,
    "europe_pmc": europe_pmc,
    "pubmed": pubmed,
    "unpaywall": unpaywall,
}


def _selected_sources(args: argparse.Namespace, config: dict) -> list[str]:
    requested = args.source or config.get("sources", [])
    if "all" in requested:
        return ["openalex", "crossref", "europe_pmc", "pubmed", "unpaywall"]
    return [source for source in requested if source in SOURCE_MODULES]


def _candidate_key(record: dict) -> tuple[str, str, str]:
    return (str(record.get("doi") or ""), str(record.get("candidate_url") or record.get("pdf_url_candidate") or ""), str(record.get("reason") or ""))


def _decision_candidate(decision: dict) -> dict:
    return {
        "query_id": decision.get("query_id", ""),
        "title": decision.get("title", ""),
        "doi": decision.get("doi", ""),
        "pmid": decision.get("pmid", ""),
        "pmcid": decision.get("pmcid", ""),
        "journal": decision.get("journal", ""),
        "publication_year": decision.get("publication_year"),
        "publication_date": decision.get("publication_date", ""),
        "source": "legality_gate",
        "candidate_url": decision.get("candidate_url", ""),
        "pdf_url_candidate": decision.get("pdf_url_candidate", ""),
        "landing_url": decision.get("landing_url", ""),
        "license": decision.get("license", ""),
        "oa_status": decision.get("oa_status", ""),
        "host_type": decision.get("host_type", ""),
        "version": decision.get("version", ""),
        "reason": decision.get("reason", ""),
        "suggested_manual_check": decision.get("required_next_step", ""),
        "is_legal_oa_candidate": decision.get("is_legal_oa", False),
        "is_download_allowed_now": False,
    }


def _dedupe_new_candidates(records: list[dict]) -> list[dict]:
    existing_keys = {_candidate_key(record) for record in read_jsonl("candidates.jsonl")}
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for record in records:
        key = _candidate_key(record)
        if key in existing_keys or key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _rank_policy_from_config(config: dict) -> dict:
    policy = config.get("journal_rank_policy") or {}
    return policy if isinstance(policy, dict) else {}


def _topic_guard_from_config(config: dict) -> dict:
    guard = config.get("topic_guard") or {}
    return guard if isinstance(guard, dict) else {}


def _record_scope_from_config(config: dict) -> dict:
    scope = config.get("record_scope") or {}
    if not isinstance(scope, dict):
        scope = {}
    if not scope:
        journals = config.get("journals") or []
        scope = {
            "journals": journals,
            "publication_date_from": config.get("publication_date_from") or "",
            "publication_date_to": config.get("publication_date_to") or "",
            "require_journal_match": bool(journals),
            "require_date_match": bool(config.get("publication_date_from") and config.get("publication_date_to")),
        }
    return scope


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    user_request = " ".join(args.request).strip()
    if user_request and not args.config and not args.input:
        summary = run_literature_graph_agent(
            user_request,
            allow_network_metadata=args.allow_network_metadata,
            allow_network_rank=args.allow_network_rank,
            download=args.download,
            allow_download=args.allow_download,
            yes=args.yes,
            max_results=args.max_results or 30,
            max_downloads=args.max_downloads or 5,
            output_dir=args.output_dir,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    specs, config = _load_specs(args)
    if args.query_name:
        config["query_name"] = args.query_name
    historical_scope = args.round_scope or "all"
    active_scope_tokens = scope_tokens(specs, config)
    def scoped(record: dict) -> bool:
        return record_in_scope(record, scope=historical_scope, tokens=active_scope_tokens, since_round=args.since_round)

    audit_dir = Path("orchestrator/audits/round_013")
    if args.analyze_failures:
        records = read_jsonl("failures.jsonl")
        summary_payload = summarize_failures(records)
        summary_payload["historical_scope"] = "all"
        _write_json(audit_dir / "failure_summary.json", summary_payload)
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
        return 0
    if args.summarize_dedupe:
        candidate_summary = summarize_dedupe(read_jsonl("candidates.jsonl"), key_name="candidate")
        search_log_summary = summarize_dedupe(read_jsonl("search_log.jsonl"), key_name="search_log")
        _write_json(audit_dir / "candidate_dedupe_summary.json", candidate_summary)
        _write_json(audit_dir / "search_log_dedupe_summary.json", search_log_summary)
        print(json.dumps({"candidate_dedupe_summary": candidate_summary, "search_log_dedupe_summary": search_log_summary}, ensure_ascii=False, indent=2))
        return 0
    dry_run = bool(args.dry_run or (config.get("dry_run", True) and not args.allow_download))
    selected_sources = _selected_sources(args, config)
    metadata_only = bool(args.metadata_only)
    rank_policy = _rank_policy_from_config(config)
    topic_guard_policy = _topic_guard_from_config(config)
    record_scope_policy = _record_scope_from_config(config)
    min_impact_factor = args.min_impact_factor
    if min_impact_factor is None and rank_policy.get("min_impact_factor") not in (None, ""):
        min_impact_factor = float(rank_policy.get("min_impact_factor"))
    require_jcr_q1 = bool(args.require_jcr_q1 or rank_policy.get("require_jcr_q1"))
    rank_check = bool(args.rank_check or min_impact_factor is not None or require_jcr_q1)
    journal_rank_cache = str(args.journal_rank_cache or rank_policy.get("cache_path") or "journal_rank_cache.jsonl")
    allow_network_rank = bool(args.allow_network_rank or rank_policy.get("allow_network_rank"))
    metadata_records: list[dict] = []
    candidate_records: list[dict] = []
    network_calls_by_source = {source: 0 for source in selected_sources}
    search_details: list[dict] = []
    oa_check_info = {
        "checked_doi_count": 0,
        "confirmed_oa_count": 0,
        "no_oa_count": 0,
        "missing_doi_count": 0,
        "candidates_written": 0,
    }
    legality_stats = {
        "legality_checked": 0,
        "allowed_for_future_download": 0,
        "candidate_needs_confirmation": 0,
        "blocked_count": 0,
        "forbidden_count": 0,
        "missing_url_count": 0,
    }
    download_stats = {
        "planned_downloads": 0,
        "skipped_dry_run": 0,
        "blocked_downloads": 0,
        "manifest_records_written": 0,
        "final_preflight_passed": 0,
        "final_preflight_blocked": 0,
    }
    landing_stats = {
        "landing_pages_checked": 0,
        "landing_network_calls": 0,
        "candidate_pdf_links_found": 0,
        "landing_requires_confirmation_count": 0,
        "landing_with_existing_oa_evidence_count": 0,
        "landing_blocked_login_paywall_count": 0,
        "landing_blocked_forbidden_count": 0,
        "landing_blocked_direct_pdf_count": 0,
        "landing_candidates_written": 0,
        "landing_plan_records_written": 0,
        "landing_pages_selected": 0,
        "excluded_mock_count": 0,
        "excluded_known_failed_count": 0,
        "landing_duplicate_doi_count": 0,
        "landing_known_failure_breakdown": {},
        "landing_priority_breakdown": {},
    }
    topic_guard_stats = {
        "topic_guard_enabled": bool(topic_guard_policy.get("enabled")),
        "topic_guard_type": str(topic_guard_policy.get("type") or ""),
        "topic_guard_input_count": 0,
        "topic_guard_filtered_count": 0,
        "topic_guard_removed_count": 0,
    }
    record_scope_stats = {
        "scope_input_count": 0,
        "scope_kept_count": 0,
        "scope_removed_count": 0,
    }
    failures_before = len(read_jsonl("failures.jsonl"))
    if args.oa_check and "unpaywall" in selected_sources:
        records_to_check = unpaywall.collect_oa_check_inputs(args.max_results)
        records_to_check, oa_scope_stats = apply_record_scope_filters(records_to_check, record_scope_policy)
        for key, value in oa_scope_stats.items():
            record_scope_stats[f"oa_{key}"] = value
        if topic_guard_policy.get("enabled") and records_to_check:
            oa_topic_input_count = len(records_to_check)
            records_to_check = apply_topic_guard(records_to_check, topic_guard_policy)
            topic_guard_stats["oa_topic_guard_input_count"] = oa_topic_input_count
            topic_guard_stats["oa_topic_guard_filtered_count"] = len(records_to_check)
            topic_guard_stats["oa_topic_guard_removed_count"] = oa_topic_input_count - len(records_to_check)
            write_search_log(
                "oa_topic_guard",
                {
                    "topic_guard_enabled": topic_guard_stats["topic_guard_enabled"],
                    "topic_guard_type": topic_guard_stats["topic_guard_type"],
                    "oa_topic_guard_input_count": oa_topic_input_count,
                    "oa_topic_guard_filtered_count": len(records_to_check),
                    "oa_topic_guard_removed_count": oa_topic_input_count - len(records_to_check),
                    "downloads": 0,
                },
            )
        checked, candidates, info = unpaywall.check_records(
            records_to_check,
            config,
            dry_run=not args.allow_network_metadata,
            allow_network=args.allow_network_metadata,
            max_results=args.max_results,
        )
        metadata_records.extend(checked)
        candidate_records.extend(candidates)
        network_calls_by_source["unpaywall"] += int(info.get("network_calls", 0))
        oa_check_info = info
        search_details.append(dict(source="unpaywall", **info))
    if selected_sources:
        for spec in specs:
            for source in selected_sources:
                if source == "unpaywall":
                    continue
                module = SOURCE_MODULES[source]
                records = module.search(spec, config, dry_run=not args.allow_network_metadata, allow_network=args.allow_network_metadata)
                metadata_records.extend(records)
                if hasattr(module, "candidates_from_records"):
                    candidate_records.extend(module.candidates_from_records(records))
                info = getattr(module, "LAST_SEARCH_INFO", {})
                network_calls_by_source[source] += int(info.get("network_calls", 0))
                search_details.append(dict(info))
    else:
        mock_records = deduplicate_records(_mock_records())
        metadata_records = [record for spec in specs for record in apply_query_filters(mock_records, spec)]

    metadata_records = deduplicate_records(metadata_records)
    metadata_records, record_scope_stats = apply_record_scope_filters(metadata_records, record_scope_policy)
    if record_scope_stats["scope_input_count"]:
        write_search_log("record_scope_filter", dict(record_scope_stats, scope=record_scope_policy, downloads=0))
    candidate_records, candidate_scope_stats = apply_record_scope_filters(candidate_records, record_scope_policy)
    for key, value in candidate_scope_stats.items():
        record_scope_stats[f"candidate_{key}"] = value
    topic_guard_stats["topic_guard_input_count"] = len(metadata_records)
    if topic_guard_policy.get("enabled") and metadata_records:
        guarded_records = apply_topic_guard(metadata_records, topic_guard_policy)
        topic_guard_stats["topic_guard_filtered_count"] = len(guarded_records)
        topic_guard_stats["topic_guard_removed_count"] = len(metadata_records) - len(guarded_records)
        metadata_records = guarded_records
        write_search_log("topic_guard", dict(topic_guard_stats, downloads=0))
    if topic_guard_policy.get("enabled") and candidate_records:
        candidate_input_count = len(candidate_records)
        guarded_candidates = apply_topic_guard(candidate_records, topic_guard_policy)
        topic_guard_stats["candidate_topic_guard_input_count"] = candidate_input_count
        topic_guard_stats["candidate_topic_guard_filtered_count"] = len(guarded_candidates)
        topic_guard_stats["candidate_topic_guard_removed_count"] = candidate_input_count - len(guarded_candidates)
        candidate_records = guarded_candidates
        write_search_log(
            "candidate_topic_guard",
            {
                "topic_guard_enabled": topic_guard_stats["topic_guard_enabled"],
                "topic_guard_type": topic_guard_stats["topic_guard_type"],
                "candidate_topic_guard_input_count": candidate_input_count,
                "candidate_topic_guard_filtered_count": len(guarded_candidates),
                "candidate_topic_guard_removed_count": candidate_input_count - len(guarded_candidates),
                "downloads": 0,
            },
        )
    rank_stats = {
        "rank_check": rank_check,
        "min_impact_factor": min_impact_factor,
        "require_jcr_q1": require_jcr_q1,
        "rank_input_count": len(metadata_records),
        "rank_filtered_count": 0,
        "rank_cache_path": journal_rank_cache,
        "rank_network_enabled": allow_network_rank,
    }
    if rank_check and metadata_records:
        ranked_records = enrich_records_with_rank(
            metadata_records,
            min_impact_factor=min_impact_factor,
            require_jcr_q1=require_jcr_q1,
            allow_network=allow_network_rank,
            cache_path=journal_rank_cache,
        )
        rank_stats["rank_filtered_count"] = len(ranked_records)
        rank_stats["rank_removed_count"] = len(metadata_records) - len(ranked_records)
        rank_stats["rank_cache_hits"] = sum(1 for record in ranked_records if (record.get("journal_rank") or {}).get("cache_hit"))
        metadata_records = ranked_records
        write_search_log("journal_rank_check", dict(rank_stats, downloads=0))
    if args.legality_check:
        gate_inputs = read_jsonl("metadata_results.jsonl") + read_jsonl("candidates.jsonl") + metadata_records + candidate_records
        gate_inputs = [record for record in gate_inputs if str(record.get("source") or "") != "legality_gate"]
        gate_inputs, gate_scope_stats = apply_record_scope_filters(gate_inputs, record_scope_policy)
        for key, value in gate_scope_stats.items():
            record_scope_stats[f"legality_{key}"] = value
        decisions = [assess_legal_oa_candidate(record).to_dict() for record in gate_inputs]
        write_legality_audits(decisions)
        legality_stats["legality_checked"] = len(decisions)
        legality_stats["allowed_for_future_download"] = sum(decision["decision"] == "allowed_for_future_download" for decision in decisions)
        legality_stats["candidate_needs_confirmation"] = sum(decision["decision"] == "candidate_needs_confirmation" for decision in decisions)
        legality_stats["blocked_count"] = sum(str(decision["decision"]).startswith("blocked") for decision in decisions)
        legality_stats["forbidden_count"] = sum(decision["decision"] in {"blocked_forbidden_source", "blocked_requires_login_or_bypass"} for decision in decisions)
        legality_stats["missing_url_count"] = sum(decision["decision"] == "blocked_missing_url" for decision in decisions)
        legality_candidates = [_decision_candidate(decision) for decision in decisions if decision["decision"] in {"candidate_needs_confirmation", "allowed_for_future_download"}]
        candidate_records.extend(legality_candidates)
        write_search_log("legality_check", dict(legality_stats, downloads=0, network_calls_by_source=network_calls_by_source))
    metadata_written = write_metadata_results(metadata_records)
    unique_candidates = _dedupe_new_candidates(candidate_records)
    candidates_written = write_candidates(unique_candidates)
    if args.discover_landing_pages:
        max_landing = args.max_results or config.get("max_results_per_source") or 5
        all_landing_source_records = read_jsonl("legality_audit.jsonl") + read_jsonl("candidates.jsonl") + read_jsonl("metadata_results.jsonl")
        landing_source_records = filter_records_for_scope(all_landing_source_records, scope=historical_scope, tokens=active_scope_tokens, since_round=args.since_round)
        landing_inputs, ranking_stats = select_landing_page_records(landing_source_records, max_results=max_landing, failure_records=read_jsonl("failures.jsonl"))
        landing_stats["landing_pages_selected"] = int(ranking_stats.get("selected_count", 0))
        landing_stats["excluded_mock_count"] = int(ranking_stats.get("excluded_mock_count", 0))
        landing_stats["excluded_known_failed_count"] = int(ranking_stats.get("excluded_known_failed_count", 0))
        landing_stats["landing_duplicate_doi_count"] = int(ranking_stats.get("duplicate_doi_count", 0))
        landing_stats["landing_known_failure_breakdown"] = ranking_stats.get("known_failure_breakdown", {})
        landing_stats["landing_priority_breakdown"] = ranking_stats.get("priority_breakdown", {})
        write_search_log(
            "doi_landing_page_input_ranking",
            {
                "selected_count": landing_stats["landing_pages_selected"],
                "excluded_mock_count": landing_stats["excluded_mock_count"],
                "excluded_known_failed_count": landing_stats["excluded_known_failed_count"],
                "duplicate_doi_count": landing_stats["landing_duplicate_doi_count"],
                "known_failure_breakdown": landing_stats["landing_known_failure_breakdown"],
                "priority_breakdown": landing_stats["landing_priority_breakdown"],
                "historical_scope": historical_scope,
                "all_input_count": len(all_landing_source_records),
                "input_count": len(landing_source_records),
                "max_results": max_landing,
                "downloads": 0,
            },
        )
        landing_candidates: list[dict] = []
        for record in landing_inputs:
            result = resolve_landing_page(record, dry_run=not args.allow_network_metadata, allow_network=args.allow_network_metadata)
            landing_stats["landing_pages_checked"] += 1
            landing_stats["landing_network_calls"] += int(result.network_calls)
            landing_stats["candidate_pdf_links_found"] += int(result.candidate_count)
            landing_stats["landing_requires_confirmation_count"] += int(result.requires_confirmation_count)
            landing_stats["landing_with_existing_oa_evidence_count"] += int(result.with_existing_oa_evidence_count)
            if result.reason == "doi_landing_blocked_requires_login_or_paywall":
                landing_stats["landing_blocked_login_paywall_count"] += 1
                write_failure("doi landing page blocked for login or paywall", result.to_dict())
            if result.reason == "doi_landing_blocked_forbidden":
                landing_stats["landing_blocked_forbidden_count"] += 1
                write_failure("doi landing page blocked forbidden", result.to_dict())
            if result.reason in {"doi_landing_direct_pdf_url_not_fetched", "doi_landing_redirect_to_pdf_not_fetched", "doi_landing_pdf_response_not_fetched"}:
                landing_stats["landing_blocked_direct_pdf_count"] += 1
                write_failure("doi landing page blocked direct pdf", result.to_dict())
            if result.status == "failed":
                write_failure("doi landing page discovery failed", result.to_dict())
            for candidate in result.candidates:
                if candidate.get("planned_status") == "candidate_direct_pdf_discovered_not_downloaded_round_011":
                    write_download_plan(candidate)
                    landing_stats["landing_plan_records_written"] += 1
                else:
                    landing_candidates.append(candidate)
            if historical_scope != "current" or args.allow_network_metadata:
                write_search_log(
                    "doi_landing_page_discovery",
                    {
                        "source": "doi_landing_page",
                        "query_id": record.get("query_id", ""),
                        "doi": record.get("doi", ""),
                        "landing_url": result.landing_url,
                        "candidate_count": result.candidate_count,
                        "blocked_count": result.blocked_count,
                        "requires_confirmation_count": result.requires_confirmation_count,
                        "network_calls": result.network_calls,
                        "dry_run": result.dry_run,
                        "downloads": 0,
                    },
                )
        write_search_log(
            "doi_landing_page_discovery_summary",
            {
                "historical_scope": historical_scope,
                "landing_pages_checked": landing_stats["landing_pages_checked"],
                "landing_pages_selected": landing_stats["landing_pages_selected"],
                "candidate_pdf_links_found": landing_stats["candidate_pdf_links_found"],
                "network_calls": landing_stats["landing_network_calls"],
                "downloads": 0,
            },
        )
        landing_stats["landing_candidates_written"] = write_candidates(_dedupe_new_candidates(landing_candidates))
    if args.plan_downloads or args.download or args.discover_pdf_urls:
        max_downloads = args.max_downloads or config.get("max_downloads") or args.max_results
        def plan_filter(record: dict) -> bool:
            if (historical_scope != "all" or args.since_round is not None) and not scoped(record):
                return False
            return record_matches_scope(record, record_scope_policy)[0]

        requests = plan_downloads_from_legality_audit(max_downloads=max_downloads, record_filter=plan_filter)
        effective_dry_run = bool(args.dry_run or not args.allow_download)
        results = execute_download_plan(requests, dry_run=effective_dry_run, allow_download=bool(args.allow_download and not effective_dry_run))
        download_stats["planned_downloads"] = len(requests)
        download_stats["skipped_dry_run"] = sum(result.status == "skipped_dry_run" for result in results)
        download_stats["blocked_downloads"] = sum(result.status == "blocked" for result in results)
        download_stats["actual_downloads"] = sum(result.status == "downloaded" for result in results)
        download_stats["download_network_calls"] = sum(result.network_calls for result in results)
        download_stats["manifest_records_written"] = sum(1 for result in results if result.manifest_record)
        download_stats["final_preflight_passed"] = download_stats["skipped_dry_run"]
        download_stats["final_preflight_blocked"] = download_stats["blocked_downloads"]
    failures_written = max(0, len(read_jsonl("failures.jsonl")) - failures_before)
    first = specs[0]
    filtered_count = len(metadata_records)
    summary = {
        "query_count": len(specs),
        "dry_run": dry_run,
        "metadata_only": metadata_only,
        "oa_check": bool(args.oa_check),
        "legality_check": bool(args.legality_check),
        "downloads_enabled": bool(args.download and args.allow_download and not dry_run),
        "allow_download": bool(args.allow_download),
        "max_downloads": args.max_downloads,
        "source": ",".join(selected_sources),
        "sources": selected_sources,
        "year_from": first.year_from,
        "year_to": first.year_to,
        "keywords": first.keywords,
        "include_terms": first.include_terms,
        "exclude_terms": first.exclude_terms,
        "filtered_mock_records_count": filtered_count,
        "metadata_results_written": metadata_written,
        "candidates_written": candidates_written,
        "failures_written": failures_written,
        "oa_checked": oa_check_info.get("checked_doi_count", 0),
        "confirmed_oa_candidates": oa_check_info.get("confirmed_oa_count", 0),
        "no_oa": oa_check_info.get("no_oa_count", 0),
        "missing_doi": oa_check_info.get("missing_doi_count", 0),
        **legality_stats,
        **download_stats,
        **landing_stats,
        **topic_guard_stats,
        **record_scope_stats,
        **rank_stats,
        "download_policy": config.get("download_policy", "legal_oa_only"),
        "historical_scope": historical_scope,
        "scope_tokens": sorted(active_scope_tokens),
        "since_round": args.since_round,
        "network_calls": sum(network_calls_by_source.values()) + int(download_stats.get("download_network_calls", 0)) + int(landing_stats.get("landing_network_calls", 0)),
        "network_calls_by_source": network_calls_by_source,
        "downloads": 0,
        "status": "dry_run" if dry_run else "download_attempt",
        "search_details": search_details,
        "queries": [spec.to_dict() for spec in specs],
    }
    summary["downloads"] = int(download_stats.get("actual_downloads", 0))
    if summary["downloads"]:
        summary["status"] = "downloaded"
    write_search_log(
        "cli_dry_run",
        {
            "query_count": len(specs),
            "sources": selected_sources,
            "filtered_mock_records_count": filtered_count,
            "network_calls": sum(network_calls_by_source.values()),
            "network_calls_by_source": network_calls_by_source,
            "downloads": 0,
            "metadata_results_written": metadata_written,
            "candidates_written": candidates_written,
            "failures_written": failures_written,
            "oa_check": bool(args.oa_check),
            "legality_check": bool(args.legality_check),
            "oa_checked": oa_check_info.get("checked_doi_count", 0),
            "confirmed_oa_candidates": oa_check_info.get("confirmed_oa_count", 0),
            "no_oa": oa_check_info.get("no_oa_count", 0),
            "missing_doi": oa_check_info.get("missing_doi_count", 0),
            **legality_stats,
            **download_stats,
            **landing_stats,
            **topic_guard_stats,
            **record_scope_stats,
            **rank_stats,
        },
    )
    if not dry_run:
        summary["reason"] = "explicit_allow_download_round_completed_with_final_preflight"
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
