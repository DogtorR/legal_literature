"""Journal rank lookup and cache helpers.

LetPub is treated as a public third-party journal metrics source. The agent
checks a local JSONL cache first and only queries LetPub when explicitly
allowed. Failed lookups are cached as well, so repeated tasks do not keep
retrying the same journal.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from .manifest import append_jsonl, read_jsonl, write_failure
from .metadata import normalize_title
from .sources.common import safe_headers


DEFAULT_CACHE = Path("journal_rank_cache.jsonl")
LETPUB_SEARCH_URL = "https://www.letpub.com/journal-selector"
LETPUB_TIMEOUT = 20
LETPUB_RATE_LIMIT_SECONDS = 1.0


@dataclass(slots=True)
class JournalRankRecord:
    journal: str
    normalized_journal: str
    source: str = "letpub"
    status: str = "unknown"
    issn: str = ""
    letpub_journal_id: str = ""
    letpub_url: str = ""
    impact_factor: float | None = None
    impact_factor_series: list[float] = field(default_factory=list)
    citescore: float | None = None
    wos_quartile: str = ""
    jif_quartiles: list[dict[str, str]] = field(default_factory=list)
    checked_at: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href = ""
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            attrs_dict = {key.lower(): value or "" for key, value in attrs}
            href = attrs_dict.get("href", "")
            if "/journal-selector/journal/" in href:
                self._href = href
                self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href:
            text = " ".join(part.strip() for part in self._text if part.strip())
            self.links.append((text, self._href))
            self._href = ""
            self._text = []


def normalize_journal_name(value: str) -> str:
    text = normalize_title(unescape(value or ""))
    return re.sub(r"\b(the|journal|of)\b", "", text).strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_cache_by_key(path: str | Path = DEFAULT_CACHE) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        journal_key = str(record.get("normalized_journal") or "")
        issn_key = str(record.get("issn") or "").lower()
        if journal_key:
            latest[f"journal:{journal_key}"] = record
        if issn_key:
            latest[f"issn:{issn_key}"] = record
    return latest


def cache_lookup(journal: str, issn: str = "", path: str | Path = DEFAULT_CACHE) -> dict[str, Any] | None:
    latest = _latest_cache_by_key(path)
    if issn:
        cached = latest.get(f"issn:{issn.lower()}")
        if cached:
            return cached
    return latest.get(f"journal:{normalize_journal_name(journal)}")


def write_rank_cache(record: JournalRankRecord | dict[str, Any], path: str | Path = DEFAULT_CACHE) -> None:
    payload = record.to_dict() if isinstance(record, JournalRankRecord) else dict(record)
    append_jsonl(path, payload)


def _fetch(url: str, params: dict[str, Any] | None = None) -> str:
    full_url = f"{url}?{urlencode(params or {})}" if params else url
    request = Request(full_url, headers=safe_headers({"Accept": "text/html"}))
    with urlopen(request, timeout=LETPUB_TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


def _first_search_hit(html: str, journal: str) -> tuple[str, str]:
    parser = _SearchResultParser()
    parser.feed(html)
    wanted = normalize_journal_name(journal)
    if not parser.links:
        return "", ""
    for title, href in parser.links:
        if normalize_journal_name(title) == wanted:
            return title, href
    for title, href in parser.links:
        if wanted and wanted in normalize_journal_name(title):
            return title, href
    return parser.links[0]


def _numbers_from_series(html: str, series_name: str) -> list[float]:
    pattern = re.compile(
        r"name\s*:\s*['\"]%s['\"].{0,600}?data\s*:\s*\[([^\]]+)\]" % re.escape(series_name),
        re.I | re.S,
    )
    match = pattern.search(html)
    if not match:
        return []
    values: list[float] = []
    for value in re.findall(r"-?\d+(?:\.\d+)?", match.group(1)):
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def _impact_factor_series(html: str) -> list[float]:
    direct = _numbers_from_series(html, "IF value")
    if direct:
        return direct
    series_pattern = re.compile(
        r"name\s*:\s*['\"](?P<name>[^'\"]+)['\"].{0,800}?data\s*:\s*\[(?P<data>[^\]]+)\]",
        re.I | re.S,
    )
    candidates: list[tuple[int, list[float]]] = []
    for match in series_pattern.finditer(html):
        name = unescape(match.group("name")).lower()
        values: list[float] = []
        for value in re.findall(r"-?\d+(?:\.\d+)?", match.group("data")):
            try:
                values.append(float(value))
            except ValueError:
                continue
        if not values:
            continue
        start = max(0, match.start() - 300)
        nearby = unescape(html[start:match.start()]).lower()
        score = 0
        if "if" in name or "impact" in name:
            score += 3
        if "if" in nearby or "impact factor" in nearby or "jif" in nearby:
            score += 1
        if score:
            candidates.append((score, values))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _extract_float_near(label: str, html: str) -> float | None:
    match = re.search(re.escape(label) + r".{0,120}?(-?\d+(?:\.\d+)?)", html, re.I | re.S)
    if not match:
        return None
    return float(match.group(1))


def _extract_issn(html: str) -> str:
    match = re.search(r"term=([0-9]{4}-[0-9Xx]{4})%5BISSN%5D", html)
    return match.group(1) if match else ""


def _extract_wos_quartile(html: str) -> str:
    match = re.search(r"WOS Quartile:\s*<span[^>]*>\s*(Q[1-4])\s*</span>", html, re.I)
    return match.group(1).upper() if match else ""


def _extract_jif_quartiles(html: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    table_match = re.search(r"Quartiles By JIF</td>.*?</table>", html, re.I | re.S)
    if not table_match:
        return records
    rows = re.findall(r"<tr>(.*?)</tr>", table_match.group(0), re.I | re.S)
    for row in rows[1:]:
        cells = [
            re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", cell)).strip()
            for cell in re.findall(r"<td[^>]*>(.*?)</td>", row, re.I | re.S)
        ]
        if len(cells) >= 4:
            records.append({"category": cells[0], "collection": cells[1], "quartile": cells[2], "rank": cells[3]})
    return records


def parse_letpub_detail(html: str, journal: str, href: str) -> JournalRankRecord:
    if_series = _impact_factor_series(html)
    citescore = _extract_float_near("CiteScore", html)
    id_match = re.search(r"/journal-selector/journal/(\d+)", href)
    return JournalRankRecord(
        journal=journal,
        normalized_journal=normalize_journal_name(journal),
        status="found",
        issn=_extract_issn(html),
        letpub_journal_id=id_match.group(1) if id_match else "",
        letpub_url=urljoin(LETPUB_SEARCH_URL, href),
        impact_factor=if_series[-1] if if_series else None,
        impact_factor_series=if_series,
        citescore=citescore,
        wos_quartile=_extract_wos_quartile(html),
        jif_quartiles=_extract_jif_quartiles(html),
        checked_at=_now(),
        reason="letpub_detail_parsed",
    )


def query_letpub(journal: str) -> JournalRankRecord:
    normalized = normalize_journal_name(journal)
    if not normalized:
        return JournalRankRecord(journal=journal, normalized_journal="", status="failed", checked_at=_now(), reason="missing_journal_name")
    try:
        time.sleep(LETPUB_RATE_LIMIT_SECONDS)
        search_html = _fetch(
            LETPUB_SEARCH_URL,
            {
                "view": "search",
                "title": journal,
                "subjectarea": "",
                "category": "",
                "if_min": "",
                "if_max": "",
                "sortby": "title",
                "pagenum": 1,
            },
        )
        title, href = _first_search_hit(search_html, journal)
        if not href:
            return JournalRankRecord(journal=journal, normalized_journal=normalized, status="not_found", checked_at=_now(), reason="letpub_search_no_result")
        time.sleep(LETPUB_RATE_LIMIT_SECONDS)
        detail_html = _fetch(urljoin(LETPUB_SEARCH_URL, href))
        return parse_letpub_detail(detail_html, title or journal, href)
    except Exception as exc:
        write_failure("letpub journal rank lookup failed", {"journal": journal, "error": str(exc)})
        return JournalRankRecord(journal=journal, normalized_journal=normalized, status="failed", checked_at=_now(), reason=str(exc))


def get_journal_rank(journal: str, *, issn: str = "", cache_path: str | Path = DEFAULT_CACHE, allow_network: bool = False) -> dict[str, Any]:
    cached = cache_lookup(journal, issn=issn, path=cache_path)
    if cached:
        return dict(cached, cache_hit=True)
    if not allow_network:
        return JournalRankRecord(
            journal=journal,
            normalized_journal=normalize_journal_name(journal),
            status="missing_cache",
            checked_at=_now(),
            reason="network_rank_lookup_not_allowed",
        ).to_dict() | {"cache_hit": False}
    record = query_letpub(journal)
    write_rank_cache(record, path=cache_path)
    payload = record.to_dict()
    payload["cache_hit"] = False
    return payload


def enrich_records_with_rank(
    records: Iterable[dict[str, Any]],
    *,
    min_impact_factor: float | None = None,
    require_jcr_q1: bool = False,
    allow_network: bool = False,
    cache_path: str | Path = DEFAULT_CACHE,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        data = dict(record)
        journal = str(data.get("journal") or data.get("host_venue") or "")
        rank = get_journal_rank(journal, cache_path=cache_path, allow_network=allow_network) if journal else {}
        data["journal_rank"] = rank
        data["journal_impact_factor"] = rank.get("impact_factor")
        data["journal_citescore"] = rank.get("citescore")
        data["journal_wos_quartile"] = rank.get("wos_quartile", "")
        reasons = list(data.get("filter_reasons") or [])
        if min_impact_factor is not None:
            impact_factor = rank.get("impact_factor")
            if impact_factor is None or float(impact_factor) < float(min_impact_factor):
                reasons.append("impact_factor_below_threshold_or_missing")
                data["filter_reasons"] = sorted(set(reasons))
                continue
        if require_jcr_q1 and str(rank.get("wos_quartile") or "").upper() != "Q1":
            reasons.append("journal_not_wos_q1_or_missing")
            data["filter_reasons"] = sorted(set(reasons))
            continue
        if reasons:
            data["filter_reasons"] = sorted(set(reasons))
        enriched.append(data)
    return enriched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Look up and cache LetPub journal metrics.")
    parser.add_argument("--journal", action="append", required=True, help="Journal title to query")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="JSONL cache path")
    parser.add_argument("--allow-network", action="store_true", help="Allow public LetPub lookup when cache is missing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = [get_journal_rank(journal, cache_path=args.cache, allow_network=args.allow_network) for journal in args.journal]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
