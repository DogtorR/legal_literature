from lit_agent.journal_rank import enrich_records_with_rank, get_journal_rank, parse_letpub_detail, write_rank_cache


DETAIL_HTML = """
<html>
<a href="https://www.ncbi.nlm.nih.gov/nlmcatalog?term=2041-1723%5BISSN%5D">PMC</a>
<div><b>CiteScore</b><br>23.4</div>
WOS Quartile: <span style="background: #FFEEEE;">Q1</span>
<table><tr><td>Quartiles By JIF</td><td>Collection</td><td>Quartile</td><td>Rank</td></tr>
<tr><td>Category: MULTIDISCIPLINARY SCIENCES</td><td>SCIE</td><td>Q1</td><td>10/136</td></tr></table>
<script>
series : [{ name:'IF value', type:'line', data : [11.329, 12.353, 14.7] }]
</script>
</html>
"""


def test_parse_letpub_detail_extracts_metrics():
    record = parse_letpub_detail(DETAIL_HTML, "Nature Communications", "/journal-selector/journal/8411")
    assert record.status == "found"
    assert record.issn == "2041-1723"
    assert record.letpub_journal_id == "8411"
    assert record.impact_factor == 14.7
    assert record.citescore == 23.4
    assert record.wos_quartile == "Q1"


def test_rank_cache_prevents_network_lookup(tmp_path):
    cache = tmp_path / "journal_rank_cache.jsonl"
    record = parse_letpub_detail(DETAIL_HTML, "Nature Communications", "/journal-selector/journal/8411")
    write_rank_cache(record, cache)
    cached = get_journal_rank("Nature Communications", cache_path=cache, allow_network=False)
    assert cached["cache_hit"] is True
    assert cached["impact_factor"] == 14.7


def test_enrich_records_filters_by_impact_factor_and_q1(tmp_path):
    cache = tmp_path / "journal_rank_cache.jsonl"
    record = parse_letpub_detail(DETAIL_HTML, "Nature Communications", "/journal-selector/journal/8411")
    write_rank_cache(record, cache)
    records = [{"title": "A", "journal": "Nature Communications"}, {"title": "B", "journal": "Unknown Journal"}]
    filtered = enrich_records_with_rank(records, min_impact_factor=10, require_jcr_q1=True, cache_path=cache)
    assert len(filtered) == 1
    assert filtered[0]["journal_impact_factor"] == 14.7
    assert filtered[0]["journal_wos_quartile"] == "Q1"
