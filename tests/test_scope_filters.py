from types import SimpleNamespace

from lit_agent.scope import filter_records_for_scope, record_in_scope, scope_tokens


def test_scope_tokens_include_config_query_name_and_spec_ids():
    tokens = scope_tokens([SimpleNamespace(input_id="round_012_q001")], {"query_name": "round_012_batch"})
    assert tokens == {"round_012_q001", "round_012_batch"}


def test_current_scope_keeps_only_matching_query_records():
    tokens = {"round_012_q001"}
    records = [
        {"query_id": "round_012_q001", "doi": "10.1/a"},
        {"query_id": "round_008_q001", "doi": "10.1/b"},
    ]

    scoped = filter_records_for_scope(records, scope="current", tokens=tokens)

    assert scoped == [records[0]]


def test_since_round_filters_round_prefixes():
    assert record_in_scope({"query_id": "round_012_q001"}, since_round=12)
    assert not record_in_scope({"query_id": "round_008_q001"}, since_round=12)
