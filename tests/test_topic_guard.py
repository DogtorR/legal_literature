from lit_agent.topic_guard import apply_topic_guard, is_crispr_detection_record


def test_crispr_detection_record_passes():
    record = {"title": "CRISPR Cas12 assay for viral detection"}
    assert is_crispr_detection_record(record)


def test_detection_without_crispr_fails():
    record = {"title": "Rapid pathogen detection by PCR"}
    assert not is_crispr_detection_record(record)


def test_crispr_without_detection_fails():
    record = {"title": "CRISPR genome editing in cells"}
    assert not is_crispr_detection_record(record)


def test_crispr_screening_without_detection_fails():
    record = {"title": "CRISPR base editor screening identifies MEN1 mutations"}
    assert not is_crispr_detection_record(record)


def test_reco_cas_liquid_biopsy_variant_profiling_passes():
    record = {"title": "Single-nucleotide variant profiling in liquid biopsy with RECO-Cas"}
    assert is_crispr_detection_record(record)


def test_cas_boundary_does_not_match_caspase():
    record = {"title": "Caspase variant profiling in liquid biopsy"}
    assert not is_crispr_detection_record(record)


def test_apply_topic_guard_keeps_only_dual_term_records():
    records = [
        {"title": "CRISPR Cas12 assay for viral detection"},
        {"title": "Rapid pathogen detection by PCR"},
        {"title": "CRISPR genome editing in cells"},
    ]
    kept = apply_topic_guard(records, {"enabled": True, "type": "crispr_detection"})
    assert len(kept) == 1
    assert kept[0]["topic_guard_passed"] is True
    assert kept[0]["title"] == "CRISPR Cas12 assay for viral detection"
