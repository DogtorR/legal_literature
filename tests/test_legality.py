from lit_agent.legality import assess_oa_evidence, is_forbidden_url


def test_forbidden_urls_are_detected():
    assert is_forbidden_url("https://sci-hub.example/10.1000/test")
    assert is_forbidden_url("https://libgen.example/book")
    assert is_forbidden_url("https://z-library.example/item")


def test_uncertain_oa_is_candidate_not_downloadable():
    decision = assess_oa_evidence({"pdf_url": "https://example.org/paper.pdf"})
    assert decision["is_legal_oa"] is False
    assert decision["decision"] == "candidate"


def test_explicit_allowed_source_can_be_downloadable_in_future_rounds():
    decision = assess_oa_evidence(
        {
            "source_type": "PubMed Central",
            "pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC000000/pdf/example.pdf",
            "oa_evidence": "PubMed Central public PDF",
        }
    )
    assert decision["is_legal_oa"] is True
    assert decision["decision"] == "downloadable"
