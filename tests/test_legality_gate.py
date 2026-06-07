from lit_agent.legality import assess_legal_oa_candidate, is_forbidden_text, is_forbidden_url


def test_unpaywall_confirmed_oa_allowed_for_future_not_now():
    decision = assess_legal_oa_candidate(
        {
            "source": "unpaywall",
            "doi": "10.1000/a",
            "candidate_url": "https://publisher.example/a.pdf",
            "unpaywall_is_oa": True,
            "host_type": "publisher",
            "license": "cc-by",
        }
    )
    assert decision.decision == "allowed_for_future_download"
    assert decision.is_legal_oa is True
    assert decision.is_download_allowed_now is False


def test_creative_commons_license_url_is_supported():
    decision = assess_legal_oa_candidate(
        {
            "source": "unpaywall",
            "doi": "10.1000/a",
            "candidate_url": "https://publisher.example/a.pdf",
            "license": "https://creativecommons.org/licenses/by/4.0/",
        }
    )
    assert decision.decision == "allowed_for_future_download"


def test_europe_pmc_pmcid_allowed_for_future_not_now():
    decision = assess_legal_oa_candidate({"source": "europe_pmc", "pmcid": "PMC1", "candidate_url": "https://europepmc.org/article/PMC/PMC1", "oa_status": "open"})
    assert decision.decision == "allowed_for_future_download"
    assert decision.is_download_allowed_now is False


def test_crossref_and_openalex_only_are_candidates():
    crossref = assess_legal_oa_candidate({"source": "crossref", "doi": "10.1000/a", "pdf_url_candidate": "https://publisher.example/a.pdf", "license": "cc-by"})
    openalex = assess_legal_oa_candidate({"source": "openalex", "doi": "10.1000/b", "candidate_url": "https://publisher.example/b.pdf", "oa_status": "gold"})
    assert crossref.decision == "candidate_needs_confirmation"
    assert openalex.decision == "candidate_needs_confirmation"


def test_forbidden_url_and_text_are_blocked():
    assert is_forbidden_url("https://annas-archive.example/file.pdf")
    assert is_forbidden_text("requires campus vpn and cookie reuse")
    decision = assess_legal_oa_candidate({"source": "unpaywall", "candidate_url": "https://sci-hub.example/file.pdf", "unpaywall_is_oa": True})
    assert decision.decision == "blocked_forbidden_source"
    decision = assess_legal_oa_candidate({"source": "publisher", "candidate_url": "https://example.org/file.pdf", "evidence": "requires session token reuse"})
    assert decision.decision == "blocked_requires_login_or_bypass"


def test_missing_url_and_unsupported_license():
    missing = assess_legal_oa_candidate({"source": "unpaywall", "doi": "10.1000/a", "unpaywall_is_oa": True})
    unsupported = assess_legal_oa_candidate({"source": "unpaywall", "doi": "10.1000/a", "candidate_url": "https://example.org/a", "license": "all rights reserved"})
    assert missing.decision == "blocked_missing_url"
    assert unsupported.decision == "blocked_unsupported_license"
