"""Scope rules for broad CRISPR corpus collection."""

from __future__ import annotations

from typing import Any


PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY = """
(
CRISPR OR "CRISPR-Cas" OR Cas9 OR Cas12 OR Cas12a OR Cas13 OR Cas13a OR Cas14 OR Cpf1 OR C2c2 OR SHERLOCK OR DETECTR OR HOLMES
)
AND
(
detection OR diagnosis OR diagnostic OR diagnostics OR biosensor OR biosensing OR assay OR sensor
OR structure OR structural OR conformation OR conformational OR mechanism OR mechanistic
OR cleavage OR "collateral cleavage" OR PAM OR PFS
OR crRNA OR sgRNA OR "guide RNA"
OR ribonucleoprotein OR RNP
OR engineering OR engineered OR variant OR mutant OR domain
)
""".strip()

BROAD_CRISPR_CORPUS_QUERIES = [
    PRIMARY_CRISPR_BROAD_BOOLEAN_QUERY,
    "CRISPR structure",
    "CRISPR-Cas mechanism",
    "Cas9 structure",
    "Cas12 structure",
    "Cas12a structure",
    "Cas13 structure",
    "Cas13a structure",
    "Cas12 collateral cleavage",
    "Cas13 collateral cleavage",
    "CRISPR PAM recognition",
    "CRISPR PFS recognition",
    "CRISPR guide RNA",
    "CRISPR crRNA",
    "CRISPR sgRNA",
    "CRISPR biosensor",
    "CRISPR diagnostics",
    "CRISPR detection",
    "SHERLOCK",
    "DETECTR",
    "HOLMES",
]

CRISPR_CORE_TERMS = [
    "CRISPR",
    "CRISPR-Cas",
    "Cas9",
    "Cas12",
    "Cas12a",
    "Cas13",
    "Cas13a",
    "Cas14",
    "Cpf1",
    "C2c2",
    "SHERLOCK",
    "DETECTR",
    "HOLMES",
]

INCLUDE_APPLICATION_TERMS = [
    "detection",
    "diagnostic",
    "diagnostics",
    "diagnosis",
    "biosensor",
    "biosensing",
    "assay",
    "sensor",
    "nucleic acid detection",
    "molecular diagnostics",
    "point-of-care",
    "POCT",
    "pathogen detection",
    "viral detection",
    "bacterial detection",
]

INCLUDE_STRUCTURE_MECHANISM_TERMS = [
    "structure",
    "structural",
    "crystal structure",
    "cryo-EM",
    "conformation",
    "conformational",
    "mechanism",
    "mechanistic",
    "cleavage",
    "collateral cleavage",
    "trans-cleavage",
    "cis-cleavage",
    "PAM",
    "PFS",
    "crRNA",
    "sgRNA",
    "guide RNA",
    "ribonucleoprotein",
    "RNP",
    "domain",
    "mutant",
    "variant",
    "engineered",
    "engineering",
    "specificity",
    "off-target mechanism",
]

PURE_EDITING_TERMS = [
    "genome editing",
    "gene editing",
    "base editing",
    "prime editing",
    "gene therapy",
    "therapeutic editing",
    "crop editing",
    "plant genome editing",
    "breeding",
    "knockout",
    "knock-in",
    "knockin",
    "knock-out",
    "homology-directed repair",
    "HDR editing",
    "non-homologous end joining",
    "NHEJ editing",
]


def _contains_any(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def evaluate_crispr_scope(record: dict[str, Any], *, include_reviews: bool = True, conservative: bool = True) -> dict[str, Any]:
    text = " ".join(
        str(record.get(key) or "")
        for key in ("title", "abstract", "publication_type", "journal", "keywords", "keywords_matched")
    )
    pub_type = str(record.get("publication_type") or "").lower()
    matched_core = _contains_any(text, CRISPR_CORE_TERMS)
    matched_application = _contains_any(text, INCLUDE_APPLICATION_TERMS)
    matched_structure = _contains_any(text, INCLUDE_STRUCTURE_MECHANISM_TERMS)
    matched_exclude = _contains_any(text, PURE_EDITING_TERMS)

    if not include_reviews and "review" in pub_type:
        status = "excluded"
        reason = "excluded_review"
    elif not matched_core:
        status = "excluded"
        reason = "missing_crispr_core_term"
    elif matched_application:
        status = "included"
        reason = "mixed_editing_with_detection_retained" if matched_exclude else "matched_broad_crispr_scope"
    elif matched_structure:
        if matched_exclude and conservative:
            status = "needs_manual_review"
            reason = "mixed_editing_with_structure_or_mechanism"
        else:
            status = "included"
            reason = "matched_broad_crispr_scope"
    elif matched_exclude:
        status = "excluded"
        reason = "pure_editing_likely"
    else:
        status = "needs_manual_review"
        reason = "crispr_core_without_clear_application_or_mechanism"

    return {
        "scope_status": status,
        "scope_reason": reason,
        "matched_include_terms": sorted(set(matched_core + matched_application + matched_structure)),
        "matched_exclude_terms": sorted(set(matched_exclude)),
        "is_pure_editing_likely": bool(matched_exclude and not (matched_application or matched_structure)),
        "is_detection_related": bool(matched_application),
        "is_structure_or_mechanism_related": bool(matched_structure),
    }
