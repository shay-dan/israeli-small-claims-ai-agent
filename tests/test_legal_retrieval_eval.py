"""Replay the privacy-safe legal retrieval baseline stored in tests/fixtures."""

import json
from pathlib import Path

from small_claims_agent.legal import retrieve_authorities_for_case
from small_claims_agent.models import CaseData


EVAL_CASES_PATH = Path(__file__).parent / "fixtures" / "legal_retrieval_eval_cases.json"


def _load_cases() -> list[dict]:
    payload = json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))
    assert payload["suite_id"] == "legal-retrieval-baseline-v1"
    return payload["cases"]


def test_legal_retrieval_eval_suite_has_unique_cases_and_expected_outputs():
    cases = _load_cases()
    case_ids = [item["id"] for item in cases]
    assert len(cases) >= 20
    assert len(case_ids) == len(set(case_ids))
    assert all("expected" in item and isinstance(item["expected"], list) for item in cases)


def test_legal_retrieval_matches_saved_baseline_cases():
    for item in _load_cases():
        case = CaseData(
            category=item["category"],
            product_or_service_details=item["product"],
            event_summary=item["event"],
            primary_remedy=item.get("primary", "undecided"),
            alternative_remedy=item.get("alternative", "none"),
            damage_items=item.get("amounts", []),
        )
        actual = {record.id for record in retrieve_authorities_for_case(case)}
        expected = set(item["expected"])
        assert actual == expected, (
            f"{item['id']}: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
