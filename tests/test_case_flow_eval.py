"""Replay synthetic conversations through interview, validation, retrieval and local drafting."""

import json
from decimal import Decimal
from pathlib import Path

from small_claims_agent.graph import build_graph, initial_state
from small_claims_agent.interview import (
    inferred_completed_fields,
    next_interview_field,
    opening_story_missing_parts,
    seed_case_from_opening,
)
from small_claims_agent.llm import DemoAgentClient
from small_claims_agent.models import CaseData, DamageItem
from small_claims_agent.validators import validate_case


EVAL_CASES_PATH = Path(__file__).parent / "fixtures" / "case_flow_eval_cases.json"


def _load_suite() -> dict:
    payload = json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))
    assert payload["suite_id"] == "case-flow-eval-v1"
    return payload


def _replay_conversation(scenario: dict, defaults: dict) -> CaseData:
    from small_claims_agent.interview import apply_chat_answer, seed_case_from_opening

    case = seed_case_from_opening(scenario["opening"])
    assert case.event_summary == scenario["opening"]
    answers = defaults | scenario.get("answers", {})
    completed = inferred_completed_fields(case)
    asked: list[str] = []

    while field := next_interview_field(case, completed):
        assert field in answers, f"{scenario['id']}: missing synthetic answer for {field}"
        assert field not in asked, f"{scenario['id']}: interview repeated {field} without a correction"
        case = apply_chat_answer(case, field, answers[field])
        completed.add(field)
        asked.append(field)

    case.facts_confirmed = True
    return case


def test_case_flow_eval_suite_is_reproducible_and_privacy_safe():
    suite = _load_suite()
    ids = [case["id"] for case in suite["cases"]]
    assert len(ids) >= 8
    assert len(ids) == len(set(ids))
    assert all(case["opening"].strip() and case.get("expected") for case in suite["cases"])
    # The saved examples use synthetic placeholders, never real identity/contact data.
    serialized = EVAL_CASES_PATH.read_text(encoding="utf-8")
    assert "050-0000000" not in serialized
    assert "123456789" not in serialized


def test_full_chat_to_draft_synthetic_scenarios():
    suite = _load_suite()
    for scenario in suite["cases"]:
        case = _replay_conversation(scenario, suite["defaults"])
        expected = scenario["expected"]

        assert case.category.value == expected["category"], scenario["id"]
        assert float(case.claim_total) == expected["claim_total"], scenario["id"]
        validation = validate_case(case)
        assert validation.ready, f"{scenario['id']}: {validation.errors}; {validation.missing_fields}"

        state = initial_state(case)
        state["generate_requested"] = True
        result = build_graph(DemoAgentClient()).invoke(
            state,
            {"configurable": {"thread_id": f"case-flow-eval-{scenario['id']}"}},
        )
        draft = result.get("draft_markdown", "")
        assert draft, f"{scenario['id']}: graph returned no draft"
        assert not result.get("validation_errors"), (
            f"{scenario['id']}: {result.get('validation_errors')}"
        )
        actual_authorities = {item["id"] for item in result.get("authorities", [])}
        assert actual_authorities == set(expected["authority_ids"]), (
            f"{scenario['id']}: missing={sorted(set(expected['authority_ids']) - actual_authorities)}, "
            f"unexpected={sorted(actual_authorities - set(expected['authority_ids']))}"
        )
        for fragment in expected.get("draft_contains", []):
            assert fragment in draft, f"{scenario['id']}: expected draft fragment {fragment!r}"
        for fragment in expected.get("draft_must_not_contain", []):
            assert fragment not in draft, f"{scenario['id']}: forbidden draft fragment {fragment!r}"

        # Ensure core verified interview values survive all the way to the final draft.
        assert case.claimant.full_name in draft
        assert case.defendant.full_name in draft
        assert case.defendant.address in draft


def test_incomplete_opening_stays_in_interview_and_never_creates_a_draft():
    case = seed_case_from_opening("קניתי מזגן")
    assert opening_story_missing_parts(case.event_summary) == ["מה השתבש", "כמה שילמתם"]
    assert next_interview_field(case, inferred_completed_fields(case)) == "event_summary"

    state = initial_state(case)
    state["generate_requested"] = True
    result = build_graph(DemoAgentClient()).invoke(
        state,
        {"configurable": {"thread_id": "case-flow-incomplete-opening"}},
    )
    assert not result.get("draft_markdown")
    assert result.get("questions")


def test_claim_over_the_small_claim_limit_is_blocked_before_drafting():
    suite = _load_suite()
    scenario = suite["cases"][0]
    case = _replay_conversation(scenario, suite["defaults"])
    case.damage_items = [
        DamageItem(label="השבת מחיר המוצר", amount=Decimal("40000"), kind="direct")
    ]
    assert not validate_case(case).ready

    state = initial_state(case)
    state["generate_requested"] = True
    result = build_graph(DemoAgentClient()).invoke(
        state,
        {"configurable": {"thread_id": "case-flow-over-limit"}},
    )
    assert not result.get("draft_markdown")
    assert any("תקרת התביעה הקטנה" in item for item in result.get("validation_errors", []))
