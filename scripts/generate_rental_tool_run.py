from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from small_claims_agent.graph import build_graph, initial_state
from small_claims_agent.drafting import compose_claim_markdown
from small_claims_agent.interview import (
    apply_chat_answer,
    confirmation_is_yes,
    seed_case_from_opening,
)
from small_claims_agent.legal import retrieve_authorities_for_case
from small_claims_agent.llm import LLMGateway, controlled_cause_paragraphs
from small_claims_agent.models import CaseData
from small_claims_agent.pdf_export import save_pdf
from small_claims_agent.validators import validate_case, validate_draft


OPENING = (
    "שכרתי דירה בירושלים והעברתי למשכיר פיקדון בסך 5,000 ₪. "
    "בתום השכירות החזרתי את הדירה במצב תקין, אך המשכיר מסרב להחזיר את הפיקדון."
)

INTERVIEW_ANSWERS = (
    ("transaction_date", "1.1.26"),
    ("incident_date", "1.6.26"),
    ("incident_place", "רחוב הדוגמה 10 ירושלים"),
    ("product_or_service_details", "שכירת דירה למגורים"),
    ("defendant.full_name", "יוסף כהן (דמות בדיונית)"),
    ("defendant_role", "משכיר"),
    ("defendant.identifier", "לא ידוע"),
    ("defendant.address", "רחוב הדוגמה 10 ירושלים"),
    ("defendant.phone", "לא ידוע"),
    (
        "prior_attempts",
        "שלחתי למשכיר שתי הודעות בוואטסאפ ביום 2.6.2026 וביום 5.6.2026, "
        "ביקשתי את הפיקדון בחזרה והוא סירב להחזירו",
    ),
    ("damage_items", "השבת הפיקדון - 5000"),
    ("primary_remedy", "החזר כספי"),
    ("alternative_remedy", "אין"),
    ("warranty_details", "לא ידוע"),
    (
        "evidence_confirmation",
        "חוזה שכירות\nאסמכתה להעברת הפיקדון\nתמונות הדירה בעת המסירה\nתכתובת וואטסאפ",
    ),
    ("claimant.full_name", "דנה לוי (דמות בדיונית)"),
    ("claimant.identifier", "000000000"),
    ("claimant.address", "רחוב הבדיקה 5 ירושלים"),
    ("claimant.phone", "0500000000"),
    ("claimant.email", "demo@example.com"),
)


def _preserve_opening_facts(local_case: CaseData, extracted_case: CaseData) -> CaseData:
    """Mirror the app's rule: the LLM may enrich, but never rewrite opening facts."""
    extracted_case.event_summary = local_case.event_summary
    extracted_case.transaction_date = local_case.transaction_date
    extracted_case.incident_date = local_case.incident_date
    extracted_case.incident_place = local_case.incident_place
    extracted_case.warranty_details = local_case.warranty_details
    extracted_case.defendant_role = local_case.defendant_role
    extracted_case.product_or_service_details = local_case.product_or_service_details
    extracted_case.chronology = []
    extracted_case.prior_attempts = []
    extracted_case.damage_items = local_case.damage_items
    if extracted_case.category.value == "unsupported":
        extracted_case.category = local_case.category
    return extracted_case


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_ROOT / ".env")

    client = LLMGateway()
    if not client.available:
        raise RuntimeError("No provider API key is configured; refusing to create a live-tool output")

    graph = build_graph(client)
    local_case = seed_case_from_opening(OPENING)

    # This is the same structured extraction entry point used for the opening chat message.
    opening_state = initial_state(local_case)
    opening_state["last_user_message"] = OPENING
    opening_result = graph.invoke(
        opening_state,
        {"configurable": {"thread_id": f"rental-opening-{uuid.uuid4()}"}},
    )
    if opening_result.get("manual_mode"):
        raise RuntimeError(f"LLM opening extraction fell back: {opening_result.get('model_status')}")
    case = _preserve_opening_facts(
        local_case,
        CaseData.model_validate(opening_result["case"]),
    )

    # Every answer passes through the same local parser and validators as the Streamlit chat.
    accepted_fields: list[str] = []
    for field, answer in INTERVIEW_ANSWERS:
        case = apply_chat_answer(case, field, answer)
        accepted_fields.append(field)

    # Run the same LLM interview consistency checkpoint used before confirmation.
    consistency = client.review_interview(case)
    if consistency.verdict != "pass" or consistency.issues:
        raise RuntimeError(
            "LLM consistency checkpoint requested clarification: "
            + json.dumps(consistency.model_dump(mode="json"), ensure_ascii=False)
        )

    if not confirmation_is_yes("מאשר"):
        raise RuntimeError("Confirmation parser did not accept the confirmation")
    case_data = case.model_dump(mode="python")
    case_data["facts_confirmed"] = True
    case = CaseData.model_validate(case_data)

    validation = validate_case(case)
    if not validation.ready:
        raise RuntimeError(
            "Case validation failed: "
            + json.dumps(
                {"missing_fields": validation.missing_fields, "errors": validation.errors},
                ensure_ascii=False,
            )
        )

    result = None
    rejected_draft_attempts = 0
    for _ in range(3):
        final_state = initial_state(case)
        final_state["generate_requested"] = True
        candidate = graph.invoke(
            final_state,
            {"configurable": {"thread_id": f"rental-draft-{uuid.uuid4()}"}},
        )
        if candidate.get("validation_errors"):
            rejected_draft_attempts += 1
            continue
        used_local_fallback = str(candidate.get("model_status", "")).startswith("local draft fallback")
        was_locally_replaced = any(
            "טיוטת המודל לא עברה" in item for item in candidate.get("warnings", [])
        )
        if used_local_fallback or was_locally_replaced:
            rejected_draft_attempts += 1
            continue
        result = candidate
        break
    if result is None:
        records = retrieve_authorities_for_case(case)
        sections = client.draft_sections(
            case,
            [record.model_dump(mode="json") for record in records],
        )
        sections.cause_of_action = controlled_cause_paragraphs(
            [record.model_dump(mode="json") for record in records],
            case,
        )
        diagnostic_markdown = compose_claim_markdown(case, sections, records)
        diagnostic_issues = validate_draft(
            diagnostic_markdown,
            {record.id for record in records},
            case,
        )
        raise RuntimeError(
            f"No provider-generated draft passed local validation after {rejected_draft_attempts} attempts. "
            + "Latest raw-draft issues: "
            + json.dumps(diagnostic_issues, ensure_ascii=False)
        )

    authority_ids = {item["id"] for item in result.get("authorities", [])}
    draft_issues = validate_draft(result["draft_markdown"], authority_ids, case)
    if draft_issues:
        raise RuntimeError(
            "Final local draft validation failed: "
            + json.dumps(draft_issues, ensure_ascii=False)
        )

    output_path = PROJECT_ROOT / "output" / "pdf" / "tool_run_rental_claim.pdf"
    save_pdf(
        result["draft_markdown"],
        output_path,
        title="טיוטת כתב תביעה קטנה — תרחיש שכירות",
    )
    page_count = len(PdfReader(str(output_path)).pages)

    metadata = {
                "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
                "opening_extraction_model": opening_result.get("model_status"),
                "accepted_interview_fields": len(accepted_fields),
                "consistency_checkpoint": consistency.verdict,
                "case_validation": "pass",
                "category": case.category.value,
                "court_city": case.court_city,
                "claim_total": str(validation.claim_total),
                "estimated_fee": str(validation.filing_fee),
                "authority_ids": sorted(authority_ids),
                "draft_model": result.get("model_status"),
                "rejected_provider_draft_attempts": rejected_draft_attempts,
                "draft_review": result.get("draft_review", {}).get("verdict", "missing"),
                "local_draft_validation": "pass",
                "pdf_pages": page_count,
                "warnings": result.get("warnings", []),
            }
    metadata_path = PROJECT_ROOT / "output" / "examples" / "rental_live_run_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
