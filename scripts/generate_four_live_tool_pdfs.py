from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from small_claims_agent.graph import build_graph, initial_state
from small_claims_agent.interview import apply_chat_answer, confirmation_is_yes, seed_case_from_opening
from small_claims_agent.llm import GeminiAgentClient
from small_claims_agent.models import CaseData
from small_claims_agent.pdf_export import markdown_to_pdf_bytes
from small_claims_agent.validators import validate_case, validate_draft


@dataclass(frozen=True)
class Scenario:
    slug: str
    opening: str
    answers: tuple[tuple[str, str], ...]
    expected_category: str
    expected_court: str
    expected_authorities: frozenset[str]


def _answers(
    *,
    transaction_date: str,
    incident_date: str,
    place: str,
    product: str,
    defendant: str,
    role: str,
    defendant_address: str,
    attempt: str,
    damages: str,
    primary: str,
    alternative: str,
    warranty: str,
    evidence: str,
    claimant: str,
    claimant_id: str,
    claimant_address: str,
    claimant_phone: str,
    claimant_email: str,
) -> tuple[tuple[str, str], ...]:
    return (
        ("transaction_date", transaction_date),
        ("incident_date", incident_date),
        ("incident_place", place),
        ("product_or_service_details", product),
        ("defendant.full_name", defendant),
        ("defendant_role", role),
        ("defendant.identifier", "לא ידוע"),
        ("defendant.address", defendant_address),
        ("defendant.phone", "לא ידוע"),
        ("prior_attempts", attempt),
        ("damage_items", damages),
        ("primary_remedy", primary),
        ("alternative_remedy", alternative),
        ("warranty_details", warranty),
        ("evidence_confirmation", evidence),
        ("claimant.full_name", claimant),
        ("claimant.identifier", claimant_id),
        ("claimant.address", claimant_address),
        ("claimant.phone", claimant_phone),
        ("claimant.email", claimant_email),
    )


SCENARIOS = (
    Scenario(
        slug="restaurant_duplicate_charge",
        opening=(
            "במסעדה חויבתי פעמיים על אותה ארוחה ושילמתי בסך הכול 20,000 שקל "
            "במקום 10,000 שקל. המנהל מסרב לזכות את החיוב העודף."
        ),
        answers=_answers(
            transaction_date="1.2.26",
            incident_date="1.2.26",
            place="מסעדת יוסי ובניו קניון מלחה ירושלים",
            product="ארוחה במסעדה",
            defendant="מסעדת יוסי ובניו (עסק בדיוני)",
            role="נותן שירות",
            defendant_address="קניון מלחה ירושלים",
            attempt=(
                "הגעתי למסעדה ביום 2.2.2026 וביקשתי זיכוי בגין החיוב הכפול, "
                "והמנהל סירב לזכות את החיוב העודף"
            ),
            damages="השבת החיוב הכפול העודף - 10000",
            primary="החזר כספי",
            alternative="אין",
            warranty="לא ידוע",
            evidence="קבלה\nפירוט כרטיס אשראי\nתכתובת עם מנהל המסעדה",
            claimant="נועה ישראלי (דמות בדיונית)",
            claimant_id="000000001",
            claimant_address="רחוב הבדיקה 3 ירושלים",
            claimant_phone="0500000001",
            claimant_email="noa.demo@example.com",
        ),
        expected_category="service_contract",
        expected_court="ירושלים",
        expected_authorities=frozenset({"contracts-remedies-2", "contracts-remedies-9"}),
    ),
    Scenario(
        slug="online_course_cancellation",
        opening=(
            "שילמתי 20,000 שקל עבור קורס הכשרה בהייטק. בהסכם נכתב שאפשר לבטל "
            "בתוך 14 ימים, ביקשתי לבטל אחרי יומיים, אך החברה סירבה להשיב את הכסף "
            "בטענה שנכנסתי לאתר."
        ),
        answers=_answers(
            transaction_date="10.2.26",
            incident_date="12.2.26",
            place="עסקה מקוונת באתר courses-demo.co.il של קורסים דיגיטליים בע״מ",
            product="קורס הכשרה מקוון בהייטק",
            defendant="קורסים דיגיטליים בע״מ (חברה בדיונית)",
            role="נותן שירות",
            defendant_address="רחוב המסחר 12 תל אביב",
            attempt=(
                "שלחתי לחברה הודעת דוא״ל ביום 12.2.2026 והתקשרתי למוקד, "
                "ביקשתי לבטל את הקורס ולקבל החזר והחברה סירבה"
            ),
            damages="השבת התמורה ששולמה - 20000",
            primary="ביטול העסקה והחזר כספי",
            alternative="אין",
            warranty="לא ידוע",
            evidence="הסכם ההרשמה\nאישור תשלום\nצילום תנאי הביטול\nתכתובת דוא״ל",
            claimant="אורי לוי (דמות בדיונית)",
            claimant_id="000000002",
            claimant_address="רחוב הבדיקה 7 תל אביב",
            claimant_phone="0500000002",
            claimant_email="uri.demo@example.com",
        ),
        expected_category="service_contract",
        expected_court="תל אביב-יפו",
        expected_authorities=frozenset({"contracts-remedies-2", "contracts-remedies-9"}),
    ),
    Scenario(
        slug="moving_service_no_show",
        opening=(
            "הזמנתי שירות הובלה למעבר דירה ושילמתי מראש מקדמה מלאה בסך 2,500 שקל. "
            "המובילים לא הגיעו במועד שנקבע והחברה מסרבת להחזיר את המקדמה."
        ),
        answers=_answers(
            transaction_date="1.3.26",
            incident_date="15.3.26",
            place="רחוב הנמל 8 חיפה",
            product="שירות הובלת דירה",
            defendant="הובלות הצפון בע״מ (חברה בדיונית)",
            role="נותן שירות",
            defendant_address="דרך העצמאות 20 חיפה",
            attempt=(
                "התקשרתי לחברה שלוש פעמים ושלחתי הודעת דוא״ל ביום 15.3.2026, "
                "ביקשתי החזר של המקדמה והחברה סירבה"
            ),
            damages="השבת המקדמה - 2500\nהוצאה על הובלה חלופית - 800",
            primary="ביטול העסקה והחזר כספי",
            alternative="אין",
            warranty="לא ידוע",
            evidence="הזמנת שירות\nאישור תשלום המקדמה\nחשבונית הובלה חלופית\nתכתובת דוא״ל",
            claimant="מיכל כהן (דמות בדיונית)",
            claimant_id="000000003",
            claimant_address="רחוב הבדיקה 9 חיפה",
            claimant_phone="0500000003",
            claimant_email="michal.demo@example.com",
        ),
        expected_category="service_contract",
        expected_court="חיפה",
        expected_authorities=frozenset(
            {"contracts-remedies-2", "contracts-remedies-9", "contracts-remedies-10"}
        ),
    ),
    Scenario(
        slug="broken_laptop",
        opening=(
            "קניתי מחשב נייד ב-4,000 שקל. פתחתי את הקופסה בחנות וגיליתי שהמסך "
            "שבור; החנות מסרבת להחליף את המחשב או להשיב את הכסף."
        ),
        answers=_answers(
            transaction_date="1.4.26",
            incident_date="1.4.26",
            place="חנות מחשבי הדרום רחוב רגר 15 באר שבע",
            product="מחשב נייד מדגם DemoBook 14",
            defendant="מחשבי הדרום בע״מ (חברה בדיונית)",
            role="מוכר",
            defendant_address="רחוב רגר 15 באר שבע",
            attempt=(
                "הגעתי לחנות ולדלפק השירות ביום 1.4.2026 וביקשתי להחליף את המחשב השבור, "
                "אך החנות סירבה להחליף אותו או להשיב את הכסף"
            ),
            damages="השבת מחיר המחשב - 4000",
            primary="החלפה",
            alternative="ביטול העסקה והחזר כספי",
            warranty="אחריות לשנה לפי תעודת האחריות",
            evidence="קבלה\nצילום המסך השבור\nתעודת אחריות\nתכתובת עם שירות הלקוחות",
            claimant="רועי אברהם (דמות בדיונית)",
            claimant_id="000000004",
            claimant_address="רחוב הבדיקה 11 באר שבע",
            claimant_phone="0500000004",
            claimant_email="roi.demo@example.com",
        ),
        expected_category="consumer_product",
        expected_court="באר שבע",
        expected_authorities=frozenset(
            {
                "sale-law-11",
                "consumer-warranty-regulations",
                "contracts-remedies-2",
                "contracts-remedies-9",
            }
        ),
    ),
)


def _preserve_opening_facts(local_case: CaseData, extracted_case: CaseData) -> CaseData:
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


def _run_scenario(graph, client: GeminiAgentClient, scenario: Scenario) -> tuple[bytes, dict]:
    local_case = seed_case_from_opening(scenario.opening)
    opening_state = initial_state(local_case)
    opening_state["last_user_message"] = scenario.opening
    opening_result = graph.invoke(
        opening_state,
        {"configurable": {"thread_id": f"{scenario.slug}-opening-{uuid.uuid4()}"}},
    )
    if opening_result.get("manual_mode"):
        raise RuntimeError(f"{scenario.slug}: opening extraction fell back")
    case = _preserve_opening_facts(local_case, CaseData.model_validate(opening_result["case"]))

    for field, answer in scenario.answers:
        case = apply_chat_answer(case, field, answer)

    consistency = client.review_interview(case)
    if consistency.verdict != "pass" or consistency.issues:
        raise RuntimeError(
            f"{scenario.slug}: consistency clarification required: "
            + json.dumps(consistency.model_dump(mode="json"), ensure_ascii=False)
        )

    if not confirmation_is_yes("מאשר"):
        raise RuntimeError(f"{scenario.slug}: confirmation parser failed")
    case_data = case.model_dump(mode="python")
    case_data["facts_confirmed"] = True
    case = CaseData.model_validate(case_data)

    validation = validate_case(case)
    if not validation.ready:
        raise RuntimeError(
            f"{scenario.slug}: case validation failed: "
            + json.dumps(
                {"missing": validation.missing_fields, "errors": validation.errors},
                ensure_ascii=False,
            )
        )
    if case.category.value != scenario.expected_category or case.court_city != scenario.expected_court:
        raise RuntimeError(
            f"{scenario.slug}: unexpected classification/court: {case.category.value}, {case.court_city}"
        )

    result = None
    rejected_attempts = 0
    for _ in range(3):
        final_state = initial_state(case)
        final_state["generate_requested"] = True
        candidate = graph.invoke(
            final_state,
            {"configurable": {"thread_id": f"{scenario.slug}-draft-{uuid.uuid4()}"}},
        )
        used_fallback = str(candidate.get("model_status", "")).startswith("local draft fallback")
        replaced = any("טיוטת Gemini לא עברה" in item for item in candidate.get("warnings", []))
        review_passed = candidate.get("draft_review", {}).get("verdict") == "pass"
        if candidate.get("validation_errors") or used_fallback or replaced or not review_passed:
            rejected_attempts += 1
            continue
        result = candidate
        break
    if result is None:
        raise RuntimeError(f"{scenario.slug}: no native Gemini draft passed in 3 attempts")

    authority_ids = frozenset(item["id"] for item in result.get("authorities", []))
    if authority_ids != scenario.expected_authorities:
        raise RuntimeError(
            f"{scenario.slug}: unexpected RAG authorities: {sorted(authority_ids)}"
        )
    draft_issues = validate_draft(result["draft_markdown"], set(authority_ids), case)
    if draft_issues:
        raise RuntimeError(
            f"{scenario.slug}: local draft validation failed: "
            + json.dumps(draft_issues, ensure_ascii=False)
        )

    pdf_bytes = markdown_to_pdf_bytes(
        result["draft_markdown"],
        title=f"טיוטת כתב תביעה קטנה - {scenario.slug}",
    )
    metadata = {
        "scenario": scenario.slug,
        "opening_model": opening_result.get("model_status"),
        "accepted_interview_fields": len(scenario.answers),
        "consistency": consistency.verdict,
        "case_validation": "pass",
        "category": case.category.value,
        "court_city": case.court_city,
        "claim_total": str(validation.claim_total),
        "estimated_fee": str(validation.filing_fee),
        "authority_ids": sorted(authority_ids),
        "draft_model": result.get("model_status"),
        "rejected_draft_attempts": rejected_attempts,
        "draft_review": result.get("draft_review", {}).get("verdict"),
        "local_draft_validation": "pass",
        "warnings": result.get("warnings", []),
    }
    return pdf_bytes, metadata


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_ROOT / ".env")
    client = GeminiAgentClient()
    if not client.available:
        raise RuntimeError("GEMINI_API_KEY is missing; refusing to create live-tool outputs")
    graph = build_graph(client)

    output_dir = PROJECT_ROOT / "output" / "pdf"
    metadata_dir = PROJECT_ROOT / "output" / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    final_log: list[dict] = []
    for scenario in SCENARIOS:
        pdf_bytes, metadata = _run_scenario(graph, client, scenario)
        output_path = output_dir / f"tool_run_{scenario.slug}.pdf"
        output_path.write_bytes(pdf_bytes)
        reader = PdfReader(str(output_path))
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        if any(
            token in extracted
            for token in ("[PLAINTIFF", "[DEFENDANT", "[INCIDENT_PLACE]", "[JURISDICTION_BASIS]")
        ):
            raise RuntimeError(f"{scenario.slug}: privacy placeholder found in final PDF")
        metadata = {
            **metadata,
            "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
            "pdf_pages": len(reader.pages),
        }
        final_log.append(metadata)
        print(json.dumps(metadata, ensure_ascii=False), flush=True)

    metadata_path = metadata_dir / "live_tool_run_metadata.json"
    metadata_path.write_text(
        json.dumps(final_log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
