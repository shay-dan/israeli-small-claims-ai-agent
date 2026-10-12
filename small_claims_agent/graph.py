from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from .drafting import compose_claim_markdown
from .legal import classify_category, retrieve_authorities_for_case
from .llm import DemoAgentClient, GeminiAgentClient, ModelUnavailable, controlled_cause_paragraphs
from .models import CaseData, ClaimCategory, ClaimGraphState, DraftReview, DraftSections
from .validators import select_questions, validate_case, validate_draft


ALLOWED_UPDATE_ROOTS = {
    "category",
    "transaction_date",
    "incident_date",
    "incident_place",
    "product_or_service_details",
    "warranty_details",
    "defendant_role",
    "event_summary",
    "chronology",
    "prior_attempts",
    "damage_items",
    "evidence",
    "witnesses",
    "confirmed_no_more_evidence",
    "prior_small_claims_count_this_year",
}


def _ensure_verified_factual_anchors(sections: DraftSections, case: CaseData) -> DraftSections:
    """Add any verified date/place that Gemini shortened or omitted."""
    factual_text = "\n".join(sections.factual_background)
    anchors: list[str] = []
    if case.transaction_date and case.transaction_date not in factual_text:
        anchors.append(f"מועד העסקה שנמסר ואושר הוא {case.transaction_date}.")
    if case.incident_date and case.incident_date not in factual_text:
        anchors.append(f"מועד האירוע או ההפרה שנמסר ואושר הוא {case.incident_date}.")
    if case.incident_place and case.incident_place not in factual_text:
        anchors.append(f"מקום העסקה או האירוע שנמסר ואושר הוא {case.incident_place}.")
    if anchors:
        sections.factual_background = sections.factual_background + anchors
    return sections


def _restore_verified_privacy_placeholders(sections: DraftSections, case: CaseData) -> DraftSections:
    """Restore only state-backed values after Gemini worked on the redacted case."""
    replacements = {
        "[PLAINTIFF_NAME]": case.claimant.full_name,
        "[PLAINTIFF_ID]": case.claimant.identifier,
        "[PLAINTIFF_ADDRESS]": case.claimant.address,
        "[PLAINTIFF_PHONE]": case.claimant.phone,
        "[PLAINTIFF_EMAIL]": case.claimant.email,
        "[DEFENDANT_NAME]": case.defendant.full_name,
        "[DEFENDANT_ID]": case.defendant.identifier,
        "[DEFENDANT_ADDRESS]": case.defendant.address,
        "[DEFENDANT_PHONE]": case.defendant.phone,
        "[DEFENDANT_EMAIL]": case.defendant.email,
        "[INCIDENT_PLACE]": case.incident_place,
        "[JURISDICTION_BASIS]": case.jurisdiction_basis,
    }

    def restore(text: str) -> str:
        for placeholder, verified_value in replacements.items():
            if verified_value:
                text = text.replace(placeholder, verified_value)
        return text

    sections.factual_background = [restore(item) for item in sections.factual_background]
    sections.cause_of_action = [restore(item) for item in sections.cause_of_action]
    sections.resolution_attempts = [restore(item) for item in sections.resolution_attempts]
    sections.relief_explanation = restore(sections.relief_explanation)
    return sections


def _sanitize_draft_review(
    review: DraftReview, case: CaseData, authority_ids: set[str]
) -> DraftReview:
    """Remove reviewer instructions that contradict verified state or allowed law."""

    def supported(message: str) -> bool:
        normalized = " ".join(message.split())
        # Party details are inserted by deterministic code and checked locally
        # before this LLM review. The reviewer sees only privacy placeholders,
        # so a comparison between those placeholders cannot establish a real
        # mismatch and must never be shown as a factual finding.
        if any(
            marker in normalized
            for marker in (
                "[PLAINTIFF_",
                "[DEFENDANT_",
                "[INCIDENT_PLACE]",
                "[JURISDICTION_BASIS]",
            )
        ):
            return False
        if (
            case.prior_attempts
            and "השתלשלות האירועים" in normalized
            and any(word in normalized for word in ("פנייה", "פניות", "סירוב", "סירובה"))
        ):
            # Contact attempts are intentionally rendered in their own required
            # section. Requiring the same events in chronology would contradict
            # the local no-duplication rule and make the claim repetitive.
            return False
        if "גוף שלישי" in normalized:
            # The court-facing template deliberately uses neutral third-person
            # prose ("התובע/ת") instead of conversational first person.
            return False
        if "consumer-warranty-regulations" in authority_ids or "תקנות הגנת הצרכן" in normalized:
            if "תקנות הגנת הצרכן" in normalized and any(
                word in normalized for word in ("הסר", "הסירי", "מחק", "מחקי", "לא חלות")
            ):
                return False
        asks_for_unknown_pii = any(word in normalized for word in ("מזהה", "טלפון", "דוא\"ל", "דוא״ל"))
        if asks_for_unknown_pii and any(word in normalized for word in ("עדכן", "הוסף", "השלם")):
            return False
        return True

    cleaned = DraftReview(
        verdict=review.verdict,
        factual_consistency_issues=[item for item in review.factual_consistency_issues if supported(item)],
        unsupported_assertions=[item for item in review.unsupported_assertions if supported(item)],
        clarity_issues=[item for item in review.clarity_issues if supported(item)],
        revision_instructions=[item for item in review.revision_instructions if supported(item)],
    )
    if not (
        cleaned.factual_consistency_issues
        or cleaned.unsupported_assertions
        or cleaned.clarity_issues
        or cleaned.revision_instructions
    ):
        cleaned.verdict = "pass"
    return cleaned


def _merge_updates(case: CaseData, updates: list[dict[str, Any]]) -> CaseData:
    data = case.model_dump(mode="python")
    for update in updates:
        path = str(update.get("path", ""))
        root = path.split(".", 1)[0]
        if root not in ALLOWED_UPDATE_ROOTS or "." in path:
            continue
        if root == "category" and case.category != ClaimCategory.UNSUPPORTED:
            continue
        value = _normalize_model_value(root, update.get("value"))
        if value in (None, "", []):
            continue
        candidate = deepcopy(data)
        candidate[root] = value
        try:
            # Accept each update independently. One malformed Gemini field must
            # not discard valid fields or terminate the interview.
            CaseData.model_validate(candidate)
        except (ValidationError, TypeError, ValueError):
            continue
        data = candidate
    return CaseData.model_validate(data)


def _normalize_model_value(root: str, value: Any) -> Any:
    if root in {"chronology", "prior_attempts"}:
        entries = value if isinstance(value, list) else [value]
        normalized: list[str] = []
        for entry in entries:
            if isinstance(entry, str):
                text = entry
            elif isinstance(entry, dict):
                text = next(
                    (entry.get(key) for key in ("description", "text", "event", "value") if entry.get(key)),
                    "",
                )
            else:
                text = ""
            if isinstance(text, str) and text.strip():
                normalized.append(" ".join(text.split()))
        return normalized
    if root == "damage_items":
        entries = value if isinstance(value, list) else [value]
        normalized_items: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("amount") in (None, ""):
                continue
            label = entry.get("label") or entry.get("description") or entry.get("name") or "רכיב נזק"
            normalized_items.append(
                {"label": str(label), "amount": entry["amount"], "kind": entry.get("kind", "direct")}
            )
        return normalized_items
    if root == "evidence":
        entries = value if isinstance(value, list) else [value]
        return [
            {
                "name": str(entry.get("name") or entry.get("description") or "ראיה שנמסרה"),
                "description": str(entry.get("details") or ""),
            }
            for entry in entries
            if isinstance(entry, dict)
        ]
    if root == "witnesses":
        return value if isinstance(value, list) else [value]
    if root in {"event_summary", "transaction_date", "incident_date", "incident_place", "product_or_service_details", "warranty_details", "defendant_role"}:
        return value if isinstance(value, str) else None
    return value


def build_graph(client: GeminiAgentClient | DemoAgentClient | None = None):
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is not installed. Run pip install -r requirements.txt") from exc

    client = client or GeminiAgentClient()

    def extract_node(state: ClaimGraphState) -> dict:
        case = CaseData.model_validate(state.get("case") or {})
        user_text = state.get("last_user_message", "")
        if not user_text:
            return {"case": case.model_dump(mode="json"), "model_status": "idle"}
        try:
            extraction = client.extract(user_text, case)
            case = _merge_updates(case, [item.model_dump() for item in extraction.updates])
            if case.category == ClaimCategory.UNSUPPORTED:
                case.category = extraction.inferred_category
            return {
                "case": case.model_dump(mode="json"),
                "contradictions": extraction.contradictions,
                "questions": extraction.candidate_question_topics,
                "model_status": getattr(client, "model_name", "unknown"),
                "manual_mode": isinstance(client, DemoAgentClient),
            }
        except ModelUnavailable as exc:
            fallback = DemoAgentClient()
            extraction = fallback.extract(user_text, case)
            if case.category == ClaimCategory.UNSUPPORTED:
                case.category = extraction.inferred_category
            return {
                "case": case.model_dump(mode="json"),
                "model_status": str(exc),
                "manual_mode": True,
                "warnings": ["Gemini אינו זמין; הנתונים נשמרו והמערכת עברה למצב ידני."],
            }

    def validate_node(state: ClaimGraphState) -> dict:
        case = CaseData.model_validate(state.get("case") or {})
        result = validate_case(case)
        existing = state.get("questions", [])
        return {
            "missing_fields": result.missing_fields,
            "validation_errors": result.errors,
            "warnings": list(dict.fromkeys(state.get("warnings", []) + result.warnings)),
            "questions": select_questions(result.missing_fields, existing),
        }

    def route_after_validation(state: ClaimGraphState) -> str:
        if state.get("validation_errors") or state.get("missing_fields") or not state.get("generate_requested"):
            return "clarify"
        return "retrieve"

    def clarify_node(state: ClaimGraphState) -> dict:
        return {"draft_markdown": state.get("draft_markdown", "")}

    def retrieve_node(state: ClaimGraphState) -> dict:
        case = CaseData.model_validate(state.get("case") or {})
        case.category = classify_category(case.event_summary, case.category)
        authorities = retrieve_authorities_for_case(case)
        return {
            "case": case.model_dump(mode="json"),
            "authorities": [record.model_dump(mode="json") for record in authorities],
        }

    def draft_node(state: ClaimGraphState) -> dict:
        case = CaseData.model_validate(state["case"])
        authorities = state.get("authorities", [])
        active_client = client
        try:
            sections = active_client.draft_sections(case, authorities)
            status = getattr(active_client, "model_name", "unknown")
        except Exception as exc:
            sections = DemoAgentClient().draft_sections(case, authorities)
            status = f"local draft fallback: {type(exc).__name__}"
        sections = _restore_verified_privacy_placeholders(sections, case)
        sections = _ensure_verified_factual_anchors(sections, case)
        from .models import AuthorityRecord

        records = [AuthorityRecord.model_validate(item) for item in authorities]
        # Legal conclusions are deterministic and authority-bound even when Gemini
        # supplies the narrative prose.
        sections.cause_of_action = controlled_cause_paragraphs(authorities, case)
        markdown = compose_claim_markdown(case, sections, records)
        return {
            "draft_sections": sections.model_dump(mode="json"),
            "draft_markdown": markdown,
            "model_status": status,
        }

    def draft_check_node(state: ClaimGraphState) -> dict:
        allowed = {item["id"] for item in state.get("authorities", [])}
        case = CaseData.model_validate(state.get("case") or {})
        issues = validate_draft(state.get("draft_markdown", ""), allowed, case)
        if not issues:
            return {"validation_errors": []}

        # A malformed model narrative must not strand the user after a complete
        # interview. Recompose once with the deterministic local drafter.
        from .models import AuthorityRecord

        authorities = state.get("authorities", [])
        sections = DemoAgentClient().draft_sections(case, authorities)
        sections = _restore_verified_privacy_placeholders(sections, case)
        sections = _ensure_verified_factual_anchors(sections, case)
        sections.cause_of_action = controlled_cause_paragraphs(authorities, case)
        records = [AuthorityRecord.model_validate(item) for item in authorities]
        fallback_markdown = compose_claim_markdown(case, sections, records)
        fallback_issues = validate_draft(fallback_markdown, allowed, case)
        return {
            "draft_markdown": fallback_markdown,
            "draft_sections": sections.model_dump(mode="json"),
            "validation_errors": fallback_issues,
            "warnings": list(
                dict.fromkeys(
                    state.get("warnings", [])
                    + ["טיוטת Gemini לא עברה את הבדיקה המקומית, ולכן נוצרה טיוטה מקומית תקינה."]
                )
            ),
        }

    def route_after_draft_check(state: ClaimGraphState) -> str:
        return "clarify" if state.get("validation_errors") else "review"

    def review_node(state: ClaimGraphState) -> dict:
        case = CaseData.model_validate(state["case"])
        allowed = {item["id"] for item in state.get("authorities", [])}
        try:
            review = client.review(case, state["draft_markdown"], allowed)
        except Exception:
            review = DemoAgentClient().review(case, state["draft_markdown"], allowed)
        review = _sanitize_draft_review(review, case, allowed)
        # The MVP records reviewer feedback but never lets it overwrite facts.
        if review.verdict == "revise" and state.get("revision_count", 0) < 1:
            return {
                "draft_review": review.model_dump(mode="json"),
                "revision_count": 1,
                "warnings": list(dict.fromkeys(state.get("warnings", []) + [
                    "בקרת האיכות הציעה תיקון; ההערות מוצגות למשתמש ולא משנות עובדות אוטומטית."
                ])),
            }
        return {"draft_review": review.model_dump(mode="json")}

    builder = StateGraph(ClaimGraphState)
    builder.add_node("extract", extract_node)
    builder.add_node("validate", validate_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("draft", draft_node)
    builder.add_node("draft_check", draft_check_node)
    builder.add_node("review", review_node)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "validate")
    builder.add_conditional_edges("validate", route_after_validation, {"clarify": "clarify", "retrieve": "retrieve"})
    builder.add_edge("clarify", END)
    builder.add_edge("retrieve", "draft")
    builder.add_edge("draft", "draft_check")
    builder.add_conditional_edges("draft_check", route_after_draft_check, {"clarify": "clarify", "review": "review"})
    builder.add_edge("review", END)
    return builder.compile(checkpointer=InMemorySaver())


def initial_state(case: CaseData | None = None) -> ClaimGraphState:
    case = deepcopy(case) if case else CaseData()
    return {
        "case": case.model_dump(mode="json"),
        "messages": [],
        "last_user_message": "",
        "generate_requested": False,
        "manual_mode": False,
        "model_status": "idle",
        "missing_fields": [],
        "validation_errors": [],
        "warnings": [],
        "questions": [],
        "contradictions": [],
        "authorities": [],
        "draft_sections": DraftSections(
            factual_background=[], cause_of_action=[], resolution_attempts=[], relief_explanation=""
        ).model_dump(mode="json"),
        "draft_markdown": "",
        "draft_review": {},
        "revision_count": 0,
    }
