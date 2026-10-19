from __future__ import annotations

import json
import re
import unicodedata

from .models import CaseData
from .llm import DraftClarificationReview
from .privacy import redact_text, safe_case_context


CLARIFIABLE_DRAFT_FIELDS = {
    "event_summary",
    "transaction_date",
    "incident_date",
    "incident_place",
    "product_or_service_details",
    "prior_attempts",
    "damage_items",
    "primary_remedy",
    "alternative_remedy",
    "warranty_details",
    "evidence",
}


def sanitize_draft_clarifications(
    review: DraftClarificationReview, case: CaseData, draft: str
) -> list[dict[str, str]]:
    """Keep only capped, privacy-safe questions anchored in both supplied texts."""
    safe_case = safe_case_context(case)
    case_strings = _flatten_strings(safe_case)
    safe_draft = redact_text(draft, case)
    issues: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in review.issues[:3]:
        field = item.field.strip()
        case_excerpt = _clean_excerpt(item.case_excerpt)
        draft_excerpt = _clean_excerpt(item.draft_excerpt)
        question = redact_text(item.clarification_question.strip(), case)
        if field not in CLARIFIABLE_DRAFT_FIELDS or not question:
            continue
        if not _excerpt_is_supported(case_excerpt, case_strings):
            continue
        if not _excerpt_is_supported(draft_excerpt, [safe_draft]):
            continue
        # Redacted party/address placeholders are not useful as user-facing evidence.
        if "[" in case_excerpt or "[" in draft_excerpt or "[" in question:
            continue
        if len(question) > 600 or "[PLAINTIFF_" in question or "[DEFENDANT_" in question:
            continue
        signature = (field, _normalize(case_excerpt), _normalize(draft_excerpt))
        if signature in seen:
            continue
        seen.add(signature)
        issues.append(
            {
                "field": field,
                "case_excerpt": case_excerpt,
                "draft_excerpt": draft_excerpt,
                "clarification_question": question,
            }
        )
    return issues


def _flatten_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _flatten_strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _flatten_strings(item)]
    return [json.dumps(value, ensure_ascii=False, default=str)]


def _clean_excerpt(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())[:240]


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value or "")).casefold().strip()


def _excerpt_is_supported(excerpt: str, sources: list[str]) -> bool:
    normalized = _normalize(excerpt)
    if len(normalized) < 4:
        return False
    return any(normalized in _normalize(source) for source in sources)
