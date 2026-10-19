from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .interview import AnswerAssessment, InterviewAnswerError, apply_chat_answer_with_assessment
from .llm import ModelUnavailable
from .models import CaseData


REPAIRABLE_FIELDS = frozenset(
    {
        "event_summary",
        "transaction_date",
        "incident_date",
        "product_or_service_details",
        "defendant_role",
        "prior_attempts",
        "damage_items",
        "primary_remedy",
        "alternative_remedy",
        "warranty_details",
    }
)

CONSISTENCY_FIELDS = frozenset(
    {
        "transaction_date",
        "incident_date",
        "product_or_service_details",
        "prior_attempts",
        "damage_items",
        "primary_remedy",
        "alternative_remedy",
        "warranty_details",
    }
)


def consistency_issue_is_supported(case: CaseData, field: str) -> bool:
    """Require deterministic evidence before surfacing an LLM consistency concern."""
    if field in {"transaction_date", "incident_date"}:
        try:
            transaction = datetime.strptime(case.transaction_date, "%d.%m.%Y").date()
            incident = datetime.strptime(case.incident_date, "%d.%m.%Y").date()
        except (TypeError, ValueError):
            return False
        return incident < transaction
    if field == "product_or_service_details":
        return not bool(case.product_or_service_details.strip())
    if field == "prior_attempts":
        return not bool(case.prior_attempts)
    if field == "damage_items":
        return not bool(case.damage_items)
    if field in {"primary_remedy", "alternative_remedy"}:
        return (
            case.primary_remedy == case.alternative_remedy
            and case.primary_remedy.value not in {"undecided", "none"}
        )
    # Warranty details may legitimately be unknown; a model-only concern is
    # insufficient to overwrite or re-open this field.
    return False


@dataclass(frozen=True)
class RepairOutcome:
    accepted: bool
    case: CaseData | None = None
    assessment: AnswerAssessment | None = None
    repaired_answer: str = ""
    clarification_question: str = ""
    diagnostic: dict[str, Any] | None = None


def attempt_llm_repair(
    client: Any,
    case: CaseData,
    field: str,
    question: str,
    failed_answers: list[str],
) -> RepairOutcome | None:
    if field not in REPAIRABLE_FIELDS or not getattr(client, "available", False):
        return None
    try:
        proposal = client.repair_answer(field, question, failed_answers[-2:], case)
    except ModelUnavailable as exc:
        return RepairOutcome(accepted=False, diagnostic=exc.diagnostic)

    if proposal.action == "accept" and proposal.confidence >= 0.7 and proposal.repaired_answer.strip():
        try:
            updated, assessment = apply_chat_answer_with_assessment(
                case, field, proposal.repaired_answer
            )
        except InterviewAnswerError:
            return RepairOutcome(
                accepted=False,
                clarification_question=proposal.clarification_question
                or "הבנתי את הכוונה, אבל הערך עדיין לא עבר את הבדיקה המקומית. נסו לנסח אותו מחדש.",
            )
        assessment.source = "model"
        return RepairOutcome(
            accepted=True,
            case=updated,
            assessment=assessment,
            repaired_answer=proposal.repaired_answer,
        )

    return RepairOutcome(
        accepted=False,
        clarification_question=proposal.clarification_question
        or "לא הצלחתי להבין את התשובה בבטחה. נסו לנסח אותה בדרך אחרת.",
    )
