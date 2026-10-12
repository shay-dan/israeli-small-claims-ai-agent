from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


class ClaimCategory(StrEnum):
    CONSUMER_PRODUCT = "consumer_product"
    SERVICE_CONTRACT = "service_contract"
    LANDLORD_TENANT = "landlord_tenant"
    UNSUPPORTED = "unsupported"


class RemedyType(StrEnum):
    UNDECIDED = "undecided"
    NONE = "none"
    REFUND = "refund"
    CANCELLATION = "cancellation"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    OTHER = "other"


class Party(BaseModel):
    full_name: str = ""
    identifier: str = ""
    address: str = ""
    phone: str = ""
    email: str = ""
    legal_form: str = "individual"


class DamageItem(BaseModel):
    label: str
    amount: Decimal = Field(ge=0)
    kind: Literal["direct", "consequential", "non_pecuniary", "other"] = "direct"

    @field_validator("amount", mode="before")
    @classmethod
    def normalize_amount(cls, value: Any) -> Decimal:
        if isinstance(value, str):
            value = value.replace(",", "").replace("₪", "").strip()
        return Decimal(str(value or 0))


class EvidenceItem(BaseModel):
    name: str
    description: str = ""


class Witness(BaseModel):
    full_name: str
    phone: str = ""
    email: str = ""


class CaseData(BaseModel):
    claimant: Party = Field(default_factory=Party)
    defendant: Party = Field(default_factory=lambda: Party(legal_form="company"))
    category: ClaimCategory = ClaimCategory.UNSUPPORTED
    court_city: str = ""
    jurisdiction_basis: str = ""
    transaction_date: str = ""
    incident_date: str = ""
    incident_place: str = ""
    product_or_service_details: str = ""
    warranty_details: str = ""
    defendant_role: str = ""
    event_summary: str = ""
    chronology: list[str] = Field(default_factory=list)
    prior_attempts: list[str] = Field(default_factory=list)
    desired_non_monetary_relief: str = ""
    primary_remedy: RemedyType = RemedyType.UNDECIDED
    alternative_remedy: RemedyType = RemedyType.NONE
    damage_items: list[DamageItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    witnesses: list[Witness] = Field(default_factory=list)
    confirmed_no_more_evidence: bool = False
    prior_small_claims_count_this_year: int | None = Field(default=None, ge=0)
    facts_confirmed: bool = False

    @property
    def claim_total(self) -> Decimal:
        return sum((item.amount for item in self.damage_items), Decimal("0"))


class FieldUpdate(BaseModel):
    path: str = Field(description="Dot path within CaseData")
    value: Any
    evidence_text: str = Field(description="Short quote or paraphrase supporting the update")


class StateExtraction(BaseModel):
    updates: list[FieldUpdate] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    candidate_question_topics: list[str] = Field(default_factory=list)
    inferred_category: ClaimCategory = ClaimCategory.UNSUPPORTED


class AuthorityRecord(BaseModel):
    id: str
    categories: list[ClaimCategory]
    law_name: str
    section: str
    quote_he: str
    source_url: str
    source_kind: Literal["official_primary", "official_guidance", "public_consolidation"]
    excerpt_kind: Literal["quote", "summary"]
    verification_url: str = ""
    effective_date: str
    verified_at: str


class DraftSections(BaseModel):
    factual_background: list[str]
    cause_of_action: list[str]
    resolution_attempts: list[str]
    relief_explanation: str


class DraftReview(BaseModel):
    verdict: Literal["pass", "revise"]
    factual_consistency_issues: list[str] = Field(default_factory=list)
    unsupported_assertions: list[str] = Field(default_factory=list)
    clarity_issues: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)


class AnswerRepair(BaseModel):
    action: Literal["accept", "clarify"]
    repaired_answer: str = ""
    clarification_question: str = ""
    confidence: float = Field(default=0, ge=0, le=1)


class ConsistencyIssue(BaseModel):
    field: str
    message: str
    clarification_question: str


class InterviewConsistencyReview(BaseModel):
    verdict: Literal["pass", "clarify"]
    issues: list[ConsistencyIssue] = Field(default_factory=list)


class ClaimGraphState(TypedDict, total=False):
    case: dict[str, Any]
    messages: list[dict[str, str]]
    last_user_message: str
    generate_requested: bool
    manual_mode: bool
    model_status: str
    missing_fields: list[str]
    validation_errors: list[str]
    warnings: list[str]
    questions: list[str]
    contradictions: list[str]
    authorities: list[dict[str, Any]]
    draft_sections: dict[str, Any]
    draft_markdown: str
    draft_review: dict[str, Any]
    revision_count: int
