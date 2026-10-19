from __future__ import annotations

import re
import unicodedata

from .models import CaseData


PLACEHOLDERS = {
    "claimant_name": "[PLAINTIFF_NAME]",
    "claimant_id": "[PLAINTIFF_ID]",
    "claimant_address": "[PLAINTIFF_ADDRESS]",
    "claimant_phone": "[PLAINTIFF_PHONE]",
    "claimant_email": "[PLAINTIFF_EMAIL]",
    "defendant_name": "[DEFENDANT_NAME]",
    "defendant_id": "[DEFENDANT_ID]",
    "defendant_address": "[DEFENDANT_ADDRESS]",
    "defendant_phone": "[DEFENDANT_PHONE]",
    "defendant_email": "[DEFENDANT_EMAIL]",
}


_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+972|0)[\d\- ]{8,12}(?!\d)")
_IDENTIFIER_PATTERN = re.compile(r"(?<!\d)\d{9}(?!\d)")
_SELF_NAME_PATTERN = re.compile(
    r"(?:^|[\s,.;])(?:שמי|קוראים\s+לי|השם\s+שלי(?:\s+הוא)?)\s+"
    r"(?P<name>[א-ת][א-ת'׳\-]{1,20}(?:\s+[א-ת][א-ת'׳\-]{1,20}){0,2})"
)
_ADDRESS_PATTERN = re.compile(
    r"(?:^|[\s,.;])(?:אני\s+גר(?:ה)?\s+ב(?:רחוב|שדרות|דרך)?|"
    r"כתובתי(?:\s+היא)?|הכתובת\s+שלי(?:\s+היא)?)\s+"
    r"[א-תA-Za-z0-9][^,.;\n]{1,70}"
)


def free_text_privacy_findings(text: str) -> list[str]:
    """Detect likely personal identifiers before an opening is sent to any LLM provider.

    This deliberately favors a short, explainable deny-list. It does not try to
    identify every Hebrew person name because doing so would also block ordinary
    product and business names. Identity fields are collected later in dedicated
    local-only interview steps.
    """
    value = normalize_text(text)
    findings: list[str] = []
    checks = (
        ("כתובת דוא״ל", _EMAIL_PATTERN),
        ("מספר טלפון", _PHONE_PATTERN),
        ("מספר זהות אפשרי", _IDENTIFIER_PATTERN),
        ("שם פרטי שהוצג כפרט זיהוי", _SELF_NAME_PATTERN),
        ("כתובת מגורים אפשרית", _ADDRESS_PATTERN),
    )
    for label, pattern in checks:
        if pattern.search(value):
            findings.append(label)
    if re.search(r"(?:תעודת\s*זהות|ת[״\"']?ז)\s*[:\-]?\s*\d", value):
        findings.append("מספר זהות אפשרי")
    return list(dict.fromkeys(findings))


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", value).strip()


def alias_map(case: CaseData) -> dict[str, str]:
    values = {
        "claimant_name": case.claimant.full_name,
        "claimant_id": case.claimant.identifier,
        "claimant_address": case.claimant.address,
        "claimant_phone": case.claimant.phone,
        "claimant_email": case.claimant.email,
        "defendant_name": case.defendant.full_name,
        "defendant_id": case.defendant.identifier,
        "defendant_address": case.defendant.address,
        "defendant_phone": case.defendant.phone,
        "defendant_email": case.defendant.email,
    }
    aliases = {normalize_text(value): PLACEHOLDERS[key] for key, value in values.items() if normalize_text(value)}
    if normalize_text(case.incident_place):
        # A place description may include or even equal a business name. Keep
        # the more specific party placeholder in that collision so the quality
        # reviewer does not mistake a privacy artifact for a factual mismatch.
        aliases.setdefault(normalize_text(case.incident_place), "[INCIDENT_PLACE]")
    # Mask claimant name variants because users commonly refer to themselves by
    # first name. Do not split company names: generic words such as "מוצרי חשמל"
    # otherwise corrupt ordinary legal prose during redaction.
    for token in normalize_text(case.claimant.full_name).split():
        if len(token) >= 2 and token not in {"דמות", "בדיונית"}:
            aliases.setdefault(token, PLACEHOLDERS["claimant_name"])
    return aliases


def redact_text(text: str, case: CaseData) -> str:
    redacted = normalize_text(text)
    for alias, placeholder in sorted(alias_map(case).items(), key=lambda item: len(item[0]), reverse=True):
        redacted = re.sub(re.escape(alias), placeholder, redacted, flags=re.IGNORECASE)
    redacted = _EMAIL_PATTERN.sub("[EMAIL]", redacted)
    redacted = _PHONE_PATTERN.sub("[PHONE]", redacted)
    redacted = _IDENTIFIER_PATTERN.sub("[IDENTIFIER]", redacted)
    return redacted


def safe_case_context(case: CaseData) -> dict:
    data = case.model_dump(mode="json")
    for side in ("claimant", "defendant"):
        original = data[side]
        data[side] = {
            # A placeholder means "a value exists but is hidden". Leaving
            # absent optional fields empty prevents the LLM judge from claiming
            # that an unknown ID, phone or email was supplied and then omitted.
            "full_name": PLACEHOLDERS[f"{side}_name"] if normalize_text(original["full_name"]) else "",
            "identifier": PLACEHOLDERS[f"{side}_id"] if normalize_text(original["identifier"]) else "",
            "address": PLACEHOLDERS[f"{side}_address"] if normalize_text(original["address"]) else "",
            "phone": PLACEHOLDERS[f"{side}_phone"] if normalize_text(original["phone"]) else "",
            "email": PLACEHOLDERS[f"{side}_email"] if normalize_text(original["email"]) else "",
            "legal_form": original["legal_form"],
        }
    data["jurisdiction_basis"] = "[JURISDICTION_BASIS]" if data.get("jurisdiction_basis") else ""
    return _redact_nested(data, case)


def _redact_nested(value, case: CaseData):
    if isinstance(value, str):
        return redact_text(value, case)
    if isinstance(value, list):
        return [_redact_nested(item, case) for item in value]
    if isinstance(value, dict):
        return {key: _redact_nested(item, case) for key, item in value.items()}
    return value
