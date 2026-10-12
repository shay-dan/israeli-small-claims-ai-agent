from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_UP

from .courts import is_vague_online_place, resolve_jurisdiction
from .models import CaseData, RemedyType


SMALL_CLAIM_LIMIT_ILS = Decimal("39900")
LIMIT_EFFECTIVE_DATE = date(2026, 1, 1)
MINIMUM_FILING_FEE_ILS = Decimal("50")


@dataclass(frozen=True)
class ValidationResult:
    missing_fields: list[str]
    errors: list[str]
    warnings: list[str]
    claim_total: Decimal
    filing_fee: Decimal

    @property
    def ready(self) -> bool:
        return not self.missing_fields and not self.errors


QUESTION_TEXT = {
    "claimant.full_name": "מה שמך המלא כפי שיופיע בכתב התביעה?",
    "claimant.identifier": "מה מספר הזהות של התובע/ת? התשובה נשמרת מקומית ואינה נשלחת למודל.",
    "claimant.address": "מה הכתובת המלאה של התובע/ת? התשובה נשמרת מקומית.",
    "claimant.phone": "מה מספר הטלפון של התובע/ת? התשובה נשמרת מקומית.",
    "defendant.full_name": "מה השם המשפטי המלא של הנתבע/ת או בית העסק?",
    "defendant.address": "מה כתובת המסירה המלאה של הנתבע/ת?",
    "event_summary": "תארו בקצרה מה קרה, בסדר כרונולוגי ובמילים שלכם.",
    "transaction_date": "מתי בוצעה העסקה או נוצרה ההתחייבות?",
    "incident_date": "מתי התגלה הפגם או אירעה ההפרה?",
    "incident_place": "היכן בוצעה העסקה או התרחש המעשה שעליו מבוססת התביעה?",
    "product_or_service_details": "מה בדיוק נרכש או הוזמן?",
    "defendant_role": "מה תפקיד הנתבע בעסקה?",
    "prior_attempts": "אילו ניסיונות עשיתם לפתור את המחלוקת, ומתי?",
    "damage_items": "פרטו כל סכום שאתם מבקשים, בנפרד, ואת הסיבה לו.",
    "evidence_confirmation": "אילו מסמכים או ראיות קיימים, או שאין בידיכם ראיות נוספות?",
}


def calculate_filing_fee(total: Decimal) -> Decimal:
    calculated = (total * Decimal("0.01")).quantize(Decimal("1"), rounding=ROUND_UP)
    return max(MINIMUM_FILING_FEE_ILS, calculated)


def _parsed_case_date(value: str) -> date | None:
    value = value.strip()
    for separator in (".", "/", "-"):
        parts = value.split(separator)
        if len(parts) != 3:
            continue
        try:
            if len(parts[0]) == 4:
                year, month, day = map(int, parts)
            else:
                day, month, year = map(int, parts)
                if year < 100:
                    year += 2000
            return date(year, month, day)
        except ValueError:
            return None
    return None


def validate_case(case: CaseData) -> ValidationResult:
    missing: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    required_values = {
        "claimant.full_name": case.claimant.full_name,
        "claimant.identifier": case.claimant.identifier,
        "claimant.address": case.claimant.address,
        "claimant.phone": case.claimant.phone,
        "defendant.full_name": case.defendant.full_name,
        "defendant.address": case.defendant.address,
        "event_summary": case.event_summary,
        "transaction_date": case.transaction_date,
        "incident_date": case.incident_date,
        "incident_place": case.incident_place,
        "product_or_service_details": case.product_or_service_details,
        "defendant_role": case.defendant_role,
        "jurisdiction_basis": case.jurisdiction_basis,
        "court_city": case.court_city,
    }
    missing.extend(path for path, value in required_values.items() if not str(value).strip())
    if not case.prior_attempts:
        missing.append("prior_attempts")
    if not case.damage_items:
        missing.append("damage_items")
    if not case.evidence and not case.confirmed_no_more_evidence:
        missing.append("evidence_confirmation")
    if case.primary_remedy == RemedyType.UNDECIDED:
        missing.append("primary_remedy")
    if not case.facts_confirmed:
        missing.append("facts_confirmed")

    if case.claimant.legal_form == "company":
        errors.append("חברה בע״מ אינה יכולה להגיש תביעה קטנה כתובעת.")
    if case.claim_total > SMALL_CLAIM_LIMIT_ILS:
        errors.append(
            f"סכום הסעדים הכספיים ({case.claim_total:,.0f} ₪) עולה על תקרת התביעה הקטנה "
            f"({SMALL_CLAIM_LIMIT_ILS:,.0f} ₪, נכון ל-{LIMIT_EFFECTIVE_DATE.isoformat()})."
        )
    if case.claim_total <= 0 and case.damage_items:
        errors.append("סכום התביעה הכספית חייב להיות חיובי.")
    parsed_dates: dict[str, date] = {}
    for field, label, value in (
        ("transaction_date", "מועד העסקה", case.transaction_date),
        ("incident_date", "מועד האירוע", case.incident_date),
    ):
        if value and any(term in value for term in ("לפני", "אתמול", "שבוע שעבר", "חודש שעבר")):
            errors.append(f"{label} אינו תאריך מוחלט ויש לבררו לפני יצירת הטיוטה.")
        parsed = _parsed_case_date(value) if value else None
        if parsed:
            parsed_dates[field] = parsed
            if parsed > date.today():
                errors.append(f"{label} נמצא בעתיד; יש לבדוק את השנה ולתקן את התאריך.")
    if (
        parsed_dates.get("transaction_date")
        and parsed_dates.get("incident_date")
        and parsed_dates["incident_date"] < parsed_dates["transaction_date"]
    ):
        errors.append("מועד האירוע או ההפרה מוקדם ממועד העסקה; יש לבדוק את התאריכים.")
    if case.incident_place.strip() in {"בחנות", "בבית", "בסניף"} or is_vague_online_place(case.incident_place):
        errors.append(
            "מקום העסקה או האירוע כללי מדי; בעסקה פיזית נדרש יישוב ולפי האפשר סניף או כתובת, "
            "ובעסקה מקוונת נדרשים האתר או האפליקציה ושם החברה."
        )
    resolved = resolve_jurisdiction(case)
    if case.incident_place and case.defendant.address and not resolved:
        errors.append("לא ניתן להסיק בבטחה בית משפט מוסמך מהיישובים שנמסרו; יש לדייק את מקום העסקה או מען הנתבע.")
    if resolved and (case.court_city != resolved.court_city or case.jurisdiction_basis != resolved.basis):
        errors.append("פרטי הסמכות המקומית אינם תואמים לעובדות ולמיפוי בתי המשפט.")
    if case.alternative_remedy not in {RemedyType.NONE, RemedyType.UNDECIDED, RemedyType.OTHER}:
        if case.alternative_remedy == case.primary_remedy:
            errors.append("הסעד החלופי זהה לסעד העיקרי.")
    if not case.defendant.identifier:
        warnings.append(
            "מספר הזיהוי של הנתבע אינו ידוע. ניתן להגיש תביעה, אך ללא מספר מזהה עלול להיות קושי "
            "בפתיחת תיק הוצאה לפועל אם פסק הדין לא יקויים."
        )
    if not case.claimant.email:
        warnings.append("לא הוזנה כתובת דוא״ל לתובע/ת.")
    if case.category.value == "unsupported":
        warnings.append("סוג המחלוקת טרם סווג; לא יצורפו הפניות חוק עד לסיווג נתמך.")

    return ValidationResult(
        missing_fields=missing,
        errors=errors,
        warnings=warnings,
        claim_total=case.claim_total,
        filing_fee=calculate_filing_fee(case.claim_total),
    )


def select_questions(missing_fields: list[str], candidate_topics: list[str] | None = None) -> list[str]:
    ordered = list(dict.fromkeys((candidate_topics or []) + missing_fields))
    questions = [QUESTION_TEXT[path] for path in ordered if path in QUESTION_TEXT]
    return questions[:2]


REQUIRED_DRAFT_HEADINGS = (
    "כתב תביעה",
    "פרטי הצדדים",
    "סמכות מקומית",
    "הרקע העובדתי",
    "עילת התביעה",
    "הסעד המבוקש",
    "רשימת נספחים",
)


def validate_draft(markdown: str, allowed_authority_ids: set[str], case: CaseData | None = None) -> list[str]:
    issues = [f"חסרה כותרת נדרשת: {heading}" for heading in REQUIRED_DRAFT_HEADINGS if heading not in markdown]
    privacy_placeholders = re.findall(
        r"\[(?:PLAINTIFF|DEFENDANT)_[A-Z_]+\]|\[(?:INCIDENT_PLACE|JURISDICTION_BASIS)\]",
        markdown,
    )
    if privacy_placeholders:
        issues.append("הטיוטה מכילה מצייני פרטיות שלא שוחזרו לערכים המאומתים.")
    if "[חסר]" in markdown or "TODO" in markdown:
        issues.append("הטיוטה מכילה מציין מקום שלא הושלם.")
    if re.search(r"https?://|www\.", markdown, flags=re.IGNORECASE):
        issues.append("כתב התביעה כולל קישור למקור חיצוני; מקורות מלאים שייכים לדו״ח המקורות הנפרד.")
    if "סוגי התרופות" in markdown:
        issues.append("המונח המשפטי 'תרופות' נוסח באופן עמום; יש לפרט את הסעדים המשפטיים הרלוונטיים.")
    for token in _authority_tokens(markdown):
        if token not in allowed_authority_ids:
            issues.append(f"הטיוטה כוללת מקור משפטי שאינו ברשימה המאושרת: {token}")
    if "משום שכולם" in markdown or "משום שבחנות" in markdown:
        issues.append("נימוק הסמכות המקומית הועתק מתשובה גולמית ואינו משפטי או עובדתי דיו.")
    if "אני רוצה לתבוע" in _section(markdown, "הרקע העובדתי", "עילת התביעה"):
        issues.append("הרקע העובדתי מכיל ניסוח שיחתי גולמי במקום עובדות מעובדות.")
    if any(phrase in _section(markdown, "רשימת נספחים", "הצהרה") for phrase in ("יש לי", "יש ברשותי")):
        issues.append("רשימת הנספחים כוללת תשובת משתמש גולמית.")
    facts = _numbered_or_bulleted_lines(_section(markdown, "הרקע העובדתי", "עילת התביעה"))
    attempts = _numbered_or_bulleted_lines(_section(markdown, "ניסיונות ליישוב המחלוקת", "הסעד המבוקש"))
    if {_normalize_line(line) for line in facts} & {_normalize_line(line) for line in attempts}:
        issues.append("אותו משפט מופיע גם ברקע העובדתי וגם בניסיונות ליישוב המחלוקת.")
    if case:
        if not case.facts_confirmed:
            issues.append("העובדות לא אושרו על ידי המשתמש לפני יצירת הטיוטה.")
        expected_title = f"בית המשפט לתביעות קטנות ב{case.court_city}"
        if expected_title not in markdown:
            issues.append("כותרת בית המשפט אינה תואמת למיפוי הסמכות המקומית.")
        party_section = _section(markdown, "פרטי הצדדים", "סמכות מקומית")
        for value, label in (
            (case.claimant.full_name, "שם התובע/ת"),
            (case.claimant.identifier, "מספר הזיהוי של התובע/ת"),
            (case.claimant.address, "כתובת התובע/ת"),
            (case.claimant.phone, "טלפון התובע/ת"),
            (case.claimant.email, "דוא״ל התובע/ת"),
            (case.defendant.full_name, "שם הנתבע/ת"),
            (case.defendant.identifier, "מספר הזיהוי של הנתבע/ת"),
            (case.defendant.address, "כתובת הנתבע/ת"),
            (case.defendant.phone, "טלפון הנתבע/ת"),
        ):
            if value and value not in party_section:
                issues.append(f"{label} שנקלט בראיון אינו מופיע בפרטי הצדדים.")
        if case.warranty_details and case.warranty_details.strip().rstrip(".") not in markdown:
            issues.append("פרטי האחריות שנקלטו בראיון אינם מופיעים בטיוטה.")
        appendix_section = _section(markdown, "רשימת נספחים", "הצהרה")
        for evidence in case.evidence:
            if evidence.name not in appendix_section:
                issues.append(f"הנספח ״{evidence.name}״ שנקלט בראיון אינו מופיע ברשימת הנספחים.")
        relief_section = _section(markdown, "הסעד המבוקש", "רשימת נספחים")
        for item in case.damage_items:
            matching_lines = [line for line in relief_section.splitlines() if item.label in line]
            if not matching_lines or not any(_line_contains_amount(line, item.amount) for line in matching_lines):
                issues.append(f"רכיב הסכום ״{item.label}״ אינו תואם לסכום שאושר בראיון.")
        total_lines = [
            line for line in relief_section.splitlines() if "סה״כ סעד כספי מבוקש" in line
        ]
        if not total_lines or not any(_line_contains_amount(line, case.claim_total) for line in total_lines):
            issues.append("סכום התביעה הכולל בטיוטה אינו תואם לסכום שאושר בראיון.")
        factual_section = _section(markdown, "הרקע העובדתי", "עילת התביעה")
        description_section = _section(markdown, "תיאור המקרה", "השתלשלות האירועים")
        if case.event_summary:
            if len(description_section.strip()) < 30:
                issues.append("תיאור המקרה חסר או קצר מכדי לשקף את האירוע שנמסר.")
            elif case.product_or_service_details and case.product_or_service_details not in description_section:
                issues.append("המוצר או השירות שנמסר אינו מופיע בתיאור המקרה.")
            if re.search(
                r"(?:^|\s)(?:אני|אנחנו|קניתי|רכשתי|שילמתי|פתחתי|ראיתי|גיליתי|ניסיתי|פניתי|ביקשתי)(?:\s|$)",
                description_section,
            ):
                issues.append("תיאור המקרה מנוסח בגוף ראשון או בשפה שיחתית במקום בניסוח עובדתי רשמי.")
        for required_fact, label in (
            (case.transaction_date, "תאריך העסקה"),
            (case.incident_date, "תאריך האירוע"),
            (case.incident_place, "מקום האירוע"),
        ):
            display_fact = _display_date_for_draft(required_fact)
            if required_fact and required_fact not in factual_section and display_fact not in factual_section:
                issues.append(f"{label} אינו מופיע ברקע העובדתי.")
        if case.primary_remedy == case.alternative_remedy and case.primary_remedy != RemedyType.NONE:
            issues.append("הסעד העיקרי והחלופי אינם מובחנים.")
    return issues


def approved_description_terms(event_summary: str) -> list[str]:
    """Choose stable user-supplied terms for checking that the narrative was retained."""
    words = re.findall(r"[א-ת]{3,}", event_summary)
    ignored = {"אני", "רוצה", "לתבוע", "אותם", "אותו", "אותה"}
    meaningful = [word for word in words if word not in ignored]
    return meaningful[:3]


def _display_date_for_draft(value: str) -> str:
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        year, month, day = value.split("-")
        return f"{int(day)}.{int(month)}.{year}"
    return value


def _section(markdown: str, start_heading: str, end_heading: str) -> str:
    start = markdown.find(start_heading)
    if start < 0:
        return ""
    end = markdown.find(end_heading, start + len(start_heading))
    return markdown[start:end if end >= 0 else None]


def _numbered_or_bulleted_lines(section: str) -> list[str]:
    return [line.strip() for line in section.splitlines() if line.strip().startswith(("- ", "1. ", "2. ", "3. ", "4. ", "5. "))]


def _normalize_line(line: str) -> str:
    import re

    return re.sub(r"[^א-ת0-9]+", " ", re.sub(r"^(?:-|\d+\.)\s*", "", line)).strip()


def _line_contains_amount(line: str, expected: Decimal) -> bool:
    for token in re.findall(r"(?<!\d)\d[\d,\s]*(?!\d)", line):
        digits = re.sub(r"\D", "", token)
        if digits and Decimal(digits) == expected:
            return True
    return False


def _authority_tokens(markdown: str) -> set[str]:
    tokens: set[str] = set()
    marker = "[AUTH:"
    start = 0
    while (index := markdown.find(marker, start)) >= 0:
        end = markdown.find("]", index)
        if end < 0:
            break
        tokens.add(markdown[index + len(marker) : end])
        start = end + 1
    return tokens
