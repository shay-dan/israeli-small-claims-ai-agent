from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel

from .courts import apply_jurisdiction, is_remote_transaction_place, is_vague_online_place, locality_from_text
from .legal import classify_category
from .models import CaseData, ClaimCategory, DamageItem, EvidenceItem, Party, RemedyType
from .validators import QUESTION_TEXT, calculate_filing_fee, validate_case


WELCOME_MESSAGE = (
    "שלום, אני אעזור לך להכין טיוטת כתב תביעה קטנה באמצעות ראיון קצר. "
    "נתחיל בתיאור המקרה בלבד: מה קרה, מה קנית או איזה שירות קיבלת, כמה שילמת ומה השתבש? "
    "בשלב הזה אין לכתוב שמות, מספרי זהות, כתובות או טלפונים - אבקש אותם בנפרד ואשמור אותם מקומית."
)

SAMPLE_OPENING = (
    "קניתי מזגן בסכום 3,200 ₪ בחנות X. המזגן התקלקל אחרי שבועיים. "
    "פניתי לחנות 3 פעמים והם מסרבים להחזיר כסף או לתקן. אני רוצה לתבוע אותם."
)

INTERVIEW_ORDER = (
    "event_summary",
    "transaction_date",
    "incident_date",
    "incident_place",
    "product_or_service_details",
    "defendant.full_name",
    "defendant_role",
    "defendant.identifier",
    "defendant.address",
    "defendant.phone",
    "prior_attempts",
    "damage_items",
    "primary_remedy",
    "alternative_remedy",
    "warranty_details",
    "evidence_confirmation",
    "claimant.full_name",
    "claimant.identifier",
    "claimant.address",
    "claimant.phone",
    "claimant.email",
)

OPTIONAL_FIELDS = {
    "defendant.identifier",
    "defendant.phone",
    "claimant.email",
    "warranty_details",
    "alternative_remedy",
}

QUESTION_OVERRIDES = {
    "transaction_date": "מה תאריך העסקה המדויק כפי שמופיע בקבלה או בהסכם? אפשר לכתוב 1.1.26, 01/01/26 או 01.01.2026.",
    "incident_date": "מתי התגלה הפגם או אירעה ההפרה? אפשר לכתוב 10.1.26, 10/01/26 או 10.01.2026.",
    "incident_place": (
        "היכן בוצעה העסקה או התרחש המעשה? בעסקה פיזית ציינו יישוב וסניף או כתובת. "
        "בעסקה מקוונת ציינו את האתר או האפליקציה ואת שם החברה; בעסקה טלפונית אפשר לכתוב ״עסקה טלפונית״."
    ),
    "product_or_service_details": "מה בדיוק נרכש או הוזמן? ציינו סוג מוצר או שירות ופרטים מזהים שיש בקבלה, כגון דגם, אם ידועים.",
    "defendant.full_name": "מה השם המשפטי המדויק של הנתבע/ת או בית העסק, כפי שמופיע בקבלה או בחשבונית?",
    "defendant.address": "מה כתובת המסירה המלאה של הנתבע/ת? אם זו הכתובת שכבר ציינתם כמקום העסקה, אפשר לכתוב ״אותה כתובת״.",
    "defendant_role": "מה תפקיד הנתבע בעסקה: מוכר, יצרן, יבואן, נותן שירות או תפקיד אחר?",
    "defendant.identifier": "מה מספר הזהות או מספר החברה של הנתבע/ת? אם אינו ידוע, כתבו ״לא ידוע״.",
    "defendant.phone": "האם ידוע לכם מספר הטלפון של הנתבע/ת? אם לא, כתבו ״לא ידוע״.",
    "prior_attempts": (
        "פרטו את ניסיונות פתרון המחלוקת: כמה פעמים פניתם, באילו תאריכים או בערך מתי, "
        "באיזה אמצעי ומה הייתה התגובה."
    ),
    "damage_items": (
        "פרטו את כל הסכומים הכספיים המבוקשים ואת הסיבה לכל סכום, שורה נפרדת לכל רכיב. "
        "לדוגמה: השבת מחיר המוצר - 3200 ₪"
    ),
    "primary_remedy": "מה הסעד העיקרי המבוקש: החזר כספי וביטול העסקה, תיקון, החלפה או סעד אחר?",
    "alternative_remedy": "אם בית המשפט לא ייתן את הסעד העיקרי, מה הסעד החלופי המבוקש? אפשר לכתוב ״אין״.",
    "warranty_details": (
        "האם קיימת אחריות למוצר או לשירות, ולמי פניתם מכוחה? ציינו תקופה או מסמך אם ידועים, "
        "או כתבו ״לא ידוע״."
    ),
    "evidence_confirmation": (
        "אילו מסמכים, ראיות או עדים קיימים? כתבו כל פריט בשורה נפרדת, "
        "או ״אין ראיות נוספות״."
    ),
    "claimant.email": "מה כתובת הדוא״ל שלכם? אם אינכם רוצים למסור או שאין כתובת, כתבו ״אין״.",
}

DATE_PATTERN = re.compile(r"^(?:\d{1,2}[./-]\d{1,2}[./-](?:\d{2}|\d{4})|\d{4}-\d{1,2}-\d{1,2})$")
RELATIVE_DATE_TERMS = ("לפני", "אתמול", "שלשום", "שבוע שעבר", "חודש שעבר", "לאחרונה")
VAGUE_PLACES = {"בחנות", "בבית", "באינטרנט", "בסניף", "בעסק", "אצלי", "שם"}
CITY_TYPO_HINTS = {
    "רמת גם": "רמת גן",
    "תל אביב יפו": "תל אביב-יפו",
    "באר שבעע": "באר שבע",
}
REFERENCE_ONLY = {
    "כתבתי",
    "כתבתי קודם",
    "כתבתי כבר",
    "כבר כתבתי",
    "אמרתי",
    "אמרתי קודם",
    "אמרתי לך",
    "כבר אמרתי",
    "אמרתי כבר",
    "כמו שכתבתי",
    "השתמש במה שכתבתי",
    "השתמשי במה שכתבתי",
    "השתמש",
    "השתמשי",
    "שם",
    "אותה כתובת",
    "אותה",
    "אותו מען",
    "אותו דבר",
    "כמו קודם",
    "הכתובת שכתבתי",
    "כתובת העסק",
    "כמו מקום",
    "כמו מקום העסקה",
    "כמו הכתובת הקודמת",
}
PRODUCT_WORDS = ("מזגן", "מקרר", "מכונת כביסה", "מדיח", "טלוויזיה", "מחשב", "טלפון", "ספה", "מוצר")
SERVICE_CONTEXT_WORDS = (
    "מסעדה",
    "ארוחה",
    "מלון",
    "מוסך",
    "מספרה",
    "הובלה",
    "ניקיון",
    "אירוע",
    "צילום",
)
RENTAL_CONTEXT_WORDS = ("שכרתי", "שכירות", "שוכר", "משכיר", "דירה", "פיקדון")
HEBREW_KEYBOARD_MAP = str.maketrans(
    "qwertyuiop[]asdfghjkl;'zxcvbnm,./",
    "/'קראטוןםפ][שדגכעיחלךף,זסבהנמצתץ.",
)
DAMAGE_CONFIRMATIONS = (
    "בוודאות",
    "זה הסך הכולל",
    "זה הסכום הכולל",
    "כן זה הסכום",
    "כן, זה הסכום",
    "מאשר את הפירוט",
)
OPENING_PROBLEM_TERMS = (
    "התקלקל",
    "תקלה",
    "פגם",
    "לא עובד",
    "לא פועל",
    "נשבר",
    "לא קיבלתי",
    "לא סופק",
    "לא בוצע",
    "לא הגיע",
    "לא הגיעו",
    "לא הופיע",
    "לא הופיעו",
    "לא התייצב",
    "לא התייצבו",
    "הפר",
    "סירב",
    "מסרב להחזיר",
    "מסרבת להחזיר",
    "מסרבים להחזיר",
    "נזק",
    "רעש",
    "מרעיש",
    "לא מקרר",
    "לא מחמם",
    "נזילה",
    "דליפה",
    "נשרף",
    "קצר",
    "התפוצץ",
    "התפוצצה",
    "התפנצ'ר",
    "התפנצ׳ר",
    "התפנצ'רה",
    "התפנצ׳רה",
    "נקרע",
    "נקרעה",
    "יצא האוויר",
    "התחרטתי",
    "ביקשתי ביטול",
    "ביקשתי לבטל",
    "לא אפשרו",
    "לא איפשרו",
    "לא מאפשרים",
    "לא החזירו",
    "לא החזיר",
    "לא החזירה",
    "לא מחזירים",
    "לא יכול לקבל",
    "לא יכולה לקבל",
    "סירבו לבטל",
    "סירבו להחזיר",
    "הטעו אותי",
    "לא גילוי",
    "עקץ",
    "נעקצתי",
    "חיוב כפול",
    "חויבתי פעמיים",
    "חוייבתי פעמיים",
    "חייב אותי פעמיים",
    "חייבו אותי פעמיים",
    "חובתי פעמיים",
    "נגבה פעמיים",
    "תשלום נוסף",
    "בלי סיבה",
    "ללא סיבה",
    "שירות מזלזל",
    "ניצל את התמימות",
)

# Adjectival damage descriptions are common in natural Hebrew ("המסך שבור",
# "המוצר פגום") but do not share the verbal forms above ("נשבר",
# "התקלקל").  Keep this product-agnostic: the subject extractor determines
# what was purchased, while this pattern only establishes that a defect was
# actually described.
OPENING_DAMAGE_DESCRIPTION_PATTERN = re.compile(
    r"(?:"
    r"שבור(?:ה|ים|ות)?|"
    r"סדוק(?:ה|ים|ות)?|"
    r"פג(?:ום|ומה|ומים|ומות)|"
    r"מקולקל(?:ת|ים|ות)?|"
    r"הרוס(?:ה|ים|ות)?|"
    r"מע(?:וך|וכה|וכים|וכות)|"
    r"קרוע(?:ה|ים|ות)?|"
    r"ניזוק(?:ה|ו|וּ)?|"
    r"שבר\s+(?:ב|במסך|במוצר)|"
    r"נזק\s+(?:ב|ל)"
    r")"
)


class InterviewAnswerError(ValueError):
    pass


class AnswerAssessment(BaseModel):
    path: str
    normalized_value: Any
    confidence: float
    source: Literal["user", "opening", "model", "derived"] = "user"
    status: Literal["accepted", "needs_clarification"] = "accepted"


def seed_case_from_opening(text: str) -> CaseData:
    """Capture only deterministic facts; never reuse the raw story as chronology or contact history."""
    cleaned = " ".join(text.split()).strip()
    case = CaseData(event_summary=cleaned)
    case.category = classify_category(cleaned)
    case.product_or_service_details = extract_purchased_subject(cleaned)
    amounts = _amounts_from_text(cleaned)
    if len(amounts) == 1:
        case.damage_items = [
            DamageItem(label="סכום שנמסר בתיאור הראשוני", amount=amounts[0], kind="direct")
        ]
    return case


def normalize_opening_keyboard_layout(text: str) -> tuple[str, bool]:
    """Convert a recognizable Hebrew sentence typed with an English keyboard."""
    if re.search(r"[א-ת]", text) or not re.search(r"[A-Za-z]", text):
        return text, False
    converted = text.lower().translate(HEBREW_KEYBOARD_MAP)
    cues = ("קניתי", "רכשתי", "שילמתי", "הזמנתי", "התקלקל", *PRODUCT_WORDS)
    hebrew_count = len(re.findall(r"[א-ת]", converted))
    if hebrew_count >= 5 and any(cue in converted for cue in cues):
        return converted, True
    return text, False


def extract_purchased_subject(text: str) -> str:
    """Extract only a verbatim product/service phrase from common purchase syntax."""
    normalized = " ".join(text.split()).strip()
    patterns = (
        r"(?:קניתי|רכשתי|הזמנתי)\s+(.+?)(?=\s+ב\s*[\d,]+|\s+בסכום|\s+במחיר|\s+ואחרי|\s+אחרי|[,.;]|$)",
        r"שכרתי\s+(.+?)(?=\s+ו?שילמתי|\s+בסכום|\s+במחיר|\s+אבל|\s+ואחרי|\s+אחרי|[,.;]|$)",
        r"שילמתי\s+(?:[\d,]+(?:\.\d+)?\s*(?:₪|שקלים?|שקל|ש[״\"]?ח)?\s*)?על\s+(.+?)"
        r"(?=\s+(?:ואמרו|והבטיחו|והוסבר|ואחרי|אבל|לאחר|אחרי)|[,.;]|$)",
        r"(?:מה\s+שנרכש\s+(?:זה|הוא)|מה\s+שהזמנתי\s+(?:זה|הוא)|השירות\s+הוא)\s+(.+?)"
        r"(?=\s+(?:ואז|אבל|ואחרי|לאחר|אחרי)|[,.;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" -:,.'\"״׳")
            if 2 <= len(candidate) <= 120 and re.search(r"[א-ת]", candidate):
                return candidate
    return next((product for product in PRODUCT_WORDS if product in normalized), "")


def opening_story_is_complete(text: str) -> bool:
    """Require an actual incident and a price (or explicit lack of one)."""
    return not opening_story_missing_parts(text)


def opening_story_missing_parts(text: str) -> list[str]:
    """Return the specific opening components that still need user input."""
    normalized = " ".join(text.split()).lower()
    has_subject = bool(extract_purchased_subject(normalized)) or any(
        word in normalized
        for word in (
            "שירות",
            "עסקה",
            "הזמנה",
            "קורס",
            "לימודים",
            "מנוי",
            *SERVICE_CONTEXT_WORDS,
            *RENTAL_CONTEXT_WORDS,
        )
    )
    has_problem = (
        any(term in normalized for term in OPENING_PROBLEM_TERMS)
        or bool(OPENING_DAMAGE_DESCRIPTION_PATTERN.search(normalized))
        or bool(
        re.search(
            r"(?:הפסיק|הפסיקה|חדל|חדלה)\s+ל?(?:עבוד|פעול)|"
            r"(?:לא|אינו|אינה)\s+(?:עובד|עובדת|פועל|פועלת|תקין|תקינה)|"
            r"(?:התחיל|התחילה)\s+(?:לעשות\s+)?רעש|"
            r"(?:לאחר|אחרי).{0,30}(?:כשל|הפסיק|נשבר|התקלקל)|"
            r"(?:ביקשתי|דרשתי)\s+(?:לבטל|ביטול|החזר)|"
            r"(?:לא|אינם?|אינן)\s+(?:אפשרו|איפשרו|מאפשרים|מחזירים|החזירו)|"
            r"(?:סירב|סירבה|סירבו|מסרב|מסרבת|מסרבים).{0,30}(?:לבטל|להחזיר|החזר)|"
            r"(?:אין|לא\s+(?:נותנים|נתנו|מוכנים)).{0,20}(?:החזר|זיכוי|תיקון|החלפה)|"
            r"(?:ניסיתי|ניסתה|ניסינו)\s+להחזיר|"
            r"(?:התחרטתי|התחרטה|התחרטנו)",
            normalized,
        )
        )
    )
    has_amount = bool(_amounts_from_text(normalized)) or any(
        phrase in normalized for phrase in ("לא זוכר כמה", "לא יודעת כמה", "הסכום לא ידוע")
    )
    missing: list[str] = []
    if not has_subject:
        missing.append("מה נרכש או איזה שירות הוזמן")
    if not has_problem:
        missing.append("מה השתבש")
    if not has_amount:
        missing.append("כמה שילמתם")
    return missing


def opening_clarification(text: str) -> str:
    missing = opening_story_missing_parts(text)
    if not missing:
        return ""
    if len(missing) == 1:
        missing_text = missing[0]
    else:
        missing_text = ", ".join(missing[:-1]) + " ו" + missing[-1]
    return (
        f"קלטתי את שאר התיאור, ועדיין חסר: {missing_text}. "
        "אפשר להשלים רק את הפרט החסר במשפט קצר; אין צורך לחזור על הכול."
    )


def inferred_completed_fields(case: CaseData) -> set[str]:
    completed = {"event_summary"} if opening_story_is_complete(case.event_summary) else set()
    if _is_exact_date(case.transaction_date):
        completed.add("transaction_date")
    if _is_exact_date(case.incident_date):
        completed.add("incident_date")
    if case.incident_place and _is_specific_place(case.incident_place):
        completed.add("incident_place")
    if case.product_or_service_details:
        completed.add("product_or_service_details")
    return completed


def next_interview_field(case: CaseData, completed_fields: set[str]) -> str | None:
    for field in INTERVIEW_ORDER:
        if field not in completed_fields:
            return field
    missing = set(validate_case(case).missing_fields)
    for field in INTERVIEW_ORDER:
        if field in missing:
            return field
    return None


def question_for(field: str, case: CaseData | None = None) -> str:
    if field == "event_summary":
        existing = case.event_summary if case else ""
        return opening_clarification(existing) or QUESTION_TEXT["event_summary"]
    if field == "damage_items" and case and case.damage_items:
        known = ", ".join(f"{item.amount:,.0f} ₪" for item in case.damage_items)
        return (
            f"מהתיאור הראשון זיהיתי סכום של {known}. פרטו איזה סכום אתם מבקשים מבית המשפט ומה הסיבה לו. "
            "אם זהו חיוב כולל שרק חלק ממנו עודף, כתבו רק את החלק העודף, למשל ״החזר חיוב כפול 10000״. "
            "אפשר גם לכתוב ״מחיר מוצר 3200, עוגמת נפש 800״."
        )
    if field == "prior_attempts" and case:
        opening_attempt = extract_prior_attempt_from_opening(case.event_summary, case)
        if opening_attempt:
            from .drafting import formalize_resolution_attempt

            formal_attempt = formalize_resolution_attempt(opening_attempt, case)
            return (
                f"בתיאור הראשון כבר זיהיתי ניסיון פתרון: {formal_attempt} "
                "אם זה מדויק, כתבו ״השתמש במה שכתבתי״. אם חסר פרט, ציינו רק מה לתקן או להוסיף, "
                "למשל מתי פניתם או כמה פעמים."
            )
    if field == "evidence_confirmation" and case and case.evidence:
        known = ", ".join(item.name for item in case.evidence)
        return (
            f"כבר זיהיתי את הראיות הבאות: {known}. אילו מסמכים, ראיות או עדים נוספים קיימים? "
            "אם אין נוספים, כתבו ״אין ראיות נוספות״."
        )
    if field in QUESTION_OVERRIDES:
        return QUESTION_OVERRIDES[field]
    return QUESTION_TEXT[field]


def apply_chat_answer(case: CaseData, field: str, answer: str) -> CaseData:
    updated, _ = apply_chat_answer_with_assessment(case, field, answer)
    return updated


def apply_chat_answer_with_assessment(
    case: CaseData, field: str, answer: str
) -> tuple[CaseData, AnswerAssessment]:
    raw_answer = answer.strip()
    normalized = " ".join(raw_answer.split())
    if not normalized:
        raise InterviewAnswerError("לא התקבלה תשובה. אפשר לנסות שוב במשפט קצר.")

    data = case.model_dump(mode="python")
    confidence = 1.0
    if field.startswith("claimant.") or field.startswith("defendant."):
        side, attribute = field.split(".", 1)
        party = Party.model_validate(data[side])
        normalized = _strip_reference_suffix(normalized)
        if normalized in REFERENCE_ONLY or not normalized:
            if field == "defendant.address" and case.incident_place:
                if is_remote_transaction_place(case.incident_place):
                    raise InterviewAnswerError(
                        "מקום העסקה הוא ערוץ מקוון או טלפוני, ולכן הוא אינו כתובת למסירת כתב תביעה. "
                        "כתבו את הכתובת הפיזית הרשומה של הנתבע/ת או בית העסק."
                    )
                normalized = _address_from_prior_place(case)
            elif field == "defendant.full_name":
                normalized = _business_name_candidate(case.incident_place)
                if not normalized:
                    raise InterviewAnswerError(
                        "ראיתי שציינתם קודם את מקום העסקה, אבל כדי למנוע טעות דרוש שם העסק המדויק כפי שמופיע בקבלה."
                    )
            else:
                raise InterviewAnswerError("כדי שהמסמך לא יכיל את המילה ״כתבתי״, צריך להקליד כאן את הפרט עצמו.")
        if field == "defendant.address" and _means_none(normalized):
            channel_note = (
                " כתובת האתר או הערוץ הטלפוני אינה כתובת למסירת כתב תביעה."
                if is_remote_transaction_place(case.incident_place)
                else ""
            )
            raise InterviewAnswerError(
                "נדרשת כתובת פיזית של הנתבע/ת כדי שבית המשפט יוכל למסור את כתב התביעה."
                + channel_note
                + " נסו לאתר אותה בקבלה או בחשבונית, בתחתית האתר, בעמודי ״אודות״, ״צור קשר״ "
                "או ״תנאי שימוש״, או במידע הבסיסי של רשם החברות."
            )
        if field in OPTIONAL_FIELDS and _means_none(normalized):
            setattr(party, attribute, "")
        else:
            _validate_party_answer(
                attribute,
                normalized,
                allow_latin_business_name=(side == "defendant"),
            )
            if attribute == "address":
                _validate_address(normalized)
            setattr(party, attribute, normalized)
        data[side] = party.model_dump(mode="python")
    elif field == "event_summary":
        combined = merge_partial_answer(case.event_summary, normalized)
        if not opening_story_is_complete(combined):
            raise InterviewAnswerError(opening_clarification(combined))
        data[field] = combined
        data["category"] = classify_category(combined, case.category)
        if not data.get("product_or_service_details"):
            data["product_or_service_details"] = extract_purchased_subject(combined)
        if not data.get("damage_items"):
            amounts = _amounts_from_text(combined)
            if len(amounts) == 1:
                data["damage_items"] = [
                    DamageItem(
                        label="מחיר העסקה שנמסר בתיאור הראשוני",
                        amount=amounts[0],
                        kind="direct",
                    ).model_dump(mode="python")
                ]
    elif field in {"transaction_date", "incident_date"}:
        if any(term in normalized for term in RELATIVE_DATE_TERMS):
            raise InterviewAnswerError(
                "כדי שהטיוטה לא תציג תאריך עמום, צריך יום, חודש ושנה. בדקו בקבלה, בהסכם או בהתכתבות וכתבו למשל 01.02.2026."
            )
        normalized_date = _normalize_date(normalized)
        parsed_date = date.fromisoformat("-".join(reversed(normalized_date.split("."))))
        if parsed_date > date.today():
            raise InterviewAnswerError(
                "התאריך שהוזן נמצא בעתיד. בדקו את השנה וכתבו את התאריך כפי שמופיע בקבלה, בהסכם או בהתכתבות."
            )
        other_field = "incident_date" if field == "transaction_date" else "transaction_date"
        other_value = data.get(other_field, "")
        if other_value:
            other_date = date.fromisoformat("-".join(reversed(str(other_value).split("."))))
            transaction = parsed_date if field == "transaction_date" else other_date
            incident = parsed_date if field == "incident_date" else other_date
            if incident < transaction:
                raise InterviewAnswerError(
                    "תאריך האירוע או ההפרה מוקדם מתאריך העסקה. בדקו את שני התאריכים וכתבו את התאריך הנכון."
                )
        data[field] = normalized_date
    elif field == "incident_place":
        _validate_place(normalized)
        data[field] = normalized
    elif field == "product_or_service_details":
        normalized = _strip_reference_suffix(normalized)
        if normalized in REFERENCE_ONLY or not normalized:
            if case.product_or_service_details:
                normalized = case.product_or_service_details
            else:
                raise InterviewAnswerError("לא הצלחתי לזהות בוודאות את המוצר מהתיאור הקודם. כתבו את סוג המוצר או השירות.")
        if len(normalized) < 3:
            raise InterviewAnswerError("צריך תיאור מעט מפורט יותר של המוצר או השירות.")
        data[field] = normalized
    elif field == "defendant_role":
        data[field] = _normalize_defendant_role(normalized)
    elif field == "prior_attempts":
        if normalized.lower().strip(" .?!") in REFERENCE_ONLY:
            reused_attempt = extract_prior_attempt_from_opening(case.event_summary, case)
            if not reused_attempt:
                raise InterviewAnswerError(
                    "לא מצאתי בתיאור הקודם פנייה מלאה הכוללת דרך פנייה ותגובה. "
                    "כתבו בקצרה כיצד פניתם ומה נענה לכם."
                )
            normalized = reused_attempt
            raw_answer = reused_attempt
            confidence = 0.9
        if not prior_attempt_fragment_is_relevant(normalized, case):
            raise InterviewAnswerError(
                "לא זיהיתי בתשובה ניסיון לפנות לנתבע בקשר למחלוקת הנוכחית. "
                "תארו פנייה בטלפון, במייל, בוואטסאפ או בחנות בנושא המוצר, ומה הייתה תגובת הנתבע."
            )
        has_method = prior_attempt_has_method(normalized)
        has_outcome = prior_attempt_has_outcome(normalized)
        if not has_method or not has_outcome:
            if not has_method:
                raise InterviewAnswerError(
                    "קלטתי את התגובה, אבל עדיין חסר אמצעי הפנייה - למשל טלפון, מייל או הגעה לחנות."
                )
            raise InterviewAnswerError(
                "קלטתי כיצד פניתם, אבל עדיין חסרה התגובה שקיבלתם. אפשר להוסיף אותה במשפט קצר."
            )
        data[field] = [normalized]
    elif field == "damage_items":
        if _means_no_additional_damage(normalized) and case.damage_items:
            data[field] = [item.model_dump(mode="python") for item in case.damage_items]
        else:
            known = case.damage_items[0].amount if len(case.damage_items) == 1 else None
            damage_answer = raw_answer
            if re.fullmatch(r"[\d,]+\s*(?:₪|שקל(?:ים)?)?", normalized) and _describes_duplicate_charge(
                case.event_summary
            ):
                damage_answer = f"השבת החיוב הכפול העודף - {normalized}"
            elif re.fullmatch(r"[\d,]+\s*(?:₪|שקל(?:ים)?)?", normalized) and known is not None:
                label = (
                    "השבת מחיר המוצר"
                    if case.category == ClaimCategory.CONSUMER_PRODUCT
                    else "השבת התמורה ששולמה"
                )
                damage_answer = f"{label} - {normalized}"
            data[field] = [
                item.model_dump(mode="python") for item in parse_damage_items(damage_answer, known)
            ]
    elif field == "primary_remedy":
        remedy = _parse_remedy(normalized, allow_none=False)
        data[field] = remedy
        data["desired_non_monetary_relief"] = normalized if remedy == RemedyType.OTHER else _remedy_phrase(remedy)
    elif field == "alternative_remedy":
        remedy = _parse_remedy(normalized, allow_none=True)
        if remedy not in {RemedyType.NONE, RemedyType.OTHER} and remedy == case.primary_remedy:
            raise InterviewAnswerError("הסעד החלופי צריך להיות שונה מהסעד העיקרי, או שאפשר לכתוב ״אין״.")
        data[field] = remedy
        if remedy == RemedyType.OTHER:
            data["desired_non_monetary_relief"] = normalized
    elif field == "warranty_details":
        data[field] = "" if _means_none(normalized) else _normalize_warranty_answer(normalized)
        existing_evidence = [EvidenceItem.model_validate(item) for item in data.get("evidence", [])]
        if "קבלה" in normalized:
            existing_evidence.append(EvidenceItem(name="העתק קבלה/חשבונית בגין העסקה"))
        if "תעודת אחריות" in normalized:
            existing_evidence.append(EvidenceItem(name="העתק תעודת אחריות"))
        data["evidence"] = [item.model_dump(mode="python") for item in _dedupe_evidence(existing_evidence)]
        confidence = 0.9 if data[field] else 1.0
    elif field == "evidence_confirmation":
        if _means_none(normalized) or "אין ראיות" in normalized:
            data["evidence"] = data.get("evidence", [])
        else:
            newly_supplied = [
                item.model_dump(mode="python") for item in _evidence_items_from_answer(raw_answer)
            ]
            combined = [EvidenceItem.model_validate(item) for item in data.get("evidence", []) + newly_supplied]
            data["evidence"] = [item.model_dump(mode="python") for item in _dedupe_evidence(combined)]
        data["confirmed_no_more_evidence"] = True
    elif field in data:
        data[field] = normalized
    else:
        raise InterviewAnswerError("לא ניתן לשייך את התשובה לשדה המבוקש.")

    updated = apply_jurisdiction(CaseData.model_validate(data))
    assessment = AnswerAssessment(path=field, normalized_value=_field_value(updated, field), confidence=confidence)
    return updated, assessment


def build_confirmation_summary(case: CaseData) -> str:
    from .drafting import formalize_resolution_attempt

    attempts = "; ".join(formalize_resolution_attempt(item, case) for item in case.prior_attempts)
    damages = "; ".join(f"{item.label} — {item.amount:,.0f} ₪" for item in case.damage_items)
    evidence = "; ".join(item.name for item in case.evidence) or "לא נמסרו ראיות נוספות"
    alternative = (
        remedy_label(case.alternative_remedy, case.desired_non_monetary_relief)
        if case.alternative_remedy != RemedyType.NONE
        else "אין"
    )
    return (
        "לפני יצירת הטיוטה, נא לבדוק את סיכום העובדות:\n\n"
        f"- עסקה: {case.transaction_date}; {case.product_or_service_details}; מקום: {case.incident_place}\n"
        f"- אירוע/פגם: {case.incident_date}; {_event_fact_summary(case)}\n"
        f"- נתבע/ת: {case.defendant.full_name}; תפקיד: {case.defendant_role}; מען: {case.defendant.address}\n"
        f"- ניסיונות פתרון: {attempts}\n"
        f"- סכומים: {damages}\n"
        f"- סעד עיקרי: {remedy_label(case.primary_remedy, case.desired_non_monetary_relief)}; סעד חלופי: {alternative}\n"
        f"- אחריות: {case.warranty_details or 'לא נמסרו פרטי אחריות'}\n"
        f"- ראיות: {evidence}\n"
        f"- בית משפט שחושב לפי העובדות: תביעות קטנות ב{case.court_city}\n\n"
        "אם הכול נכון, כתבו ״מאשר/ת״. לתיקון, כתבו למשל ״תקן תאריך עסקה״ או ״תקן כתובת נתבע״."
    )


def build_live_draft_preview(
    case: CaseData,
    completed_fields: set[str] | None = None,
    partial_answers: dict[str, str] | None = None,
) -> str:
    """Render the whole interview state as a draft-like, non-final document."""
    completed = completed_fields or set()
    partials = partial_answers or {}
    empty = "⬜ [טרם מולא]"

    def show(path: str, value: Any, *, optional: bool = False) -> str:
        pending = partials.get(path, "").strip()
        if pending and path not in completed:
            return f"{pending} _(נקלט חלקית; טרם אושר)_"
        rendered = str(value or "").strip()
        if rendered and rendered not in {RemedyType.UNDECIDED.value, "טרם נבחר"}:
            suffix = "" if path in completed or path not in INTERVIEW_ORDER else " _(טרם אושר)_"
            return rendered + suffix
        if optional and path in completed:
            return "לא נמסר / לא ידוע"
        return empty

    from .drafting import formal_case_description, formalize_resolution_attempt

    attempts = "; ".join(formalize_resolution_attempt(item, case) for item in case.prior_attempts)
    formal_description = formal_case_description(case) if case.event_summary else ""
    damages = "; ".join(f"{item.label} — {item.amount:,.0f} ₪" for item in case.damage_items)
    evidence = "; ".join(item.name for item in case.evidence)
    primary = "" if case.primary_remedy == RemedyType.UNDECIDED else remedy_label(
        case.primary_remedy, case.desired_non_monetary_relief
    )
    alternative = "" if "alternative_remedy" not in completed else remedy_label(
        case.alternative_remedy, case.desired_non_monetary_relief
    )
    total = f"{case.claim_total:,.0f} ₪" if case.damage_items else ""
    fee = f"{calculate_filing_fee(case.claim_total):,.0f} ₪" if case.damage_items else ""

    return (
        "### טיוטת ביניים — מתעדכנת במהלך השיחה\n\n"
        "_זו תצוגה מקדימה בלבד. ערכים המסומנים ״טרם אושר״ הגיעו מהתיאור הראשוני ועדיין יאומתו בראיון._\n\n"
        f"**בית המשפט לתביעות קטנות ב:** {show('court_city', case.court_city)}\n\n"
        "#### פרטי הצדדים\n\n"
        f"**התובע/ת:** {show('claimant.full_name', case.claimant.full_name)}  \n"
        f"**תעודת זהות:** {show('claimant.identifier', case.claimant.identifier)}  \n"
        f"**כתובת:** {show('claimant.address', case.claimant.address)}  \n"
        f"**טלפון:** {show('claimant.phone', case.claimant.phone)}  \n"
        f"**דוא״ל:** {show('claimant.email', case.claimant.email, optional=True)}\n\n"
        f"**הנתבע/ת:** {show('defendant.full_name', case.defendant.full_name)}  \n"
        f"**מספר מזהה/חברה:** {show('defendant.identifier', case.defendant.identifier, optional=True)}  \n"
        f"**כתובת למסירה:** {show('defendant.address', case.defendant.address)}  \n"
        f"**טלפון:** {show('defendant.phone', case.defendant.phone, optional=True)}  \n"
        f"**תפקיד בעסקה:** {show('defendant_role', case.defendant_role)}\n\n"
        "#### פרטי העסקה והמקרה\n\n"
        f"**תיאור המקרה:** {show('event_summary', formal_description)}  \n"
        f"**מוצר או שירות:** {show('product_or_service_details', case.product_or_service_details)}  \n"
        f"**תאריך העסקה:** {show('transaction_date', case.transaction_date)}  \n"
        f"**תאריך האירוע/ההפרה:** {show('incident_date', case.incident_date)}  \n"
        f"**מקום העסקה/האירוע:** {show('incident_place', case.incident_place)}  \n"
        f"**ניסיונות ליישוב המחלוקת:** {show('prior_attempts', attempts)}  \n"
        f"**אחריות:** {show('warranty_details', case.warranty_details, optional=True)}\n\n"
        "#### הסעדים והראיות\n\n"
        f"**רכיבי הסכום המבוקש:** {show('damage_items', damages)}  \n"
        f"**סכום תביעה נוכחי:** {show('damage_items', total)}  \n"
        f"**אגרה משוערת:** {show('damage_items', fee)}  \n"
        f"**סעד עיקרי:** {show('primary_remedy', primary)}  \n"
        f"**סעד חלופי:** {show('alternative_remedy', alternative, optional=True)}  \n"
        f"**ראיות ונספחים:** {show('evidence_confirmation', evidence, optional=True)}\n\n"
        f"**אישור העובדות:** {'✅ אושרו' if case.facts_confirmed else empty}"
    )


def confirmation_is_yes(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    normalized = re.sub(r"[.!?,;:]+$", "", normalized).strip()
    return normalized in {
        "מאשר",
        "מאשרת",
        "מאושר",
        "מאושרת",
        "אישור",
        "כן",
        "הכול נכון",
        "הכל נכון",
        "נכון",
    }


def is_recall_request(text: str) -> bool:
    normalized = " ".join(text.lower().split()).strip(" ?")
    return any(
        phrase in normalized
        for phrase in (
            "מה הזנתי",
            "מה כתבתי",
            "מה כתבת",
            "מה נקלט",
            "מה אמרתי",
            "איזה תאריך",
            "מה התאריך",
            "מה כן יש לך",
            "מה יש לך",
            "מה כבר יש לך",
            "מה כבר רשום",
            "מה רשום עד עכשיו",
            "מה קלטת עד עכשיו",
            "מה נקלט עד עכשיו",
        )
    )


def is_current_question_help_request(text: str) -> bool:
    """Recognize a request for help with the active field, not a factual answer."""
    normalized = " ".join(text.lower().split()).strip(" .?!׳״'\"")
    return any(
        phrase in normalized
        for phrase in (
            "מה חסר",
            "מה עוד חסר",
            "למה זה לא עובר",
            "למה זה לא מתקבל",
            "למה לא קיבלת",
            "איך לענות",
            "איך עונים",
            "עזרה בשאלה",
        )
    )


def is_current_question_reset_request(text: str) -> bool:
    """Recognize an explicit request to discard only the active field fragments."""
    normalized = " ".join(text.lower().split()).strip(" .?!׳״'\"")
    return normalized in {
        "אפס את השאלה",
        "אפסי את השאלה",
        "התחל את השאלה מחדש",
        "התחילי את השאלה מחדש",
        "התחל שדה מחדש",
        "התחילי שדה מחדש",
        "מחק את התשובות לשאלה",
        "מחקי את התשובות לשאלה",
    }


def build_repeated_failure_guidance(
    case: CaseData,
    field: str,
    partial_answer: str = "",
    failure_count: int = 0,
    last_error: str = "",
) -> str:
    """Explain an active-field failure without accepting or inventing a value."""
    partial = " ".join(partial_answer.split()).strip()
    lines = ["נראה שנתקענו בשאלה הזו, אז אפרט בדיוק."]
    recognized: list[str] = []
    missing: list[str] = []
    example = ""

    if field == "event_summary":
        source = partial or case.event_summary
        subject = extract_purchased_subject(source)
        amounts = _amounts_from_text(source)
        if subject:
            recognized.append(f"המוצר או השירות: {subject}")
        if amounts:
            recognized.append("סכום: " + ", ".join(f"{amount:,.0f} ₪" for amount in amounts))
        missing = opening_story_missing_parts(source)
        example = "למשל: ״קניתי מזגן ב-3,500 ₪ ואחרי שבוע הוא הפסיק לפעול״."
    elif field == "prior_attempts":
        if partial and prior_attempt_has_method(partial):
            recognized.append("אמצעי הפנייה")
        if partial and prior_attempt_has_outcome(partial):
            recognized.append("תגובת הנתבע/ת")
        if not partial or not prior_attempt_has_method(partial):
            missing.append("איך פניתם - למשל טלפון, מייל, וואטסאפ או הגעה לעסק")
        if not partial or not prior_attempt_has_outcome(partial):
            missing.append("מה ענה הנתבע/ת או מה קרה לאחר הפנייה")
        example = "למשל: ״התקשרתי פעמיים לחנות והם סירבו לתקן או להחזיר את הכסף״."
    elif field == "damage_items":
        amounts = _amounts_from_text(partial)
        if amounts:
            recognized.append("סכומים: " + ", ".join(f"{amount:,.0f} ₪" for amount in amounts))
        missing.append("סיבה נפרדת לכל סכום, כך שאפשר יהיה לחשב את התביעה ללא ניחוש")
        example = "למשל: ״מחיר מוצר 3,200; עוגמת נפש 800״."
    else:
        requirements = {
            "transaction_date": "תאריך עסקה אמיתי, מלא ושאינו בעתיד",
            "incident_date": "תאריך אירוע מלא, שאינו לפני תאריך העסקה ואינו בעתיד",
            "incident_place": "מקום פיזי ספציפי, או שם האתר/האפליקציה ושם החברה, או ״עסקה טלפונית״",
            "product_or_service_details": "סוג המוצר או השירות, ודגם אם הוא ידוע",
            "defendant.full_name": "שם העסק המדויק כפי שהוא מופיע בקבלה או בחשבונית",
            "defendant_role": "תפקיד כמו מוכר, יצרן, יבואן או נותן שירות",
            "defendant.identifier": "מספר זהות/חברה בן 9 ספרות, או ״לא ידוע״",
            "defendant.address": "כתובת פיזית מלאה למסירת כתב התביעה",
            "defendant.phone": "מספר טלפון עם קידומת, או ״לא ידוע״",
            "primary_remedy": "בחירה בין החזר וביטול, תיקון, החלפה או תיאור קצר של סעד אחר",
            "alternative_remedy": "סעד שונה מהסעד העיקרי, או ״אין״",
            "warranty_details": "פרטי האחריות, או ״לא ידוע״",
            "evidence_confirmation": "שם של כל מסמך/ראיה, או ״אין ראיות נוספות״",
            "claimant.full_name": "שם מלא כפי שיופיע בכתב התביעה",
            "claimant.identifier": "מספר זהות בן 9 ספרות",
            "claimant.address": "כתובת מלאה של התובע/ת",
            "claimant.phone": "מספר טלפון מלא עם קידומת",
            "claimant.email": "כתובת דוא״ל תקינה, או ״אין״",
        }
        missing.append(requirements.get(field, "תשובה ממוקדת התואמת לשאלה"))
        examples = {
            "transaction_date": "למשל: 1.1.26 או 01/01/2026.",
            "incident_date": "למשל: 8.1.26 או 08/01/2026.",
            "incident_place": "למשל: ״רחוב הארבעה 4, תל אביב״ או ״באתר courses.co.il של COURSES״.",
            "defendant.address": "למשל: ״רחוב המסחר 8, תל אביב״.",
            "primary_remedy": "למשל: ״ביטול העסקה והחזר כספי״.",
        }
        example = examples.get(field, "")

    if recognized:
        lines.append("מה כבר זיהיתי: " + "; ".join(recognized) + ".")
    if missing:
        lines.append("מה עדיין נדרש: " + "; ".join(missing) + ".")
    if last_error:
        lines.append("סיבת הדחייה האחרונה: " + last_error)
    if example:
        lines.append(example)
    if failure_count >= 6:
        lines.append("לא אכניס לתיק ערך שלא עבר אימות, כדי לא ליצור עובדה שגויה.")
    lines.append(
        "אפשר לכתוב ״אפס את השאלה״ כדי למחוק את הפרטים החלקיים, "
        "או ״מה נקלט?״ כדי לראות את טיוטת הביניים."
    )
    return "\n\n".join(lines)


def field_value_for_user(case: CaseData, field: str) -> str:
    value = _field_value(case, field)
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, RemedyType):
        return remedy_label(value)
    return str(value or "")


def merge_partial_answer(previous: str, current: str) -> str:
    previous = " ".join(previous.split()).strip()
    current = " ".join(current.split()).strip()
    if not previous:
        return current
    if not current or current in previous:
        return previous
    if previous in current:
        return current
    return f"{previous} {current}"


def prior_attempt_has_method(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(
        word in normalized
        for word in (
            "פנית",
            "התקשר",
            "צלצל",
            "חייג",
            "טלפנ",
            "טילפנ",
            "שוחח",
            "דיבר",
            "אמרתי לו",
            "אמרתי לה",
            "אמרתי להם",
            "שוחחתי איתו",
            "שוחחתי איתה",
            "שוחחתי איתם",
            "נפגשתי",
            "הלכתי למסעדה",
            "הגעתי למסעדה",
            "באתי למסעדה",
            "ניגשתי אליו",
            "ניגשתי אליה",
            "שלח",
            "כתבתי להם",
            "מייל",
            "טלפון",
            "ווטסאפ",
            "וואטסאפ",
            "וואצאפ",
            "מסרון",
            "sms",
            "צ׳אט",
            "צ'אט",
            "מכתב",
            "פקס",
            "הגעתי לחנות",
            "באתי לחנות",
            "ניגשתי לחנות",
            "הגעתי לסניף",
            "באתי לסניף",
            "ניגשתי לסניף",
            "ניסיתי להחזיר",
            "ניסיתי לקבל החזר",
            "החזרתי לחנות",
            "החזרתי לסניף",
            "הגעה לחנות",
            "הגעה לסניף",
            "פיזית לחנות",
            "פיזית לסניף",
        )
    )


def prior_attempt_has_outcome(text: str) -> bool:
    """Recognize the defendant's response, not the claimant saying 'אמרתי'."""
    normalized = " ".join(text.lower().split())
    return bool(
        re.search(
            r"(?:הם|הוא|היא|הנציג(?:ה)?|המוכר(?:ת)?|החנות)\s+(?:אמר|אמרה|אמרו|טען|טענה|טענו)|"
            r"(?:אמרו|נאמר|השיבו|השיבה|השיב)\s+לי|"
            r"(?:ו?אמרו|נאמר|השיבו|השיבה|השיב)\s+(?:לי\s+)?(?:ש)?(?:אין|לא)|"
            r"אין\s+(?:החזר|החזרים|החזרות|זיכוי|פיצוי|פיצויים)|"
            r"לא\s+(?:מחזיר|מחזירים|מחזירה|מזכה|מזכים|מפצה|מפצים)|"
            r"סירב|סירבה|סירבו|התנער|התנערה|התנערו|מתנער|מתנערת|מתנערים|"
            r"האשימ|מאשימ|לא\s+ענו|לא\s+קיבל(?:תי|נו)?\s+תשובה|לא\s+נענה|לא\s+נענתה|לא\s+אחרא|"
            r"לא\s+מוכנ|דחה|דחתה|דחו|סירבו?\s+.*(?:זיכוי|לזכות|להחזיר|לתקן)",
            normalized,
        )
    )


def prior_attempt_fragment_is_relevant(
    text: str, case: CaseData, has_pending_fragment: bool = False
) -> bool:
    normalized = " ".join(text.lower().split())
    method = prior_attempt_has_method(normalized)
    outcome = prior_attempt_has_outcome(normalized)
    dispute_terms = {
        "תקלה",
        "פגם",
        "התקלקל",
        "אחריות",
        "החזר",
        "להחזיר",
        "תיקון",
        "לתקן",
        "זיכוי",
        "כסף",
        "המוצר",
        "השירות",
        "חיוב",
        "חייב",
        "חויב",
        "חוייב",
        "כפול",
        "פעמיים",
    }
    dispute_terms.update(
        token
        for token in re.findall(r"[א-ת]{3,}", case.product_or_service_details.lower())
        if token
    )
    mentions_dispute = any(term in normalized for term in dispute_terms)
    return method or (mentions_dispute and outcome) or (has_pending_fragment and outcome)


def extract_prior_attempt_from_opening(text: str, case: CaseData | None = None) -> str:
    """Extract a user-stated resolution attempt for later explicit confirmation."""
    normalized = " ".join(text.split()).strip(" \"'״׳")
    match = re.search(
        r"((?:פניתי|התקשרתי|צלצלתי|חייגתי|טלפנתי|שלחתי|כתבתי|הגעתי|באתי|"
        r"ניגשתי|ניסיתי\s+להחזיר|ניסיתי\s+לקבל\s+החזר).+)$",
        normalized,
    )
    if not match:
        return ""
    candidate = re.sub(
        r"(?:[,.]\s*)?(?:אני|אנחנו)\s+(?:רוצה|רוצים|מבקש|מבקשת|מבקשים)\s+לתבוע.*$",
        "",
        match.group(1),
    ).strip(" ,.;")
    active_case = case or CaseData()
    if not prior_attempt_has_method(candidate) or not prior_attempt_has_outcome(candidate):
        return ""
    if not prior_attempt_fragment_is_relevant(candidate, active_case):
        return ""
    return candidate


def correction_field_from_text(text: str) -> str | None:
    command = correction_command(text)
    return command[0] if command else None


def correction_command(text: str) -> tuple[str, str] | None:
    normalized = " ".join(text.lower().split())
    if not re.search(r"(?:תקן|תקני|לתקן|שנה|שני)", normalized):
        return None
    mappings = (
        (("כתובת נתבע", "מען נתבע"), "defendant.address"),
        (("מען",), "defendant.address"),
        (("תאריך עסקה", "מועד עסקה"), "transaction_date"),
        (("תאריך פגם", "תאריך אירוע", "מועד פגם"), "incident_date"),
        (("מקום", "סניף"), "incident_place"),
        (("מוצר", "שירות", "דגם"), "product_or_service_details"),
        (("שם נתבע", "שם העסק"), "defendant.full_name"),
        (("נתבע/ת", "נתבעת", "נתבע"), "defendant.full_name"),
        (("ניסיונות", "פניות"), "prior_attempts"),
        (("סכומים", "סכום", "נזקים", "נזק"), "damage_items"),
        (("סעד עיקרי",), "primary_remedy"),
        (("סעד חלופי",), "alternative_remedy"),
        (("אחריות",), "warranty_details"),
        (("ראיות", "נספחים"), "evidence_confirmation"),
        (("שם תובע",), "claimant.full_name"),
        (("כתובת תובע",), "claimant.address"),
    )
    for phrases, field in mappings:
        for phrase in phrases:
            index = normalized.find(phrase)
            if index >= 0:
                value = normalized[index + len(phrase) :].strip(" /:,-")
                return field, value
    return None


def parse_damage_items(answer: str, known_transaction_amount: Decimal | None = None) -> list[DamageItem]:
    if re.search(r"(?<![\d.])-\d[\d,]*(?:\.\d+)?", answer):
        raise InterviewAnswerError("סכום נזק אינו יכול להיות שלילי. כתבו את הסכום החיובי המבוקש מבית המשפט.")
    parseable_answer = answer
    for confirmation in DAMAGE_CONFIRMATIONS:
        parseable_answer = re.sub(re.escape(confirmation), "", parseable_answer, flags=re.IGNORECASE)
    lines = [line.strip() for line in re.split(r"[;\n]+", parseable_answer) if line.strip()]
    items: list[DamageItem] = []
    for line in lines:
        matches = list(re.finditer(r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?![\d.])", line))
        if not matches:
            continue
        amounts = [_decimal_from_match(match.group()) for match in matches]
        if (
            len(amounts) == 1
            and known_transaction_amount
            and amounts[0] > known_transaction_amount
            and any(word in line for word in ("עוגמת", "טרחה", "נפש"))
            and any(word in line for word in ("מחיר", "מוצר", "סהכ", "סך הכל"))
        ):
            items.extend(
                [
                    DamageItem(label="השבת מחיר המוצר", amount=known_transaction_amount, kind="direct"),
                    DamageItem(
                        label="טרחה ועוגמת נפש",
                        amount=amounts[0] - known_transaction_amount,
                        kind="non_pecuniary",
                    ),
                ]
            )
            continue
        labels = _labels_for_amounts(line, matches)
        for amount, label in zip(amounts, labels, strict=True):
            normalized_label = _normalize_damage_label(label)
            items.append(
                DamageItem(
                    label=normalized_label,
                    amount=amount,
                    kind=_damage_kind(normalized_label),
                )
            )
    if not items:
        raise InterviewAnswerError(
            "לא זיהיתי סכום. אין צורך בסימן ₪; אפשר לכתוב למשל ״מחיר מוצר 3200, עוגמת נפש 800״."
        )
    non_pecuniary = sum((item.amount for item in items if item.kind == "non_pecuniary"), Decimal("0"))
    direct_total = sum((item.amount for item in items if item.kind == "direct"), Decimal("0"))
    comparison_amount = known_transaction_amount or direct_total
    if (
        comparison_amount
        and non_pecuniary >= comparison_amount
        and not any(phrase in answer for phrase in (*DAMAGE_CONFIRMATIONS, "מאשר נזקים", "נזק לא ממוני"))
    ):
        total = sum((item.amount for item in items), Decimal("0"))
        pecuniary_total = total - non_pecuniary
        price_note = (
            f", כולל {known_transaction_amount:,.0f} ₪ למחיר המוצר"
            if known_transaction_amount
            else ""
        )
        raise InterviewAnswerError(
            f"זיהיתי רכיבי נזק כספי והוצאות בסך {pecuniary_total:,.0f} ₪{price_note}, "
            f"ועוד {non_pecuniary:,.0f} ₪ לעוגמת נפש, "
            f"כלומר {total:,.0f} ₪ בסך הכול. אם התכוונתם לסך כולל אחר, כתבו את שני הרכיבים מחדש; "
            "אם זה אכן הפירוט המבוקש, כתבו ״בוודאות״ או ״זה הסך הכולל״."
        )
    return items


def merge_damage_confirmation(previous_answers: list[str], current: str) -> str | None:
    """Attach a natural confirmation to the last itemized damage answer."""
    normalized = " ".join(current.lower().split()).strip(" .!?")
    explicit_confirmation = any(phrase in normalized for phrase in DAMAGE_CONFIRMATIONS)
    short_confirmation = normalized in {"כן", "מאשר", "מאשרת"}
    if not explicit_confirmation and not short_confirmation:
        return None
    candidate = next(
        (answer for answer in reversed(previous_answers) if len(re.findall(r"\d[\d,]*", answer)) >= 2),
        None,
    )
    confirmation = current if explicit_confirmation else "בוודאות"
    return f"{candidate} {confirmation}" if candidate else None


def interview_progress(case: CaseData, completed_fields: set[str]) -> tuple[int, int]:
    return 1 + len(set(INTERVIEW_ORDER) & completed_fields), len(INTERVIEW_ORDER) + 2


def remedy_label(remedy: RemedyType, other_details: str = "") -> str:
    return {
        RemedyType.UNDECIDED: "טרם נבחר",
        RemedyType.NONE: "אין",
        RemedyType.REFUND: "השבת התמורה",
        RemedyType.CANCELLATION: "ביטול העסקה והשבת התמורה",
        RemedyType.REPAIR: "תיקון",
        RemedyType.REPLACEMENT: "החלפה",
        RemedyType.OTHER: other_details or "סעד אחר שטרם פורט",
    }[remedy]


def _parse_remedy(answer: str, allow_none: bool) -> RemedyType:
    if allow_none and _means_none(answer):
        return RemedyType.NONE
    if "ביטול" in answer:
        return RemedyType.CANCELLATION
    if any(word in answer for word in ("החזר", "השבה", "כסף")):
        return RemedyType.REFUND
    if "תיקון" in answer or "לתקן" in answer:
        return RemedyType.REPAIR
    if "החלפ" in answer:
        return RemedyType.REPLACEMENT
    if len(answer) >= 4:
        return RemedyType.OTHER
    raise InterviewAnswerError("לא זוהה סעד. כתבו החזר וביטול, תיקון, החלפה או תיאור קצר של סעד אחר.")


def _remedy_phrase(remedy: RemedyType) -> str:
    return "" if remedy in {RemedyType.REFUND, RemedyType.CANCELLATION, RemedyType.NONE} else remedy_label(remedy)


def _normalize_evidence(text: str) -> str:
    clean = " ".join(text.strip().split())
    clean = re.sub(r"^(יש לי|יש ברשותי|ברשותי)\s+", "", clean)
    if clean in {"קבלה", "חשבונית"}:
        return "העתק קבלה/חשבונית בגין העסקה"
    if clean in {"אחריות", "תעודת אחריות"}:
        return "העתק תעודת אחריות"
    if "קבלה" in clean and not clean.startswith("העתק"):
        return f"העתק {clean}"
    if any(
        word in clean
        for word in ("התכתבות", "תכתובת", "ווטסאפ", "וואטסאפ", "מייל", "דוא")
    ) and not clean.startswith("העתק"):
        return f"העתק {clean}"
    return clean


def _evidence_items_from_answer(text: str) -> list[EvidenceItem]:
    """Split common evidence named without punctuation, e.g. 'אחריות קבלה'."""
    items: list[EvidenceItem] = []
    for raw_item in re.split(r"[,;\n]+", text):
        item = " ".join(raw_item.split()).strip()
        if not item:
            continue
        has_receipt = any(word in item for word in ("קבלה", "חשבונית"))
        has_warranty = "אחריות" in item
        # A compact answer such as "אחריות קבלה" names two documents.
        # A descriptive line such as "חשבונית הובלה חלופית" is one document
        # and must not cause the remaining lines to be discarded.
        if has_receipt and has_warranty:
            items.append(EvidenceItem(name="העתק קבלה/חשבונית בגין העסקה"))
            items.append(EvidenceItem(name="העתק תעודת אחריות"))
        else:
            items.append(EvidenceItem(name=_normalize_evidence(item)))
    return items


def _dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    deduped: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in items:
        key = re.sub(r"[^א-תa-z0-9]+", "", item.name.lower())
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _event_fact_summary(case: CaseData) -> str:
    from .drafting import formal_event_fact

    return formal_event_fact(case)


def _normalize_warranty_answer(text: str) -> str:
    normalized = " ".join(text.split()).strip(" .")
    lower = normalized.lower()
    if re.fullmatch(r"כן(?:,?\s+)?(?:(?:ה?מוצר)\s+)?חדש", lower):
        return "המוצר נרכש כחדש וקיימת אחריות, אך לא נמסרו תקופתה או פרטי מסמך האחריות"
    if lower.startswith("כן"):
        remainder = normalized[2:].lstrip(" ,:-")
        if remainder:
            return f"קיימת אחריות; {remainder}"
        return "נמסר כי קיימת אחריות, אך לא נמסרו פרטים נוספים"
    return normalized


def _strip_reference_suffix(text: str) -> str:
    return re.sub(
        r"\s*,?\s*(?:(?:כבר\s+)?אמרתי(?:\s+לך)?|(?:כבר\s+)?כתבתי(?:\s+לך)?|כמו\s+שכתבתי)\s*$",
        "",
        text,
    ).strip()


def _business_name_candidate(place: str) -> str:
    first_segment = " ".join(place.split(",", 1)[0].split()).strip()
    # A natural place answer often combines the brand and its street address,
    # for example "חנות KSP רחוב ירושלים 3 תל אביב". The legal-name field
    # must reuse only the business token, never the whole service address.
    before_address = re.split(
        r"\s+(?=(?:רחוב|רח[׳']|שדרות|שד[׳']|דרך|קניון|מרכז|סניף)\b)",
        first_segment,
        maxsplit=1,
    )[0].strip(" ,.-")
    locality = locality_from_text(place)
    if locality:
        before_address = re.sub(
            rf"\s+{re.escape(locality)}$",
            "",
            before_address,
        ).strip(" ,.-")
    # "חנות" and "בית העסק" are generic location labels; prefixes such as
    # "מסעדת" or "מלון" commonly form part of the displayed business name.
    generic_prefix = re.match(r"^(?:חנות|בית העסק)\s+(.+)$", before_address)
    if generic_prefix:
        candidate = generic_prefix.group(1).strip(" ,.-")
        return candidate if candidate else ""
    if re.match(r"^(?:מסעדת|מסעדה|חברת|חברה|מועדון|מלון|מוסך)\s+", before_address):
        return before_address
    match = re.search(r"\b\d{1,4}\s+(.+)$", place.strip())
    if not match:
        return ""
    candidate = match.group(1).strip(" ,.-")
    return candidate if len(candidate.split()) >= 2 else ""


def _address_from_prior_place(case: CaseData) -> str:
    address = " ".join(case.incident_place.split()).strip(" ,.-")
    if case.defendant.full_name:
        address = re.sub(
            re.escape(case.defendant.full_name),
            "",
            address,
            count=1,
            flags=re.IGNORECASE,
        ).strip(" ,.-")
    address = re.sub(
        r"^(?:מסעדת|מסעדה|חנות|בית העסק|חברת|חברה|מועדון|מלון|מוסך)\s*[,:-]?\s*",
        "",
        address,
    ).strip(" ,.-")
    return address


def _means_no_additional_damage(answer: str) -> bool:
    normalized = " ".join(answer.lower().split())
    return normalized in {
        "זהו",
        "רק זה",
        "אין עוד",
        "אין נזק נוסף",
        "אין נזקים נוספים",
        "ללא נזק נוסף",
        "רק מחיר המוצר",
        "רק הסכום הזה",
    } or normalized.startswith("רק מחיר המוצר")


def _decimal_from_match(value: str) -> Decimal:
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise InterviewAnswerError("אחד הסכומים אינו מספר תקין.") from exc


def _labels_for_amounts(line: str, matches: list[re.Match[str]]) -> list[str]:
    prefix = line[: matches[0].start()].strip(" -:,₪")
    labels: list[str] = []
    if not prefix:
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            labels.append(line[match.end() : end])
    else:
        previous_end = 0
        for match in matches:
            labels.append(line[previous_end : match.start()])
            previous_end = match.end()
    return labels


def _normalize_damage_label(label: str) -> str:
    clean = re.sub(r"(?:₪|ש״ח|שח|שקל(?:ים)?)", "", label, flags=re.IGNORECASE)
    clean = re.sub(r"^(?:ו?עוד|וגם|כי|זה)\s+", "", clean.strip(" -:,"))
    if any(word in clean for word in ("עוגמת", "עגמת", "נפש", "טרחה")):
        return "טרחה ועוגמת נפש"
    if any(word in clean for word in ("מחיר", "מוצר", "השבה", "החזר", *PRODUCT_WORDS)):
        return "השבת מחיר המוצר"
    return clean or "רכיב נזק"


def _damage_kind(label: str) -> str:
    if any(word in label for word in ("עוגמת", "טרחה", "לא ממוני")):
        return "non_pecuniary"
    if any(word in label for word in ("טכנאי", "נסיעות", "הוצאה", "הוצאות", "הפסד")):
        return "consequential"
    return "direct"


def _describes_duplicate_charge(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(
        phrase in normalized
        for phrase in (
            "חיוב כפול",
            "חויבתי פעמיים",
            "חוייבתי פעמיים",
            "חייב אותי פעמיים",
            "חייבו אותי פעמיים",
            "חייב אותי פעמים",
            "חייבו אותי פעמים",
            "נגבה פעמיים",
            "שילמתי פעמיים",
        )
    )


def _amounts_from_text(text: str) -> list[Decimal]:
    matches = re.findall(
        r"(?:₪\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:₪|ש״ח|שח|שקל(?:ים)?)",
        text,
        flags=re.IGNORECASE,
    )
    values: list[Decimal] = []
    for match in matches:
        try:
            values.append(Decimal(match.replace(",", "")))
        except InvalidOperation:
            continue
    thousand_word_values = {
        "אחד": 1,
        "אחת": 1,
        "שני": 2,
        "שתי": 2,
        "שניים": 2,
        "שלוש": 3,
        "שלושה": 3,
        "ארבע": 4,
        "ארבעה": 4,
        "חמש": 5,
        "חמישה": 5,
        "שש": 6,
        "שישה": 6,
        "שבע": 7,
        "שבעה": 7,
        "שמונה": 8,
        "תשע": 9,
        "תשעה": 9,
        "עשר": 10,
        "עשרה": 10,
        "עשרת": 10,
        "עשרים": 20,
        "שלושים": 30,
        "ארבעים": 40,
        "חמישים": 50,
    }
    for match in re.finditer(
        r"(?<![\d,])(?P<number>\d[\d,]*(?:\.\d+)?|[א-ת]+)\s+אל(?:ף|פים)"
        r"(?:\s+(?:שקל|שקלים|ש\"ח|ש״ח))?",
        text,
        flags=re.IGNORECASE,
    ):
        raw_number = match.group("number").replace(",", "")
        try:
            multiplier = Decimal(raw_number)
        except InvalidOperation:
            word_value = thousand_word_values.get(raw_number)
            if word_value is None:
                continue
            multiplier = Decimal(word_value)
        values.append(multiplier * Decimal("1000"))
    if not values:
        bare_purchase_amounts = re.findall(
            r"(?:^|\s)(?:ב\s*|שילמתי\s+|עלה\s+|מחיר(?:ו|ה)?\s+)(\d[\d,]*(?:\.\d+)?)\b",
            text,
        )
        for match in bare_purchase_amounts:
            try:
                values.append(Decimal(match.replace(",", "")))
            except InvalidOperation:
                continue
    return values


def _is_exact_date(value: str) -> bool:
    if not value:
        return False
    try:
        _normalize_date(value)
        return True
    except InterviewAnswerError:
        return False


def _normalize_date(value: str) -> str:
    compact = re.sub(r"\D", "", value)
    parts = [part for part in re.split(r"[./-]", value) if part]
    if len(parts) == 3 and len(parts[0]) == 4:
        year, month, day = parts
    elif len(parts) == 3:
        day, month, year = parts
    elif len(compact) == 8:
        day, month, year = compact[:2], compact[2:4], compact[4:]
    elif len(compact) == 6:
        day, month, year = compact[:2], compact[2:4], compact[4:]
    else:
        raise InterviewAnswerError(
            "לא הצלחתי לזהות את מבנה התאריך. אפשר לכתוב 01.01.2026, 01.01.26 או 01012026."
        )
    day_i, month_i, year_i = int(day), int(month), int(year)
    if len(year) == 2:
        year_i += 2000
    if not 1 <= month_i <= 12:
        raise InterviewAnswerError(f"החודש {month_i} אינו תקין. מספר החודש צריך להיות בין 1 ל-12.")
    if day_i < 1:
        raise InterviewAnswerError("מספר היום צריך להיות 1 ומעלה. בדקו את התאריך ונסו שוב.")
    try:
        date(year_i, month_i, day_i)
    except ValueError:
        raise InterviewAnswerError(
            f"היום {day_i} אינו קיים בחודש {month_i} בשנת {year_i}. בדקו את התאריך ונסו שוב."
        )
    return f"{day_i:02d}.{month_i:02d}.{year_i:04d}"


def _is_specific_place(value: str) -> bool:
    normalized = " ".join(value.split()).strip()
    if normalized in VAGUE_PLACES or re.search(r"(?:חנות|סניף)\s*[xXאיקס]\b", normalized):
        return False
    return is_remote_transaction_place(normalized) or bool(locality_from_text(normalized))


def _validate_place(answer: str) -> None:
    # Website names and domains may legitimately be written in Latin letters.
    # Apply physical-address keyboard checks only to non-remote locations.
    if not is_remote_transaction_place(answer):
        _validate_address(answer)
    if not _is_specific_place(answer):
        if is_vague_online_place(answer):
            raise InterviewAnswerError(
                "העסקה המקוונת עדיין מתוארת באופן כללי מדי. ציינו את האתר או האפליקציה ואת שם החברה."
            )
        raise InterviewAnswerError(
            "המקום עדיין כללי מדי. בעסקה פיזית ציינו יישוב ולפי האפשר סניף או כתובת; "
            "בעסקה מקוונת ציינו את האתר או האפליקציה ואת שם החברה, או כתבו ״עסקה טלפונית״ אם כך בוצעה העסקה."
        )


def _validate_address(answer: str) -> None:
    if not answer.strip(" ,.-"):
        raise InterviewAnswerError(
            "לא הצלחתי להפיק כתובת למסירה ממקום העסקה שנכתב קודם. "
            "כתבו את הרחוב, המספר והיישוב כפי שהם מופיעים בקבלה או באתר העסק."
        )
    if re.search(r"[A-Za-z]", answer) and not re.search(r"[א-ת]", answer):
        raise InterviewAnswerError(
            "הכתובת נכתבה באותיות לטיניות ונראה שמקלדת האנגלית הייתה פעילה. החליפו לעברית וכתבו אותה שוב."
        )
    if re.search(r"תל אבי(?!ב)", answer):
        raise InterviewAnswerError("ייתכן שיש טעות בשם היישוב: האם התכוונתם ל״תל אביב״? כתבו שוב את הכתובת המתוקנת.")
    for typo, correction in CITY_TYPO_HINTS.items():
        if typo in answer:
            raise InterviewAnswerError(f"ייתכן שיש טעות בשם היישוב: האם התכוונתם ל״{correction}״? כתבו שוב את הכתובת המתוקנת.")


def _normalize_defendant_role(answer: str) -> str:
    roles: list[str] = []
    if any(word in answer for word in ("מוכר", "חנות", "בית עסק")):
        roles.append("מוכר")
    if "יבוא" in answer:
        roles.append("יבואן")
    if "יצר" in answer:
        roles.append("יצרן")
    if "שירות" in answer:
        roles.append("נותן שירות")
    if roles:
        if len(roles) == 1:
            return roles[0]
        return ", ".join(roles[:-1]) + " ו" + roles[-1]
    return answer


def _validate_party_answer(
    attribute: str,
    answer: str,
    *,
    allow_latin_business_name: bool = False,
) -> None:
    digits = re.sub(r"\D", "", answer)
    if attribute == "full_name" and re.search(r"[A-Za-z]", answer) and not re.search(r"[א-ת]", answer):
        words = re.findall(r"[A-Za-z]+", answer)
        looks_like_business_brand = allow_latin_business_name and (
            len(words) == 1
            or any(any(character.isupper() for character in word) for word in words)
            or any(word.lower() in {"ltd", "inc", "llc", "corp", "company"} for word in words)
        )
        if not looks_like_business_brand:
            raise InterviewAnswerError(
                "השם נכתב באותיות לטיניות ונראה שמקלדת האנגלית הייתה פעילה. "
                "אם זהו שמו הרשמי של העסק באנגלית, העתיקו אותו בדיוק כפי שהוא מופיע בקבלה או בחשבונית."
            )
    if attribute == "identifier" and len(digits) != 9:
        raise InterviewAnswerError("מספר זהות או מספר חברה צריך לכלול 9 ספרות, או כתבו ״לא ידוע״.")
    if attribute == "phone":
        if digits.startswith("05") and len(digits) != 10:
            raise InterviewAnswerError("מספר טלפון נייד בישראל צריך לכלול 10 ספרות, למשל 0501234567.")
        if not digits.startswith("05") and len(digits) < 9:
            raise InterviewAnswerError("מספר הטלפון נראה קצר מדי. נסו שוב עם קידומת.")
    if attribute == "email" and "@" not in answer:
        raise InterviewAnswerError("כתובת הדוא״ל אינה תקינה. נסו שוב או כתבו ״אין״.")


def _means_none(answer: str) -> bool:
    normalized = answer.replace("׳", "").replace('"', "").strip().lower()
    return normalized in {"לא", "אין", "לא ידוע", "אינו ידוע", "לא יודע", "לא יודעת", "אין סעד חלופי"} or normalized.startswith("לא ידו")


def _field_value(case: CaseData, field: str) -> Any:
    if "." in field:
        root, child = field.split(".", 1)
        return getattr(getattr(case, root), child)
    if field == "evidence_confirmation":
        return [item.name for item in case.evidence]
    return getattr(case, field)
