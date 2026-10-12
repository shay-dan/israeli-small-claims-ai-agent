from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

from pydantic import ValidationError

from .models import (
    AnswerRepair,
    CaseData,
    ClaimCategory,
    DraftReview,
    DraftSections,
    InterviewConsistencyReview,
    RemedyType,
    StateExtraction,
)
from .privacy import redact_text, safe_case_context
from .drafting import formal_event_fact, formalize_resolution_attempt


class ModelUnavailable(RuntimeError):
    """Raised when Gemini cannot provide a validated response."""


class GeminiAgentClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, max_retries: int = 1):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        self.max_retries = max_retries
        self._model = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _base_model(self):
        if not self.available:
            raise ModelUnavailable("GEMINI_API_KEY is not configured")
        if self._model is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ModelUnavailable("langchain-google-genai is not installed") from exc
            self._model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                max_retries=0,
                timeout=35,
            )
        return self._model

    def extract(self, user_text: str, case: CaseData) -> StateExtraction:
        prompt = f"""
את/ה רכיב חילוץ מידע עבור ראיון לתביעה קטנה בישראל.
החזר/י רק נתונים שהמשתמש מסר במפורש. אין להמציא פרטים, תאריכים, סכומים או ראיות.
מותר לעדכן רק שדות סקלריים בטוחים: event_summary, transaction_date, incident_date,
incident_place, product_or_service_details, warranty_details, defendant_role ו-category.
אין לעדכן chronology, prior_attempts, damage_items, evidence, witnesses, סעדים,
אישור עובדות, סמכות מקומית או פרטי זיהוי וקשר; אלה נאספים ומאומתים בקוד המקומי.
אם פרט אינו חד-משמעי, הוסף/י אותו ל-ambiguities ולא ל-updates.

מצב נוכחי נטול פרטים מזהים:
{json.dumps(safe_case_context(case), ensure_ascii=False, default=str)}

הודעת המשתמש, לאחר השחרה מקומית:
{redact_text(user_text, case)}
""".strip()
        return self._structured_call(prompt, StateExtraction)

    def draft_sections(self, case: CaseData, authorities: list[dict[str, Any]]) -> DraftSections:
        prompt = f"""
נסח/י בעברית תקינה ומקצועית את חלקי התוכן של טיוטת תביעה קטנה.
הסתמך/י אך ורק על המצב והמקורות המצורפים. אל תוסיף/י עובדה, ראיה, נזק או סעיף חוק.
אין לכלול שמות, מספרי זיהוי, כתובות או פרטי קשר; התבנית המקומית תוסיף אותם.
הרקע העובדתי חייב להיות כרונולוגי ותמציתי ולכלול את האירועים שהובילו לתביעה,
לפי סדר התרחשותם. בתוך factual_background חובה להעתיק במדויק, ללא קיצור או
שינוי פורמט, את הערכים transaction_date, incident_date ו-incident_place כפי שהם
מופיעים במצב. מותר להוסיף סביבם ניסוח משפטי, אך אסור להחליף כתובת מלאה בשם העיר בלבד.
הפניות חוק מותרות רק לפי authority id שסופק.
אין להעתיק את תיאור הפתיחה מילה במילה, אין לכתוב בגוף ראשון, ואין לחזור על אותו משפט
גם ברקע העובדתי וגם בניסיונות הפתרון. יש להבחין במפורש בין סעד עיקרי לסעד חלופי.
אין לנסח או להסיק פרטי אחריות בתוך הרקע העובדתי; התבנית המקומית מוסיפה אותם בנפרד
בדיוק מתוך השדה המאומת warranty_details.

מצב:
{json.dumps(safe_case_context(case), ensure_ascii=False, default=str)}

מקורות מותרים:
{json.dumps(authorities, ensure_ascii=False)}
""".strip()
        return self._structured_call(prompt, DraftSections)

    def repair_answer(
        self,
        field: str,
        question: str,
        failed_answers: list[str],
        case: CaseData,
    ) -> AnswerRepair:
        redacted_answers = [redact_text(answer, case) for answer in failed_answers[-2:]]
        prompt = f"""
את/ה רכיב תיקון שיחה בראיון לתביעה קטנה. המשתמש לא הצליח פעמיים לענות בפורמט שהקוד המקומי מבין.
נתח/י רק את השדה הנוכחי ואת שתי התשובות האחרונות. אין להמציא עובדה, סכום, תאריך או כוונה.

אם ניתן לנסח מחדש בוודאות את אותה תשובה בלי להוסיף מידע, החזר/י action=accept,
repaired_answer כתשובה עברית קצרה שהמשתמש היה יכול להקליד, ו-confidence של 0.7 ומעלה.
אחרת החזר/י action=clarify ושאלת הבהרה אחת ממוקדת. אין להחזיר שמות, כתובות,
מספרי זיהוי, טלפונים, דוא״ל או מצייני מקום מרובעים.

שדה: {field}
השאלה: {question}
שתי התשובות המושחרות: {json.dumps(redacted_answers, ensure_ascii=False)}
מצב מושחר: {json.dumps(safe_case_context(case), ensure_ascii=False, default=str)}
""".strip()
        return self._structured_call(prompt, AnswerRepair)

    def review_interview(self, case: CaseData) -> InterviewConsistencyReview:
        prompt = f"""
בדוק/י את מצב הראיון לאחר איסוף הפרטים ולפני אישור העובדות ויצירת הטיוטה.
התאריך הנוכחי לפי המערכת המקומית הוא {date.today().isoformat()}. אין לסמן תאריך
כעתידי אם הוא חל בתאריך זה או לפניו; אין להסתמך על תאריך ידע פנימי של המודל.
מותר להצביע רק על סתירה או עמימות מהותית שכבר קיימת: סדר תאריכים בלתי אפשרי,
פירוט כספי שנראה כסכום כולל לעומת רכיבים, סעד עיקרי וחלופי שאינם מובחנים,
או סתירה בין תיאור האירוע לניסיונות הפתרון. אין להכריע זכאות משפטית ואין לבקש PII.

אם אין בעיה מהותית, החזר/י verdict=pass ורשימת issues ריקה.
אם דרושה הבהרה, החזר/י לכל היותר שתי בעיות. field חייב להיות אחד מאלה:
transaction_date, incident_date, product_or_service_details, prior_attempts,
damage_items, primary_remedy, alternative_remedy, warranty_details.
כל clarification_question חייבת להיות שאלה אחת קצרה בעברית.

מצב מושחר:
{json.dumps(safe_case_context(case), ensure_ascii=False, default=str)}
""".strip()
        return self._structured_call(prompt, InterviewConsistencyReview)

    def review(self, case: CaseData, draft: str, authority_ids: set[str]) -> DraftReview:
        prompt = f"""
בדוק/י טיוטת תביעה קטנה כבקר/ית איכות בלבד.
בדוק/י התאמה לעובדות, עקביות, ניסוח והיעדר טענות או מקורות שלא סופקו.
אל תכריע/י זכאות משפטית ואל תשנה/י סכומים. אם נדרש תיקון, ספק/י הוראות קצרות.
אין לבקש להמציא פרטי זיהוי או קשר שהמשתמש מסר שאינם ידועים.
מצייני מקום מרובעים כגון [PLAINTIFF_NAME] ו-[DEFENDANT_ADDRESS] הם השחרת פרטיות בלבד.
מציין כזה מופיע רק כאשר קיים ערך; שדה ריק במצב פירושו שהערך לא נמסר. אין לדווח על
החלפה, השמטה או אי-התאמה בין מצייני מקום, משום שפרטי הצדדים נבדקים בקוד מקומי.
ניסיונות הפנייה והסירוב מופיעים בפרק נפרד בשם "ניסיונות ליישוב המחלוקת". אין לדרוש
להעתיק אותם גם לפרק "השתלשלות האירועים" ואין לראות בהפרדה זו השמטת עובדה.
כתב התביעה מנוסח במכוון בגוף שלישי ניטרלי באמצעות "התובע/ת" ו"הנתבע/ת". זהו סגנון
המסמך שנבחר ואין לסמן עצם השימוש בגוף שלישי כבעיית בהירות או כהוראת תיקון.
אין להסיק מאמירת המשתמש שלא נמסרה או לא קיימת אחריות כי תקנות הגנת הצרכן
(אחריות ושירות לאחר מכירה) אינן חלות. אם התקנות נמצאות ברשימת המקורות המותרים,
יש לבדוק רק שהטיוטה מציגה את תחולתן האפשרית באופן מותנה בעובדות.

מצב נטול פרטים מזהים:
{json.dumps(safe_case_context(case), ensure_ascii=False, default=str)}

מזהי מקורות מותרים: {sorted(authority_ids)}

טיוטה מושחרת:
{redact_text(draft, case)}
""".strip()
        return self._structured_call(prompt, DraftReview)

    def _structured_call(self, prompt: str, schema):
        structured = self._base_model().with_structured_output(schema.model_json_schema(), method="json_schema")
        last_error: Exception | None = None
        request = prompt
        for attempt in range(self.max_retries + 1):
            try:
                raw = structured.invoke(request)
                if isinstance(raw, schema):
                    return raw
                return schema.model_validate(raw)
            except (ValidationError, ValueError, TypeError, KeyError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                request = (
                    prompt
                    + "\n\nהפלט הקודם לא תאם לסכמה. החזר/י אובייקט מלא התואם בדיוק לסכמה. "
                    + f"שגיאת אימות: {type(exc).__name__}"
                )
            except Exception as exc:  # provider, quota, timeout, or safety error
                raise ModelUnavailable(f"Gemini request failed: {type(exc).__name__}") from exc
        raise ModelUnavailable(f"Gemini returned invalid structured output: {type(last_error).__name__}")


class DemoAgentClient:
    """Deterministic fallback used when the Gemini API is unavailable."""

    available = False
    model_name = "manual-demo"

    def extract(self, user_text: str, case: CaseData) -> StateExtraction:
        from .legal import classify_category

        return StateExtraction(
            inferred_category=classify_category(user_text, case.category),
            candidate_question_topics=[],
        )

    def draft_sections(self, case: CaseData, authorities: list[dict[str, Any]]) -> DraftSections:
        facts = [
            f"ביום {_display_date(case.transaction_date)} רכש/ה התובע/ת מהנתבע/ת {case.product_or_service_details}, "
            f"בסכום של {_transaction_amount(case):,.0f} ₪, במקום העסקה: {case.incident_place}.",
            f"ביום {_display_date(case.incident_date)} {formal_event_fact(case)}",
        ]
        causes = controlled_cause_paragraphs(authorities, case)
        primary = _remedy_label(case.primary_remedy)
        relief = f"הסעד העיקרי המבוקש הוא {primary}."
        if case.alternative_remedy not in {RemedyType.NONE, RemedyType.UNDECIDED}:
            relief += f" לחלופין בלבד, מתבקש {_remedy_label(case.alternative_remedy)}."
        return DraftSections(
            factual_background=facts,
            cause_of_action=causes,
            resolution_attempts=(
                [formalize_resolution_attempt(attempt, case) for attempt in case.prior_attempts]
                or ["לא נמסרו ניסיונות יישוב קודמים."]
            ),
            relief_explanation=relief,
        )

    def repair_answer(
        self,
        field: str,
        question: str,
        failed_answers: list[str],
        case: CaseData,
    ) -> AnswerRepair:
        return AnswerRepair(
            action="clarify",
            clarification_question="לא הצלחתי לנרמל את התשובה בבטחה. נסו לנסח אותה מחדש.",
            confidence=0,
        )

    def review_interview(self, case: CaseData) -> InterviewConsistencyReview:
        return InterviewConsistencyReview(verdict="pass")

    def review(self, case: CaseData, draft: str, authority_ids: set[str]) -> DraftReview:
        return DraftReview(verdict="pass")


def _remedy_label(remedy: RemedyType) -> str:
    return {
        RemedyType.UNDECIDED: "סעד שטרם הוגדר",
        RemedyType.NONE: "אין סעד נוסף",
        RemedyType.REFUND: "השבת התמורה הכספית",
        RemedyType.CANCELLATION: "ביטול העסקה והשבת התמורה",
        RemedyType.REPAIR: "תיקון המוצר או השירות",
        RemedyType.REPLACEMENT: "החלפת המוצר במוצר תקין",
        RemedyType.OTHER: "הסעד האחר שתואר בראיון",
    }[remedy]


def _transaction_amount(case: CaseData):
    direct = next((item.amount for item in case.damage_items if item.kind == "direct"), None)
    return direct if direct is not None else case.claim_total


def _display_date(value: str) -> str:
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        year, month, day = value.split("-")
        return f"{int(day)}.{int(month)}.{year}"
    return value


def controlled_cause_paragraphs(
    authorities: list[dict[str, Any]], case: CaseData | None = None
) -> list[str]:
    authority_ids = {item["id"] for item in authorities}
    causes: list[str] = []
    if "sale-law-11" in authority_ids:
        causes.append("לטענת התובע/ת, הפגם הנטען עשוי להוות אי-התאמה בממכר לפי סעיף 11 לחוק המכר.")
    if "consumer-warranty-regulations" in authority_ids:
        causes.append(
            "מאחר שלפי התיאור מדובר במוצר חשמלי שמחירו עולה על 150 ₪, יש לבחון את תחולת "
            "תקנות הגנת הצרכן (אחריות ושירות לאחר מכירה), התשס״ו-2006. ככל שמדובר במוצר חדש "
            "והתקנות חלות, חובות האחריות והשירות עשויות לחול מכוח הדין גם אם לא נמסרה תעודת אחריות."
        )
    if "rent-loan-7" in authority_ids:
        causes.append(
            "ככל שמדובר בליקוי שבאחריות המשכיר, יש לבחון את חובת התיקון לפי סעיף 7 "
            "לחוק השכירות והשאילה, התשל״א-1971, בכפוף לתנאי החוק ולנסיבות המקרה."
        )
    if "rent-loan-25i-e" in authority_ids:
        causes.append(
            "ככל שהוראות שכירות למגורים חלות על ההסכם, סעיף 25י(ה) לחוק השכירות "
            "והשאילה, התשל״א-1971, מחייב את המשכיר להשיב את הערובה או את יתרתה "
            "בתוך המועד הקבוע בסעיף, בכפוף לחובות השוכר ולעילות המימוש המותרות בדין."
        )
    if "contracts-remedies-2" in authority_ids:
        causes.append(
            "ככל שתוכח הפרת ההסכם, סעיף 2 לחוק החוזים (תרופות בשל הפרת חוזה), התשל״א-1970, "
            "מאפשר לנפגע לבקש, בהתאם לנסיבות, אכיפה, ביטול או פיצויים."
        )
    if "contracts-remedies-9" in authority_ids:
        causes.append(
            "אם ייקבע שהעסקה בוטלה כדין, סעיף 9 לחוק החוזים (תרופות בשל הפרת חוזה), "
            "התשל״א-1970, מסדיר את השבת מה שקיבל כל צד על פי החוזה."
        )
    if "contracts-remedies-10" in authority_ids:
        causes.append(
            "רכיבי הנזק הנוספים כפופים להוכחת קשר סיבתי וצפיות לפי סעיף 10 לחוק החוזים "
            "(תרופות בשל הפרת חוזה), התשל״א-1970."
        )
    if "contracts-remedies-13" in authority_ids:
        causes.append(
            "הפיצוי המבוקש בגין נזק שאינו ממוני נתון לשיקול דעת בית המשפט לפי סעיף 13 "
            "לחוק החוזים (תרופות בשל הפרת חוזה), התשל״א-1970."
        )
    return causes
