from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

from pydantic import BaseModel, Field, ValidationError

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


class DraftClarificationIssue(BaseModel):
    field: str
    case_excerpt: str
    draft_excerpt: str
    clarification_question: str


class DraftClarificationReview(BaseModel):
    issues: list[DraftClarificationIssue] = Field(default_factory=list, max_length=3)


class ModelUnavailable(RuntimeError):
    """Raised when a hosted provider cannot provide a validated response."""

    def __init__(self, message: str, diagnostic: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic or {}


def _provider_error_diagnostic(exc: Exception, model_name: str, operation: str) -> dict[str, Any]:
    """Build a useful provider diagnostic without retaining prompts, responses, or secrets."""
    chain: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    status_code: int | None = None
    while current is not None and id(current) not in seen and len(chain) < 6:
        seen.add(id(current))
        chain.append(type(current).__name__)
        response = getattr(current, "response", None)
        raw_status = getattr(current, "status_code", None) or getattr(response, "status_code", None)
        if raw_status is None:
            code = getattr(current, "code", None)
            raw_status = getattr(code, "value", None)
        if isinstance(raw_status, int):
            status_code = raw_status
        current = current.__cause__ or current.__context__

    names = " ".join(chain).lower()
    if status_code in {401, 403} or any(term in names for term in ("unauthorized", "permissiondenied", "authentication")):
        category, hint = "authentication", "בדקו שהמפתח תקין, פעיל ומורשה ל-Gemini API."
    elif status_code == 429 or "resourceexhausted" in names or "ratelimit" in names:
        category, hint = "quota_or_rate_limit", "נראה שהמכסה או קצב הבקשות הוגבלו; בדקו מכסה וחיוב בפרויקט Google AI."
    elif status_code == 404 or "notfound" in names:
        category, hint = "model_not_found", "בדקו את מזהה המודל שהוגדר ואת זמינותו לחשבון."
    elif status_code == 400 or "badrequest" in names or "invalidargument" in names:
        category, hint = "invalid_request", "השרת דחה את מבנה הבקשה או את אפשרויות המודל."
    elif any(term in names for term in ("connecterror", "connecttimeout", "socketerror", "sslerror", "network")):
        category, hint = "connection", "לא נוצר חיבור ל-Gemini; בדקו רשת, Proxy, חומת אש ו-TLS."
    elif any(term in names for term in ("timeout", "timedout")):
        category, hint = "timeout", "הבקשה חרגה מזמן ההמתנה; בדקו חיבור ועומס ונסו שוב."
    else:
        category, hint = "provider_error", "בדקו את פרטי השגיאה ביומן; לא נשמרו בו תוכן השיחה או סודות."

    retryable = category not in {"invalid_request", "structured_output_invalid"}
    return {
        "category": category,
        "retryable": retryable,
        "operation": operation,
        "model": model_name,
        "exception_chain": chain,
        "http_status": status_code,
        "hint": hint,
    }


class GeminiAgentClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 1,
        timeout_seconds: int = 35,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self._model = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _base_model(self):
        if not self.available:
            raise ModelUnavailable(
                "Gemini request failed: api_key_missing",
                diagnostic={
                    "category": "api_key_missing",
                    "operation": "initialize_model",
                    "model": self.model_name,
                    "exception_chain": [],
                    "http_status": None,
                    "hint": "לא הוגדר GEMINI_API_KEY בסביבת ההרצה.",
                },
            )
        if self._model is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ModelUnavailable(
                    "Gemini request failed: dependency_missing",
                    diagnostic={
                        "category": "dependency_missing",
                        "operation": "initialize_model",
                        "model": self.model_name,
                        "exception_chain": [type(exc).__name__],
                        "http_status": None,
                        "hint": "התקנת langchain-google-genai חסרה בסביבה הפעילה.",
                    },
                ) from exc
            self._model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                max_retries=0,
                timeout=self.timeout_seconds,
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
בצע/י בקרת איכות מלאה לטיוטת תביעה קטנה. קרא/י את כל המסמך מתחילתו ועד סופו ופרסר/י
את השדות המופיעים בו: תאריכים, מוצר/שירות, מקום, סכומים ורכיביהם, ניסיונות פנייה,
סעדים, אחריות, ראיות, סמכות והפניות לחוק. השווה/י כל הופעה של עובדה למצב המאושר,
וחפש/י סתירות בין חלקים, ערכים שהוחלפו, עובדות לא נתמכות או מידע חשוב שנשמט.
אל תכריע/י זכאות משפטית ואל תשנה/י עובדות או סכומים. אם נדרש תיקון, ספק/י הוראות קצרות
ומדויקות הכוללות את הכותרת והערך הבעייתי, בלי להמציא ערך חלופי.
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

שמות, כתובות ופרטי קשר הושחרו במכוון ואסור להסיק מהם אי-התאמה. הם נבדקים גם בבדיקה
מקומית מול שדות התיק; התמקד/י בבדיקת העקביות בין שאר חלקי הטיוטה.

מצב נטול פרטים מזהים:
{json.dumps(safe_case_context(case), ensure_ascii=False, default=str)}

מזהי מקורות מותרים: {sorted(authority_ids)}

הטיוטה המלאה (פרטי זיהוי וכתובות מושחרים):
{redact_text(draft, case)}
""".strip()
        return self._structured_call(prompt, DraftReview)

    def review_draft_for_clarification(
        self, case: CaseData, draft: str, authority_ids: set[str]
    ) -> DraftClarificationReview:
        allowed_fields = {
            "event_summary", "transaction_date", "incident_date", "incident_place",
            "product_or_service_details", "prior_attempts", "damage_items",
            "primary_remedy", "alternative_remedy", "warranty_details", "evidence",
        }
        prompt = f"""
בצע/י מעבר ממוקד לפני הפקת PDF: השווה/י את הטיוטה לנתוני הראיון המאושרים.
החזר/י עד 3 שאלות רק על אי-התאמות מהותיות שהמשתמש חייב להכריע בהן, כגון עובדה
בטיוטה שאינה נתמכת בנתונים, עובדה מרכזית מהראיון שהושמטה, תאריך/סכום/סעד שסותרים
את הנתונים, או ניסוח עובדתי שמחליף את משמעות דברי המשתמש. אל תבקש/י הבהרה על סגנון,
ניסוח משפטי, העדפה עיצובית או נושא שניתן לבדוק בקוד.

כל issue חייב לכלול field אחד מהרשימה: {sorted(allowed_fields)}; case_excerpt ציטוט
קצר ומדויק מתוך ערך הראיון המושחר; draft_excerpt ציטוט קצר ומדויק מתוך הטיוטה המושחרת;
ו-clarification_question בעברית שמסבירה את הפער ומבקשת מהמשתמש לאשר או לתקן.
אל תמציא/י ציטוטים: אם אין שני ציטוטים שתומכים בפער, אל תחזיר/י issue.
אל תבקש/י מהמשתמש לחשוף מידע מזהה. שמות וכתובות מושחרים בכוונה ונבדקים מקומית.
סכומים וסעדים שאינם סותרים במפורש אינם עילה לשאלה. אין להסיק עובדות או זכאות משפטית.
אם אין בעיה מהותית, החזר/י רשימת issues ריקה.

נתוני ראיון מושחרים:
{json.dumps(safe_case_context(case), ensure_ascii=False, default=str)}

מזהי מקורות מותרים: {sorted(authority_ids)}
טיוטה מלאה מושחרת:
{redact_text(draft, case)}
""".strip()
        return self._structured_call(prompt, DraftClarificationReview)

    def revise_sections(
        self,
        case: CaseData,
        authorities: list[dict[str, Any]],
        draft: str,
        review: DraftReview,
    ) -> DraftSections:
        prompt = f"""
תקן/י את חלקי הניסוח העובדתיים של הטיוטה בעקבות בקרת האיכות. קרא/י את הטיוטה המלאה
ואת ממצאי הבדיקה, ונסח/י מחדש את factual_background, resolution_attempts ו-relief_explanation
כדי לתקן סתירות, השמטות וניסוח לא ברור. החזר/י את כל השדות של DraftSections.
הסתמך/י רק על מצב התיק והמקורות. אין להוסיף, להסיק או לשנות עובדה, תאריך, סכום, סעד,
ראיה או טענה. אין לגעת בפרטי הצדדים, בפרטי קשר, בסמכות או ברשימת המקורות; אלה נבנים
מהנתונים המקומיים. השאר/י מצייני פרטיות כפי שהם אם הם מופיעים בטיוטה.

מצב מאושר ומושחר:
{json.dumps(safe_case_context(case), ensure_ascii=False, default=str)}

מזהי מקורות מותרים: {sorted({item.get('id', '') for item in authorities})}
ממצאי בקרת איכות: {json.dumps(review.model_dump(mode='json'), ensure_ascii=False)}
טיוטה מלאה ומושחרת:
{redact_text(draft, case)}
""".strip()
        return self._structured_call(prompt, DraftSections)

    def _structured_call(self, prompt: str, schema):
        try:
            structured = self._base_model().with_structured_output(schema.model_json_schema(), method="json_schema")
        except ModelUnavailable:
            raise
        except Exception as exc:
            diagnostic = _provider_error_diagnostic(exc, self.model_name, schema.__name__)
            raise ModelUnavailable(
                f"Gemini request failed: {diagnostic['category']}", diagnostic=diagnostic
            ) from exc
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
                diagnostic = _provider_error_diagnostic(exc, self.model_name, schema.__name__)
                raise ModelUnavailable(
                    f"Gemini request failed: {diagnostic['category']}", diagnostic=diagnostic
                ) from exc
        raise ModelUnavailable(
            f"Gemini returned invalid structured output: {type(last_error).__name__}",
            diagnostic={
                "category": "structured_output_invalid",
                "retryable": False,
                "provider": "gemini",
                "model": self.model_name,
                "operation": schema.__name__,
                "exception_chain": [type(last_error).__name__] if last_error else [],
                "http_status": None,
            },
        )


class OpenAICompatibleAgentClient(GeminiAgentClient):
    """Agent client for providers exposing OpenAI-compatible chat completions."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = 0,
        timeout_seconds: int = 40,
    ):
        # Reuse the shared task prompts/schema methods without initializing Gemini.
        self.provider = provider
        self.api_key = api_key
        self.model_name = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if not self.available:
            raise ModelUnavailable(
                f"{self.provider} API key is missing",
                diagnostic={"category": "api_key_missing", "retryable": True, "provider": self.provider, "model": self.model_name},
            )
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ModelUnavailable(
                    "OpenAI-compatible SDK dependency is missing",
                    diagnostic={"category": "dependency_missing", "retryable": True, "provider": self.provider, "model": self.model_name, "exception_chain": [type(exc).__name__]},
                ) from exc
            headers = None
            if self.provider == "openrouter":
                headers = {"X-Title": "Israeli Small Claims AI Agent"}
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=headers,
                timeout=self.timeout_seconds,
                max_retries=0,
            )
        return self._client

    def _structured_call(self, prompt: str, schema):
        try:
            response = self._get_client().beta.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                timeout=self.timeout_seconds,
            )
            message = response.choices[0].message
            if message.parsed is not None:
                return message.parsed
            # A refusal or malformed structured response is an output/schema
            # problem, not a provider-availability failure; do not mask it by
            # silently switching vendors.
            raise ModelUnavailable(
                f"{self.provider} returned no parsed structured output",
                diagnostic={
                    "category": "structured_output_invalid",
                    # The request/schema was accepted, but this provider/model
                    # did not produce a parseable instance. Let the gateway
                    # try the next provider; malformed requests (HTTP 400)
                    # remain non-retryable in _openai_compatible_diagnostic.
                    "retryable": True,
                    "provider": self.provider,
                    "model": self.model_name,
                    "operation": schema.__name__,
                    "exception_chain": [],
                    "http_status": None,
                },
            )
        except ModelUnavailable:
            raise
        except Exception as exc:
            diagnostic = _openai_compatible_diagnostic(exc, self.provider, self.model_name, schema.__name__)
            raise ModelUnavailable(
                f"{self.provider} request failed: {diagnostic['category']}",
                diagnostic=diagnostic,
            ) from exc


def _openai_compatible_diagnostic(
    exc: Exception, provider: str, model_name: str, operation: str
) -> dict[str, Any]:
    status = getattr(exc, "status_code", None)
    chain: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 5:
        seen.add(id(current))
        chain.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    lowered_chain = " ".join(chain).lower()
    if status == 400:
        category, retryable = "invalid_request", False
    elif any(token in lowered_chain for token in ("validationerror", "jsondecode", "responsevalidation")):
        # A valid API request can still yield malformed model output. This is
        # provider-specific, so a different provider is a useful fallback.
        category, retryable = "structured_output_invalid", True
    elif status == 429:
        category, retryable = "quota_or_rate_limit", True
    elif status in {401, 403}:
        category, retryable = "authentication", True
    elif status == 404:
        category, retryable = "model_not_found", True
    elif status == 408 or any("timeout" in name.lower() for name in chain):
        category, retryable = "timeout", True
    elif isinstance(status, int) and status >= 500:
        category, retryable = "provider_error", True
    elif any(token in " ".join(chain).lower() for token in ("connection", "network", "connecterror")):
        category, retryable = "connection", True
    else:
        category, retryable = "provider_error", True
    return {
        "category": category,
        "retryable": retryable,
        "provider": provider,
        "model": model_name,
        "operation": operation,
        "exception_chain": chain,
        "http_status": status if isinstance(status, int) else None,
    }


class LLMGateway:
    """Provider-agnostic ordered LLM chain; local deterministic fallback lives in the graph."""

    def __init__(self, providers: list[Any] | None = None):
        if providers is None:
            providers = [
                OpenAICompatibleAgentClient(
                    "groq",
                    os.getenv("GROQ_API_KEY", ""),
                    os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                    os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                ),
                OpenAICompatibleAgentClient(
                    "openrouter",
                    os.getenv("OPENROUTER_API_KEY", ""),
                    os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                    os.getenv("OPENROUTER_MODEL", "openrouter/free"),
                ),
                GeminiAgentClient(),
            ]
        self.providers = [provider for provider in providers if provider.available]
        self.last_provider: str | None = None
        self.last_model: str | None = None
        self.last_attempts: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return bool(self.providers)

    @property
    def model_name(self) -> str:
        if self.last_provider and self.last_model:
            return f"{self.last_provider} / {self.last_model}"
        if self.providers:
            return " → ".join(provider.provider if hasattr(provider, "provider") else "Gemini" for provider in self.providers)
        return "local-demo"

    def _run(self, method: str, *args):
        attempts: list[dict[str, Any]] = []
        for provider in self.providers:
            name = getattr(provider, "provider", "gemini")
            model = provider.model_name
            try:
                result = getattr(provider, method)(*args)
                self.last_provider, self.last_model = name, model
                self.last_attempts = attempts
                return result
            except ModelUnavailable as exc:
                diagnostic = dict(exc.diagnostic)
                diagnostic.setdefault("provider", name)
                diagnostic.setdefault("model", model)
                attempts.append({key: diagnostic.get(key) for key in ("provider", "model", "category", "http_status")})
                if diagnostic.get("retryable") is False:
                    self.last_attempts = attempts
                    diagnostic["provider_attempts"] = attempts
                    raise ModelUnavailable(str(exc), diagnostic=diagnostic) from exc
            except (ValidationError, ValueError, TypeError) as exc:
                diagnostic = {
                    "category": "structured_output_invalid",
                    "retryable": False,
                    "provider": name,
                    "model": model,
                    "operation": method,
                    "exception_chain": [type(exc).__name__],
                    "http_status": None,
                }
                self.last_attempts = attempts + [{"provider": name, "model": model, "category": diagnostic["category"]}]
                diagnostic["provider_attempts"] = self.last_attempts
                raise ModelUnavailable("Provider output failed local schema validation", diagnostic=diagnostic) from exc
        self.last_attempts = attempts
        raise ModelUnavailable(
            "All configured LLM providers failed",
            diagnostic={
                "category": "all_providers_failed" if attempts else "no_provider_configured",
                "retryable": True,
                "provider": None,
                "model": None,
                "operation": method,
                "provider_attempts": attempts,
                "exception_chain": [],
                "http_status": None,
            },
        )

    def extract(self, *args):
        return self._run("extract", *args)

    def draft_sections(self, *args):
        return self._run("draft_sections", *args)

    def repair_answer(self, *args):
        return self._run("repair_answer", *args)

    def review_interview(self, *args):
        return self._run("review_interview", *args)

    def review_draft_for_clarification(self, *args):
        return self._run("review_draft_for_clarification", *args)

    def review(self, *args):
        return self._run("review", *args)

    def revise_sections(self, *args):
        return self._run("revise_sections", *args)


class DemoAgentClient:
    """Deterministic local fallback used when hosted providers are unavailable."""

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

    def review_draft_for_clarification(
        self, case: CaseData, draft: str, authority_ids: set[str]
    ) -> DraftClarificationReview:
        return DraftClarificationReview()

    def review(self, case: CaseData, draft: str, authority_ids: set[str]) -> DraftReview:
        return DraftReview(verdict="pass")

    def revise_sections(
        self,
        case: CaseData,
        authorities: list[dict[str, Any]],
        draft: str,
        review: DraftReview,
    ) -> DraftSections:
        return self.draft_sections(case, authorities)


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
