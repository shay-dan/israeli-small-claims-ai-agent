from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from small_claims_agent.drafting import compose_claim_markdown, compose_source_report_markdown
from small_claims_agent.legal import retrieve_authorities_for_case
from small_claims_agent.llm import DemoAgentClient
from small_claims_agent.pdf_export import save_pdf
from small_claims_agent.sample_data import air_conditioner_sample
from small_claims_agent.validators import validate_draft


PDF_DIR = ROOT / "output" / "pdf"
EXAMPLE_DIR = ROOT / "output" / "examples"

CLAIM_BASENAME = "demo_small_claim_court_draft"
SOURCE_REPORT_BASENAME = "demo_ai_sources_and_quality_report"
DESIGN_EXPLANATION_BASENAME = "design_explanation_he"


def _write_markdown(markdown: str, basename: str) -> Path:
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    path = EXAMPLE_DIR / f"{basename}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def build_demo_outputs() -> list[Path]:
    """Create the clean court draft and the separate RAG/QC report."""
    case = air_conditioner_sample()
    authorities = retrieve_authorities_for_case(case)
    sections = DemoAgentClient().draft_sections(
        case,
        [authority.model_dump(mode="json") for authority in authorities],
    )

    claim_markdown = compose_claim_markdown(case, sections, authorities)
    validation_issues = validate_draft(
        claim_markdown,
        {authority.id for authority in authorities},
        case,
    )
    if validation_issues:
        raise RuntimeError("Deterministic demo failed local validation: " + " ".join(validation_issues))
    source_report_markdown = compose_source_report_markdown(
        case,
        authorities,
        {
            "verdict": "pass",
            "factual_consistency_issues": [],
            "unsupported_assertions": [],
            "clarity_issues": [],
            "revision_instructions": [],
        },
        [],
        "deterministic-demo (no Gemini call)",
    )

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    claim_md_path = _write_markdown(claim_markdown, CLAIM_BASENAME)
    claim_pdf_path = save_pdf(
        claim_markdown,
        PDF_DIR / f"{CLAIM_BASENAME}.pdf",
        title="טיוטת כתב תביעה",
    )
    source_md_path = _write_markdown(source_report_markdown, SOURCE_REPORT_BASENAME)
    source_pdf_path = save_pdf(
        source_report_markdown,
        PDF_DIR / f"{SOURCE_REPORT_BASENAME}.pdf",
        title="דו״ח מקורות ובקרת איכות",
    )
    return [claim_md_path, claim_pdf_path, source_md_path, source_pdf_path]


def build_design_explanation() -> Path:
    markdown = """# סוכן בינה מלאכותית להכנת טיוטת תביעה קטנה

## תהליך האפיון

העבודה החלה בניתוח המטלה ובהגדרת מוצר ראשוני בר שימוש שניתן להשלים בתוך כשבוע: מערכת שמראיינת בעברית אדם שאינו משפטן, משלימה מידע חסר, מציגה סיכום לאישור ומפיקה טיוטת כתב תביעה הניתנת לעריכה.

נעזרתי בעוזר בינה מלאכותית כשותף לתכנון ולפיתוח. דרישות המוצר והבדיקות גובשו באופן איטרטיבי: בכל הרצה נבחנה השיחה מנקודת מבט המשתמש ונוספו תיקונים. הוחלט שכל הקליטה תתבצע בצ׳אט, בשאלה ממוקדת אחת בכל תור, תוך שימוש במידע שכבר נמסר ואישור העובדות לפני הניסוח.

## כלים והחלטות מרכזיות

**ממשק:**
Streamlit נבחר לבניית הצ׳אט, עורך הטיוטה והורדת הקבצים בפייתון בלבד. חלופת FastAPI עם React מעניקה שליטה רחבה יותר במוצר ייצור, אך מגדילה את היקף הפיתוח.

**תזמור ומצב:**
LangGraph מנהל את שלבי החילוץ, האימות, האחזור, הניסוח והבקרה. Pydantic מגדיר את CaseData ומאמת כל מעבר. נבחר סוכן יחיד ולא Multi-Agent משום שהתהליך סדרתי ורגיש לעקביות.

**מודל וחיבור:**
Gemini API נבחר כדי לעמוד באילוץ העלות. LangChain Google GenAI מספק פלט מובנה ומחבר את המודל לגרף. Google Gen AI SDK הוא תלות התשתית של המתאם, ו-python-dotenv טוענת את המפתח המקומי. OpenAI ו-Claude היו חלופות אפשריות אך לא נדרשו בגרסה זו.

**אחזור משפטי:**
Targeted RAG מאחזר רשומות ממאגר JSON קטן, סגור ושקוף. כל רשומה מסמנת אם המקור רשמי או ציבורי ואם הטקסט הוא ציטוט או תמצית. Vector DB לא נבחר מפני שהקורפוס מצומצם וסיווג מבוסס כללים קל יותר לבקרה.

**יצוא:**
ReportLab יוצר את מסמכי ה-PDF ו-python-bidi מטפלת בכיוון הכתיבה. גופן Rubik מצורף לפרויקט. WeasyPrint או Chromium היו חלופות לעיצוב מורכב יותר, במחיר תלויות מערכת נוספות.

## פרטיות, אימות ומקרי קצה

פרטי זיהוי וקשר נאספים בשדות מקומיים ייעודיים ומושחרים מהקשר המודל. הודעת הפתיחה נסרקת לפני קריאת מודל כדי לזהות פרטי זיהוי נפוצים. תאריכים, סכומים, תקרת תביעה, אגרה, סמכות ומקורות נבדקים בקוד דטרמיניסטי. פלט שגוי מקבל מספר ניסיונות מוגבל ולאחריהם מסלול מקומי; כל הצעת תיקון של המודל חייבת לעבור שוב את האימות המקומי.

המערכת מפיקה שני מסמכים: כתב תביעה נקי להגשה לאחר בדיקת המשתמש, ודו״ח אופציונלי נפרד המציג את המקורות ואת בקרת האיכות. זהו מוצר הדגמה לימודי ולא ייעוץ משפטי.

"""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    return save_pdf(
        markdown,
        PDF_DIR / f"{DESIGN_EXPLANATION_BASENAME}.pdf",
        "הסבר תכנון ופיתוח",
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    paths = [build_design_explanation(), *build_demo_outputs()]
    for path in paths:
        print(path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
