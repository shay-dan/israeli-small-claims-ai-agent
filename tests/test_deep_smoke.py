from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfReader

from small_claims_agent.drafting import formal_case_description
from small_claims_agent.graph import build_graph, initial_state
from small_claims_agent.graph import _sanitize_draft_review
from small_claims_agent.interview import (
    InterviewAnswerError,
    apply_chat_answer,
    opening_story_is_complete,
    seed_case_from_opening,
)
from small_claims_agent.llm import DemoAgentClient
from small_claims_agent.models import CaseData, ClaimCategory, DraftReview
from small_claims_agent.pdf_export import markdown_to_pdf_bytes
from small_claims_agent.sample_data import air_conditioner_sample
from small_claims_agent.validators import validate_case, validate_draft


SCENARIOS = (
    {
        "id": "physical_laptop",
        "opening": (
            "קניתי לפטופ ב4000 שקל, פתחתי את הקופסה בחנות וראיתי שמסך הלפטופ שבור. "
            "ניסיתי להחזיר אותו והחנות טענה שאני שברתי אותו וסירבה להחזיר כסף."
        ),
        "product": "לפטופ",
        "place": "חנות KSP רחוב ירושלים 3 תל אביב",
        "defendant": "KSP",
        "role": "מוכר, יבואן ונותן שירות",
        "address": "רחוב ירושלים 3 תל אביב",
        "attempt": "הגעתי לחנות וביקשתי החזר כספי והחנות סירבה וטענה שאני שברתי את המוצר",
        "damages": "השבת מחיר המוצר - 4000",
        "primary": "החזר כספי וביטול העסקה",
        "alternative": "החלפה",
        "warranty": "כן, אחריות לשנתיים",
        "evidence": "קבלה\nתעודת אחריות\nצילום של המסך השבור",
        "category": ClaimCategory.CONSUMER_PRODUCT,
        "court": "תל אביב-יפו",
        "authority_ids": {"sale-law-11", "consumer-warranty-regulations", "contracts-remedies-2", "contracts-remedies-9"},
    },
    {
        "id": "online_course",
        "opening": (
            "שילמתי 20000 שקל על קורס ללימודים בהייטק. הובטח שאוכל להתחרט בתוך שבועיים, "
            "אבל אחרי יומיים ביקשתי לבטל והחברה סירבה כי נכנסתי לאתר."
        ),
        "product": "קורס ללימודים בהייטק",
        "place": "עסקה מקוונת באתר COURSES.co.il",
        "defendant": "COURSES",
        "role": "נותן שירות",
        "address": "רחוב המדע 10 ירושלים",
        "attempt": "שלחתי לחברה מייל יומיים לאחר העסקה וביקשתי לבטל והיא סירבה להשיב את הכסף",
        "damages": "השבת מחיר הקורס - 20000",
        "primary": "ביטול העסקה והחזר כספי",
        "alternative": "אין",
        "warranty": "לא ידוע",
        "evidence": "קבלה\nתנאי ההרשמה\nהתכתבות במייל",
        "category": ClaimCategory.SERVICE_CONTRACT,
        "court": "ירושלים",
        "authority_ids": {"contracts-remedies-2", "contracts-remedies-9"},
    },
    {
        "id": "telephone_cleaning_service",
        "opening": (
            "הזמנתי בטלפון שירות ניקיון ב2500 שקל, המנקים לא הגיעו במועד והחברה מסרבת להחזיר את הכסף."
        ),
        "product": "שירות ניקיון",
        "place": "עסקה טלפונית",
        "defendant": "נקי עכשיו בע״מ",
        "role": "נותן שירות",
        "address": "דרך הים 8 חיפה",
        "attempt": "צלצלתי למוקד שלוש פעמים וביקשתי החזר והחברה סירבה",
        "damages": "השבת מחיר השירות - 2500",
        "primary": "החזר כספי וביטול העסקה",
        "alternative": "אין",
        "warranty": "לא ידוע",
        "evidence": "אישור הזמנה\nפירוט שיחות",
        "category": ClaimCategory.SERVICE_CONTRACT,
        "court": "חיפה",
        "authority_ids": {"contracts-remedies-2", "contracts-remedies-9"},
    },
    {
        "id": "restaurant_duplicate_charge",
        "opening": "במסעדה חויבתי פעמיים על ארוחה ושילמתי בסך הכול 20000 שקל במקום 10000 שקל.",
        "product": "ארוחה במסעדה",
        "place": "מסעדת יוסי ובניו קניון מלחה ירושלים",
        "defendant": "מסעדת יוסי ובניו",
        "role": "נותן שירות",
        "address": "קניון מלחה ירושלים",
        "attempt": "הגעתי למסעדה ואמרתי שחויבתי פעמיים והמנהל סירב לזכות את החיוב העודף",
        "damages": "השבת החיוב הכפול העודף - 10000",
        "primary": "החזר כספי",
        "alternative": "אין",
        "warranty": "לא ידוע",
        "evidence": "קבלה\nפירוט כרטיס אשראי",
        "category": ClaimCategory.SERVICE_CONTRACT,
        "court": "ירושלים",
        "authority_ids": {"contracts-remedies-2", "contracts-remedies-9"},
    },
    {
        "id": "rental_deposit",
        "opening": "שכרתי דירה והעברתי למשכיר פיקדון של 5000 שקל, ובסיום השכירות הוא מסרב להחזיר אותו.",
        "product": "שכירת דירה",
        "place": "רחוב הרצל 4 ראשון לציון",
        "defendant": "דמות בדיונית משכיר",
        "role": "משכיר",
        "address": "רחוב הרצל 4 ראשון לציון",
        "attempt": "שלחתי למשכיר הודעת ווטסאפ וביקשתי את הפיקדון בחזרה והוא סירב",
        "damages": "השבת הפיקדון - 5000",
        "primary": "החזר כספי",
        "alternative": "אין",
        "warranty": "לא ידוע",
        "evidence": "חוזה שכירות\nאסמכתה להעברת הפיקדון\nהתכתבות ווטסאפ",
        "category": ClaimCategory.LANDLORD_TENANT,
        "court": "ראשון לציון",
        "authority_ids": {"rent-loan-25i-e", "contracts-remedies-2", "contracts-remedies-9"},
    },
    {
        "id": "generic_football",
        "opening": "קניתי כדור כדורגל ב100 שקל ואחרי יום הוא התפוצץ בשימוש רגיל.",
        "product": "כדור כדורגל",
        "place": "חנות ספורט העיר רחוב העצמאות 2 באר שבע",
        "defendant": "ספורט העיר",
        "role": "מוכר",
        "address": "רחוב העצמאות 2 באר שבע",
        "attempt": "באתי לחנות וביקשתי להחליף את הכדור והמוכר סירב",
        "damages": "השבת מחיר המוצר - 100",
        "primary": "החזר כספי",
        "alternative": "החלפה",
        "warranty": "לא ידוע",
        "evidence": "קבלה\nצילום של הכדור",
        "category": ClaimCategory.CONSUMER_PRODUCT,
        "court": "באר שבע",
        "authority_ids": {"sale-law-11", "contracts-remedies-2", "contracts-remedies-9"},
    },
)


def _completed_case(scenario: dict) -> CaseData:
    case = seed_case_from_opening(scenario["opening"])
    answers = (
        ("transaction_date", "1.1.26"),
        ("incident_date", "2.1.26"),
        ("incident_place", scenario["place"]),
        ("product_or_service_details", scenario["product"]),
        ("defendant.full_name", scenario["defendant"]),
        ("defendant_role", scenario["role"]),
        ("defendant.identifier", "לא ידוע"),
        ("defendant.address", scenario["address"]),
        ("defendant.phone", "לא ידוע"),
        ("prior_attempts", scenario["attempt"]),
        ("damage_items", scenario["damages"]),
        ("primary_remedy", scenario["primary"]),
        ("alternative_remedy", scenario["alternative"]),
        ("warranty_details", scenario["warranty"]),
        ("evidence_confirmation", scenario["evidence"]),
        ("claimant.full_name", "תובע בדיקה"),
        ("claimant.identifier", "123456789"),
        ("claimant.address", "רחוב הבדיקה 7 " + scenario["court"].replace("-יפו", "")),
        ("claimant.phone", "0501234567"),
        ("claimant.email", "test@example.com"),
    )
    for field, answer in answers:
        case = apply_chat_answer(case, field, answer)
    data = case.model_dump(mode="python")
    data["facts_confirmed"] = True
    return CaseData.model_validate(data)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item["id"])
def test_complete_scenarios_generate_valid_draft_and_pdf(scenario):
    case = _completed_case(scenario)
    validation = validate_case(case)

    assert validation.ready, (validation.missing_fields, validation.errors)
    assert case.category == scenario["category"]
    assert case.court_city == scenario["court"]

    graph = build_graph(DemoAgentClient())
    state = initial_state(case)
    state["generate_requested"] = True
    result = graph.invoke(state, {"configurable": {"thread_id": "deep-" + scenario["id"]}})

    assert not result["validation_errors"]
    assert not validate_draft(result["draft_markdown"], scenario["authority_ids"], case)
    assert {item["id"] for item in result["authorities"]} == scenario["authority_ids"]
    assert case.product_or_service_details in result["draft_markdown"]
    assert case.defendant.full_name in result["draft_markdown"]
    assert case.defendant.address in result["draft_markdown"]

    pdf = markdown_to_pdf_bytes(result["draft_markdown"])
    reader = PdfReader(BytesIO(pdf))
    assert pdf.startswith(b"%PDF")
    assert len(reader.pages) >= 1


def test_draft_edit_cannot_change_confirmed_total_without_detection():
    scenario = SCENARIOS[0]
    case = _completed_case(scenario)
    graph = build_graph(DemoAgentClient())
    state = initial_state(case)
    state["generate_requested"] = True
    result = graph.invoke(state, {"configurable": {"thread_id": "deep-amount-edit"}})
    corrupted = result["draft_markdown"].replace("4,000 ₪", "40,000 ₪")

    issues = validate_draft(corrupted, scenario["authority_ids"], case)

    assert any("סכום" in issue or "רכיב" in issue for issue in issues)


def test_pdf_keeps_declaration_and_signature_on_the_same_page():
    scenario = next(item for item in SCENARIOS if item["id"] == "rental_deposit")
    case = _completed_case(scenario)
    state = initial_state(case)
    state["generate_requested"] = True
    result = build_graph(DemoAgentClient()).invoke(
        state,
        {"configurable": {"thread_id": "deep-declaration-pagination"}},
    )
    reader = PdfReader(BytesIO(markdown_to_pdf_bytes(result["draft_markdown"])))
    page_texts = [page.extract_text() or "" for page in reader.pages]

    declaration_pages = [index for index, text in enumerate(page_texts) if "הצהרה" in text]
    # ReportLab/PyPDF may omit the word next to long underscore runs, while the
    # date label from the same signature row is extracted reliably.
    signature_pages = [index for index, text in enumerate(page_texts) if "תאריך" in text]
    assert declaration_pages
    assert declaration_pages == signature_pages


@pytest.mark.parametrize(
    ("opening", "product", "category"),
    (
        ("קניתי נעליים ב300 שקל ואחרי יומיים הסוליה נקרעה", "נעליים", ClaimCategory.CONSUMER_PRODUCT),
        ("קניתי אוזניות ב500 שקל ואחרי שעה הפסיקו לעבוד", "אוזניות", ClaimCategory.CONSUMER_PRODUCT),
        ("רכשתי שולחן ב1200 שקל והוא הגיע סדוק", "שולחן", ClaimCategory.CONSUMER_PRODUCT),
        ("קניתי שעון ב700 שקל ואחרי שבוע נשבר", "שעון", ClaimCategory.CONSUMER_PRODUCT),
        ("הזמנתי שירות צילום ב2500 שקל והצלם לא הגיע", "שירות צילום", ClaimCategory.SERVICE_CONTRACT),
    ),
)
def test_opening_parser_is_not_limited_to_a_fixed_product_catalog(opening, product, category):
    case = seed_case_from_opening(opening)

    assert opening_story_is_complete(opening)
    assert case.product_or_service_details == product
    assert case.category == category


@pytest.mark.parametrize(
    ("entered", "normalized"),
    (
        ("1.1.26", "01.01.2026"),
        ("01/1/26", "01.01.2026"),
        ("1-01-2026", "01.01.2026"),
        ("2026-1-1", "01.01.2026"),
        ("01012026", "01.01.2026"),
        ("01.0126", "01.01.2026"),
    ),
)
def test_supported_date_variants_normalize_identically(entered, normalized):
    case = apply_chat_answer(CaseData(), "transaction_date", entered)
    assert case.transaction_date == normalized


@pytest.mark.parametrize("entered", ("31.2.26", "0.1.26", "1.13.26", "32/1/26", "abc"))
def test_impossible_or_unparseable_dates_are_rejected(entered):
    with pytest.raises(InterviewAnswerError):
        apply_chat_answer(CaseData(), "transaction_date", entered)


def test_formal_description_preserves_stopped_operating_wording():
    description = formal_case_description(air_conditioner_sample())

    assert "הפסיק לפעול" in description
    assert "המזגן הפסיק לפעול" in description
    assert "במהלך שימוש רגיל" in description
    assert "200 ₪" not in description
    assert "שנרכש תמורת" not in description
    assert "התגלעה מחלוקת" not in description


def test_quality_judge_cannot_require_contact_attempts_twice():
    case = air_conditioner_sample()
    review = DraftReview(
        verdict="revise",
        factual_consistency_issues=[
            "בפרק השתלשלות האירועים חסרות פניות התובעת וסירובה של הנתבעת."
        ],
        revision_instructions=[
            "יש להוסיף להשתלשלות האירועים את הפניות ואת סירובה של הנתבעת."
        ],
    )

    cleaned = _sanitize_draft_review(review, case, set())

    assert cleaned.verdict == "pass"
    assert not cleaned.factual_consistency_issues
    assert not cleaned.revision_instructions


def test_quality_judge_accepts_intentional_third_person_court_style():
    case = air_conditioner_sample()
    review = DraftReview(
        verdict="revise",
        clarity_issues=["הניסוח מתייחס לתובע/ת בגוף שלישי ואינו מתאים."],
        revision_instructions=["יש לשנות את כתב התביעה מגוף שלישי לגוף ראשון."],
    )

    cleaned = _sanitize_draft_review(review, case, set())

    assert cleaned.verdict == "pass"
    assert not cleaned.clarity_issues
    assert not cleaned.revision_instructions


def test_streamlit_online_course_reaches_confirmation_without_loop(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    # Keep UI tests deterministic even when the developer has real provider
    # credentials in .env. load_dotenv() must not activate live API calls here.
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(key, "")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    messages = (
        (
            "שילמתי 20000 שקל על קורס ללימודים בהייטק והבטיחו שאוכל להתחרט בתוך שבועיים. "
            "אחרי יומיים ביקשתי לבטל והחברה סירבה כי כבר נכנסתי לאתר."
        ),
        "1.1.26",
        "3.1.26",
        "עסקה מקוונת באתר COURSES.co.il",
        "COURSES",
        "נותן שירות",
        "לא ידוע",
        "רחוב המדע 10 ירושלים",
        "לא ידוע",
        "שלחתי מייל וביקשתי לבטל והחברה סירבה להחזיר את הכסף",
        "20000",
        "ביטול העסקה והחזר כספי",
        "אין",
        "לא ידוע",
        "קבלה, תנאי הרשמה, התכתבות במייל",
        "תובע בדיקה",
        "123456789",
        "רחוב הבדיקה 7 ירושלים",
        "0501234567",
        "test@example.com",
    )
    for message in messages:
        app.chat_input[0].set_value(message).run()

    case = app.session_state["case"]
    assert not app.exception
    assert app.session_state["awaiting_confirmation"]
    assert case.category == ClaimCategory.SERVICE_CONTRACT
    assert case.court_city == "ירושלים"
    assert case.defendant.address == "רחוב המדע 10 ירושלים"
    assert case.claim_total == 20000


def test_streamlit_restaurant_duplicate_charge_keeps_only_excess_amount(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    # Keep UI tests deterministic even when the developer has real provider
    # credentials in .env. load_dotenv() must not activate live API calls here.
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(key, "")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    messages = (
        "במסעדה חויבתי פעמיים על ארוחה ושילמתי בסך הכול 20000 שקל במקום 10000 שקל",
        "1.1.26",
        "1.1.26",
        "מסעדת יוסי ובניו קניון מלחה ירושלים",
        "ארוחה במסעדה",
        "מסעדת יוסי ובניו",
        "נותן שירות",
        "לא ידוע",
        "קניון מלחה ירושלים",
        "לא ידוע",
        "הגעתי למסעדה וביקשתי זיכוי והמנהל סירב",
        "השבת החיוב הכפול העודף 10000",
        "החזר כספי",
        "אין",
        "לא ידוע",
        "קבלה, פירוט כרטיס אשראי",
        "תובע בדיקה",
        "123456789",
        "רחוב הבדיקה 7 ירושלים",
        "0501234567",
        "אין",
    )
    for message in messages:
        app.chat_input[0].set_value(message).run()

    case = app.session_state["case"]
    assert not app.exception
    assert app.session_state["awaiting_confirmation"]
    assert case.category == ClaimCategory.SERVICE_CONTRACT
    assert case.claim_total == 10000
    assert case.damage_items[0].label == "השבת החיוב הכפול העודף"
