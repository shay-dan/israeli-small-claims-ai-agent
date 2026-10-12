from decimal import Decimal

from small_claims_agent.drafting import (
    compose_claim_markdown,
    compose_source_report_markdown,
    formalize_resolution_attempt,
)
from small_claims_agent.graph import _merge_updates, _sanitize_draft_review, build_graph, initial_state
from small_claims_agent.interview import (
    InterviewAnswerError,
    SAMPLE_OPENING,
    apply_chat_answer,
    build_confirmation_summary,
    build_live_draft_preview,
    build_repeated_failure_guidance,
    correction_command,
    extract_prior_attempt_from_opening,
    extract_purchased_subject,
    field_value_for_user,
    inferred_completed_fields,
    is_current_question_help_request,
    is_current_question_reset_request,
    is_recall_request,
    merge_damage_confirmation,
    merge_partial_answer,
    next_interview_field,
    normalize_opening_keyboard_layout,
    opening_clarification,
    opening_story_is_complete,
    opening_story_missing_parts,
    parse_damage_items,
    prior_attempt_fragment_is_relevant,
    question_for,
    seed_case_from_opening,
)
from small_claims_agent.legal import classify_category, retrieve_authorities, retrieve_authorities_for_case
from small_claims_agent.llm import DemoAgentClient, controlled_cause_paragraphs
from small_claims_agent.models import (
    AnswerRepair,
    CaseData,
    ClaimCategory,
    DamageItem,
    DraftReview,
    FieldUpdate,
    ConsistencyIssue,
    InterviewConsistencyReview,
    RemedyType,
    StateExtraction,
)
from small_claims_agent.privacy import free_text_privacy_findings, redact_text, safe_case_context
from small_claims_agent.pdf_export import markdown_to_pdf_bytes
from small_claims_agent.sample_data import air_conditioner_sample
from small_claims_agent.repair import attempt_llm_repair, consistency_issue_is_supported
from small_claims_agent.validators import calculate_filing_fee, select_questions, validate_case, validate_draft


def test_sample_is_ready_and_fee_is_separate():
    case = air_conditioner_sample()
    result = validate_case(case)
    assert result.ready
    assert result.claim_total == Decimal("4050")
    assert result.filing_fee == Decimal("50")


def test_limit_uses_all_monetary_components():
    case = air_conditioner_sample()
    case.damage_items.append(DamageItem(label="נזק נוסף", amount="36000", kind="other"))
    result = validate_case(case)
    assert result.claim_total == Decimal("40050")
    assert any("עולה על תקרת" in error for error in result.errors)


def test_fee_rounds_up_and_has_minimum():
    assert calculate_filing_fee(Decimal("3200")) == Decimal("50")
    assert calculate_filing_fee(Decimal("12345")) == Decimal("124")


def test_negative_damage_amount_is_rejected():
    try:
        parse_damage_items("החזר -100")
    except InterviewAnswerError as exc:
        assert "שלילי" in str(exc)
    else:
        raise AssertionError("negative damage was accepted")


def test_contacting_store_by_phone_and_email_does_not_invent_physical_visit():
    case = CaseData(product_or_service_details="מזגן")
    result = formalize_resolution_attempt(
        "פניתי לחנות שלוש פעמים בטלפון ובמייל, ביקשתי החזר והחנות סירבה",
        case,
    )
    assert "בטלפון" in result and "בדוא״ל" in result
    assert "באופן פיזי" not in result


def test_questions_are_limited_to_two():
    questions = select_questions(["claimant.full_name", "defendant.address", "event_summary"])
    assert len(questions) == 2


def test_pii_is_removed_from_outbound_context():
    case = air_conditioner_sample()
    text = f"אני {case.claimant.full_name}, הטלפון שלי {case.claimant.phone} והנתבעת היא {case.defendant.full_name}"
    redacted = redact_text(text, case)
    safe = str(safe_case_context(case))
    assert case.claimant.full_name not in redacted
    assert case.claimant.phone not in redacted
    assert case.defendant.full_name not in redacted
    assert case.claimant.identifier not in safe
    assert case.defendant.address not in safe


def test_opening_privacy_scan_blocks_common_personal_identifiers():
    findings = free_text_privacy_findings(
        "שמי יוסי כהן, אני גר ברחוב הרצל 12 חיפה, הטלפון שלי 050-1234567"
    )
    assert "שם פרטי שהוצג כפרט זיהוי" in findings
    assert "כתובת מגורים אפשרית" in findings
    assert "מספר טלפון" in findings


def test_opening_privacy_scan_allows_business_case_description():
    findings = free_text_privacy_findings(
        "קניתי מזגן במחסני חשמל ברחוב ירושלים 1 תל אביב ב-3,500 שקל והוא התקלקל"
    )
    assert findings == []


def test_safe_context_does_not_claim_unknown_optional_pii_exists():
    case = CaseData()
    case.claimant.full_name = "א ב"
    case.claimant.address = "רחוב לדוגמה 1 תל אביב"
    case.defendant.full_name = "KSP"
    case.defendant.address = "רחוב ירושלים 3 תל אביב"

    safe = safe_case_context(case)

    assert safe["claimant"]["full_name"] == "[PLAINTIFF_NAME]"
    assert safe["claimant"]["email"] == ""
    assert safe["defendant"]["identifier"] == ""
    assert safe["defendant"]["phone"] == ""


def test_first_name_reference_is_redacted_after_local_collection():
    case = air_conditioner_sample()
    first_name = case.claimant.full_name.split()[0]
    assert first_name not in redact_text(f"כמו שאמרתי, אני {first_name}", case)


def test_allowlisted_authorities_only():
    case = air_conditioner_sample()
    records = retrieve_authorities(ClaimCategory.CONSUMER_PRODUCT)
    sections = DemoAgentClient().draft_sections(case, [record.model_dump(mode="json") for record in records])
    markdown = compose_claim_markdown(case, sections, records)
    allowed = {record.id for record in records}
    assert not validate_draft(markdown, allowed)
    assert validate_draft(markdown + "\n[AUTH:not-allowed]", allowed)


def test_contract_remedies_are_explained_as_concrete_legal_relief():
    paragraphs = controlled_cause_paragraphs(
        [{"id": "contracts-remedies-2"}, {"id": "contracts-remedies-9"}]
    )
    text = " ".join(paragraphs)
    assert "סוגי התרופות" not in text
    assert "אכיפה, ביטול או פיצויים" in text
    assert "השבת מה שקיבל כל צד" in text


def test_alternative_cancellation_retrieves_restitution_authority():
    case = air_conditioner_sample()
    case.primary_remedy = RemedyType.REPLACEMENT
    case.alternative_remedy = RemedyType.CANCELLATION
    authority_ids = {record.id for record in retrieve_authorities_for_case(case)}
    assert "contracts-remedies-9" in authority_ids


def test_relief_section_cannot_double_count_model_added_amounts():
    case = air_conditioner_sample()
    case.damage_items = [
        DamageItem(label="השבת מחיר המוצר", amount="3500", kind="direct"),
        DamageItem(label="טרחה ועוגמת נפש", amount="15000", kind="non_pecuniary"),
        DamageItem(label="נסיעות", amount="50", kind="consequential"),
        DamageItem(label="הפסד יום עבודה", amount="450", kind="consequential"),
    ]
    case.primary_remedy = RemedyType.CANCELLATION
    case.alternative_remedy = RemedyType.REPLACEMENT
    records = retrieve_authorities_for_case(case)
    sections = DemoAgentClient().draft_sections(
        case, [record.model_dump(mode="json") for record in records]
    )
    sections.relief_explanation = (
        "השבת 3,500 ₪ ובנוסף פיצויים בסך 19,000 ₪"  # deliberately unsafe model text
    )
    markdown = compose_claim_markdown(case, sections, records)

    assert case.claim_total == Decimal("19000")
    assert "השבת 3,500 ₪ ובנוסף פיצויים בסך 19,000 ₪" not in markdown
    assert "הסכום הכספי הכולל המבוקש הוא 19,000 ₪" in markdown
    assert "אינו מתווסף אליהם כסכום נוסף" in markdown


def test_unknown_defendant_identifier_is_warning_not_blocker():
    case = air_conditioner_sample()
    case.defendant.identifier = ""
    result = validate_case(case)
    assert result.ready
    assert any("הוצאה לפועל" in warning for warning in result.warnings)


def test_unsupported_category_retrieves_nothing():
    assert retrieve_authorities(ClaimCategory.UNSUPPORTED) == []


def test_service_dispute_is_classified_and_retrieved():
    category = classify_category("ספק שירות גבה תשלום אך לא ביצע את העבודה")
    assert category == ClaimCategory.SERVICE_CONTRACT
    assert retrieve_authorities(category)


def test_course_purchase_is_classified_as_service_even_with_bought_verb():
    assert classify_category("קניתי קורס ללימודי הייטק") == ClaimCategory.SERVICE_CONTRACT


def test_moving_an_apartment_is_classified_as_service_not_rental():
    assert (
        classify_category("הזמנתי שירות הובלת דירה והמובילים לא הגיעו")
        == ClaimCategory.SERVICE_CONTRACT
    )


def test_rental_dispute_is_classified_and_retrieved():
    category = classify_category("המשכיר לא החזיר את הפיקדון על הדירה")
    assert category == ClaimCategory.LANDLORD_TENANT
    assert retrieve_authorities(category)


def test_pdf_export_handles_mixed_direction_text():
    pdf = markdown_to_pdf_bytes("# בדיקת PDF\n\nכתב תביעה - 2026, סכום ₪3,200, טלפון 050-0000000")
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1_000


def test_opening_story_is_context_not_ready_made_chronology():
    case = seed_case_from_opening(SAMPLE_OPENING)
    assert case.event_summary == SAMPLE_OPENING
    assert case.claim_total == Decimal("3200")
    assert not case.prior_attempts
    assert not case.chronology
    assert case.category == ClaimCategory.CONSUMER_PRODUCT


def test_thin_keyboard_converted_opening_requires_incident_details():
    converted, changed = normalize_opening_keyboard_layout("ebh,h nzdi")
    assert changed
    assert converted == "קניתי מזגן"
    case = seed_case_from_opening(converted)
    assert not opening_story_is_complete(case.event_summary)
    assert "event_summary" not in inferred_completed_fields(case)
    assert next_interview_field(case, inferred_completed_fields(case)) == "event_summary"

    case = apply_chat_answer(case, "event_summary", "המזגן התקלקל אחרי שבוע ושילמתי 10,000 ₪")
    assert opening_story_is_complete(case.event_summary)
    assert "קניתי מזגן" in case.event_summary
    assert "התקלקל אחרי שבוע" in case.event_summary
    assert case.claim_total == Decimal("10000")


def test_opening_fragments_accumulate_and_common_failure_phrases_are_accepted():
    case = seed_case_from_opening("קניתי מזגן")
    problem = "אחרי שבוע מהקנייה הוא התחיל לעשות רעש, ואז אחרי יומיים הפסיק לעבוד"
    assert not opening_story_is_complete(merge_partial_answer(case.event_summary, problem))
    assert "כמה שילמתם" in opening_clarification(
        merge_partial_answer(case.event_summary, problem)
    )

    accumulated = merge_partial_answer(problem, "המזגן עלה 6000 שקל")
    case = apply_chat_answer(case, "event_summary", accumulated)
    assert opening_story_is_complete(case.event_summary)
    assert case.claim_total == Decimal("6000")


def test_complete_opening_with_stopped_working_advances_immediately():
    text = "קניתי מזגן ב3500 שקל ואחרי שבוע הפסיק לעבוד"
    case = seed_case_from_opening(text)
    assert opening_story_is_complete(text)
    assert "event_summary" in inferred_completed_fields(case)
    assert next_interview_field(case, inferred_completed_fields(case)) == "transaction_date"


def test_complete_laptop_opening_with_adjectival_damage_advances_immediately():
    text = (
        "קניתי לפטופ ב4000 שקל, פתחתי את הקופסה מהחנות וראיתי שהמסך של "
        "הלפטופ שבור, ניסיתי להחזיר את הלפטופ לחנות ולקבל את הכסף חזרה "
        "והם אמרו לי שאין החזרים ושאני שברתי אותו"
    )
    case = seed_case_from_opening(text)

    assert extract_purchased_subject(text) == "לפטופ"
    assert opening_story_missing_parts(text) == []
    assert opening_story_is_complete(text)
    assert case.claim_total == Decimal("4000")
    assert next_interview_field(case, inferred_completed_fields(case)) == "transaction_date"


def test_adjectival_damage_descriptions_are_product_agnostic():
    variants = (
        "קניתי אגרטל ב200 שקל והוא הגיע סדוק",
        "קניתי ארון ב1800 שקל והוא הגיע פגום",
        "קניתי שמלה ב350 שקל והיא הגיעה קרועה",
        "קניתי אופניים ב2200 שקל והשלדה הייתה מעוכה",
    )

    for text in variants:
        assert opening_story_is_complete(text), text


def test_follow_up_broken_condition_completes_existing_opening():
    case = seed_case_from_opening("קניתי לפטופ ב4000 שקל")

    updated = apply_chat_answer(
        case,
        "event_summary",
        "מה שהשתבש זה שהלפטופ הגיע שבור ולא במצב חדש",
    )

    assert opening_story_is_complete(updated.event_summary)
    assert "הגיע שבור" in updated.event_summary


def test_complete_course_cancellation_opening_advances_without_inventing_details():
    text = (
        "שילמתי 20000 שקל על קורס ללימודים בהייטק ואמרו לי שיש לי שבועיים "
        "להתחרט ולקבל את הכסף בחזרה, אחרי יומיים התחרטתי ואמרו לי שאני לא "
        "יכול לקבל את הכסף כי כבר נחשפתי לתכנים באתר"
    )
    case = seed_case_from_opening(text)

    assert extract_purchased_subject(text) == "קורס ללימודים בהייטק"
    assert opening_story_is_complete(text)
    assert case.event_summary == text
    assert case.product_or_service_details == "קורס ללימודים בהייטק"
    assert case.claim_total == Decimal("20000")
    assert case.category == ClaimCategory.SERVICE_CONTRACT
    assert "event_summary" in inferred_completed_fields(case)
    assert next_interview_field(case, inferred_completed_fields(case)) == "transaction_date"


def test_arbitrary_service_phrase_is_extracted_verbatim():
    text = "שילמתי 4500 שקל על סדנת צילום אבל ביקשתי לבטל וסירבו להחזיר לי כסף"
    case = seed_case_from_opening(text)

    assert extract_purchased_subject(text) == "סדנת צילום"
    assert opening_story_is_complete(text)
    assert case.product_or_service_details == "סדנת צילום"
    assert case.category == ClaimCategory.SERVICE_CONTRACT


def test_restaurant_double_charge_opening_with_thousands_advances():
    text = (
        "בעל מסעדה עקץ אותי בכסף והביא לי שירות מזלזל וניצל את התמימות שלי "
        "חייב אותי פעמים בסכום כולל של 20 אלף"
    )
    case = seed_case_from_opening(text)

    assert opening_story_is_complete(text)
    assert case.claim_total == Decimal("20000")
    assert case.category == ClaimCategory.SERVICE_CONTRACT
    assert next_interview_field(case, inferred_completed_fields(case)) == "transaction_date"


def test_hebrew_word_thousands_are_recognized_in_opening():
    text = "קיבלתי שירות וחויבתי פעמיים בלי סיבה ושילמתי עשר אלף שקל"
    case = seed_case_from_opening(text)

    assert opening_story_is_complete(text)
    assert case.claim_total == Decimal("10000")


def test_bare_requested_amount_replaces_total_duplicate_charge():
    opening = (
        "בעל מסעדה חייב אותי פעמיים בסכום כולל של 20 אלף "
        "וקיבלתי שירות מזלזל"
    )
    case = seed_case_from_opening(opening)
    assert case.claim_total == Decimal("20000")

    updated = apply_chat_answer(case, "damage_items", "10000")

    assert updated.claim_total == Decimal("10000")
    assert len(updated.damage_items) == 1
    assert "החיוב הכפול העודף" in updated.damage_items[0].label


def test_live_draft_preview_shows_filled_pending_and_empty_fields():
    case = seed_case_from_opening(
        "קניתי מזגן ב6500 שקל והוא התקלקל אחרי שבוע"
    )
    preview = build_live_draft_preview(
        case,
        inferred_completed_fields(case),
        {"prior_attempts": "התקשרתי לחנות"},
    )

    assert "טיוטת ביניים" in preview
    assert "מזגן" in preview
    assert "6,500 ₪" in preview
    assert "התקשרתי לחנות _(נקלט חלקית; טרם אושר)_" in preview
    assert "⬜ [טרם מולא]" in preview


def test_stopped_operating_phrase_is_also_a_problem_description():
    assert opening_story_is_complete(
        "קניתי מזגן ב6000 שקל ואחרי שבוע התחיל להרעיש ואחרי יומיים הפסיק לפעול"
    )


def test_football_opening_with_puncture_or_explosion_is_complete():
    variants = (
        "קניתי כדורגל ב10 שקל ואחרי יום התפנצ'ר",
        "קניתי כדור כדורגל ב10 שקלים ואחרי יום הוא התפוצץ",
    )
    for text in variants:
        case = seed_case_from_opening(text)
        assert opening_story_is_complete(text)
        assert "event_summary" in inferred_completed_fields(case)
        assert case.claim_total == Decimal("10")


def test_what_did_you_write_is_a_recall_request():
    assert is_recall_request("מה כתבת עד כה?")
    assert is_recall_request("מה נקלט?")
    assert is_recall_request("מה כן יש לך?")
    assert is_recall_request("מה קלטת עד עכשיו")


def test_current_question_help_and_reset_commands_are_distinct_from_answers():
    assert is_current_question_help_request("מה חסר לך?")
    assert is_current_question_help_request("למה זה לא מתקבל?")
    assert is_current_question_reset_request("אפס את השאלה")
    assert is_current_question_reset_request("התחל את השאלה מחדש")
    assert not is_current_question_help_request("התקשרתי לחנות")
    assert not is_current_question_reset_request("אין אחריות")


def test_repeated_failure_guidance_identifies_prior_attempt_fragment():
    case = CaseData(product_or_service_details="מזגן")
    guidance = build_repeated_failure_guidance(
        case,
        "prior_attempts",
        "התקשרתי פעמיים לחנות",
        failure_count=4,
        last_error="חסרה התגובה שקיבלתם.",
    )
    assert "אמצעי הפנייה" in guidance
    assert "מה ענה הנתבע" in guidance
    assert "אפס את השאלה" in guidance
    assert "לא אכניס לתיק" not in guidance


def test_sixth_failure_explains_that_invalid_data_will_not_enter_state():
    guidance = build_repeated_failure_guidance(
        CaseData(),
        "transaction_date",
        failure_count=6,
        last_error="התאריך נמצא בעתיד.",
    )
    assert "לא אכניס לתיק ערך שלא עבר אימות" in guidance
    assert "1.1.26" in guidance


def test_streamlit_recovers_from_repeated_invalid_answer_without_accepting_it(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest
    from small_claims_agent.llm import GeminiAgentClient

    def fake_extract(self, user_text, case):
        return StateExtraction(inferred_category=classify_category(user_text, case.category))

    def fake_repair(self, field, question, failed_answers, case):
        return AnswerRepair(
            action="clarify",
            clarification_question="נדרש תאריך מלא מהמסמך.",
            confidence=0,
        )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(GeminiAgentClient, "extract", fake_extract)
    monkeypatch.setattr(GeminiAgentClient, "repair_answer", fake_repair)

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    app.chat_input[0].set_value("קניתי מזגן ב-3200 שקל והוא התקלקל").run()
    assert app.session_state["awaiting_field"] == "transaction_date"

    for _ in range(4):
        app.chat_input[0].set_value("בתחילת ינואר").run()

    assert not app.exception
    assert app.session_state["case"].transaction_date == ""
    assert app.session_state["failure_counts"]["transaction_date"] == 4
    assert "נתקענו בשאלה" in app.session_state["messages"][-1]["content"]
    assert "אפס את השאלה" in app.session_state["messages"][-1]["content"]

    app.chat_input[0].set_value("מה חסר?").run()
    assert app.session_state["failure_counts"]["transaction_date"] == 4
    assert "תאריך עסקה אמיתי" in app.session_state["messages"][-1]["content"]

    app.chat_input[0].set_value("אפס את השאלה").run()
    assert "transaction_date" not in app.session_state["failure_counts"]
    assert "transaction_date" not in app.session_state["failed_answers"]
    assert app.session_state["awaiting_field"] == "transaction_date"
    assert "איפסתי" in app.session_state["messages"][-1]["content"]

    app.chat_input[0].set_value("1.1.26").run()
    assert app.session_state["case"].transaction_date == "01.01.2026"
    assert app.session_state["awaiting_field"] == "incident_date"


def test_streamlit_accumulates_opening_fragments_without_getting_stuck(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest
    from small_claims_agent.llm import GeminiAgentClient

    def fake_extract(self, user_text, case):
        return StateExtraction(inferred_category=classify_category(user_text, case.category))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(GeminiAgentClient, "extract", fake_extract)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()

    app.chat_input[0].set_value("ebh,h nzdi").run()
    assert app.session_state["awaiting_field"] == "event_summary"
    app.chat_input[0].set_value(
        "אחרי שבוע מהקנייה הוא התחיל לעשות רעש, ואז אחרי יומיים הפסיק לעבוד"
    ).run()
    assert app.session_state["awaiting_field"] == "event_summary"
    assert "כמה שילמתם" in app.session_state["messages"][-1]["content"]

    app.chat_input[0].set_value("המזגן עלה 6000 שקל").run()
    assert not app.exception
    assert app.session_state["case"].claim_total == Decimal("6000")
    assert app.session_state["awaiting_field"] == "transaction_date"
    events = [entry["event"] for entry in app.session_state["diagnostics"]]
    assert "opening_parsed" in events
    assert "answer_accepted" in events


def test_streamlit_accepts_complete_course_cancellation_opening(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest
    from small_claims_agent.llm import GeminiAgentClient

    def fake_extract(self, user_text, case):
        return StateExtraction(inferred_category=classify_category(user_text, case.category))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(GeminiAgentClient, "extract", fake_extract)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    opening = (
        "שילמתי 20000 שקל על קורס ללימודים בהייטק ואמרו לי שיש לי שבועיים "
        "להתחרט ולקבל את הכסף בחזרה, אחרי יומיים התחרטתי ואמרו לי שאני לא "
        "יכול לקבל את הכסף כי כבר נחשפתי לתכנים באתר"
    )

    app.chat_input[0].set_value(opening).run()

    assert not app.exception
    assert app.session_state["awaiting_field"] == "transaction_date"
    assert app.session_state["case"].product_or_service_details == "קורס ללימודים בהייטק"
    assert app.session_state["case"].category == ClaimCategory.SERVICE_CONTRACT
    assert app.session_state["case"].claim_total == Decimal("20000")

    recall_button = next(button for button in app.button if button.label == "📄 הצגת טיוטת הביניים")
    recall_button.click().run()
    preview = app.session_state["messages"][-1]["content"]
    assert "טיוטת ביניים" in preview
    assert "קורס ללימודים בהייטק" in preview
    assert "20,000 ₪" in preview
    assert "⬜ [טרם מולא]" in preview


def test_chat_interview_completes_without_a_form():
    case = seed_case_from_opening(SAMPLE_OPENING)
    completed = {"event_summary"}
    answers = {
        "transaction_date": "2026-06-01",
        "incident_date": "2026-06-15",
        "incident_place": "חנות הנתבעת בתל אביב-יפו",
        "product_or_service_details": "מזגן ביתי מדגם Demo-3200",
        "defendant.full_name": "אלפא מוצרי חשמל בע״מ",
        "defendant_role": "מוכר",
        "defendant.identifier": "לא ידוע",
        "defendant.address": "רחוב המסחר 8, תל אביב-יפו",
        "defendant.phone": "לא ידוע",
        "prior_attempts": "פניתי שלוש פעמים בטלפון ביוני 2026 והחנות סירבה לתקן או להשיב את הכסף",
        "damage_items": "השבת מחיר המוצר - 3200 ₪",
        "primary_remedy": "ביטול העסקה והחזר כספי",
        "alternative_remedy": "החלפת המזגן",
        "warranty_details": "אחריות יצרן לשנה",
        "evidence_confirmation": "קבלה\nתכתובת עם שירות הלקוחות",
        "claimant.full_name": "דנה ישראלי",
        "claimant.identifier": "000000000",
        "claimant.address": "רחוב הדוגמה 12, תל אביב-יפו",
        "claimant.phone": "050-0000000",
        "claimant.email": "אין",
    }
    asked = []
    while field := next_interview_field(case, completed):
        asked.append(field)
        case = apply_chat_answer(case, field, answers[field])
        completed.add(field)
    case.facts_confirmed = True
    assert len(asked) == len(set(asked))
    assert validate_case(case).ready
    assert case.claim_total == Decimal("3200")
    assert case.court_city == "תל אביב-יפו"


def test_multiple_damage_lines_are_parsed_separately():
    items = parse_damage_items("מחיר המוצר - 3,200 ₪\nבדיקת טכנאי - 350 ₪")
    assert [item.amount for item in items] == [Decimal("3200"), Decimal("350")]


def test_optional_interview_questions_have_chat_prompts():
    assert "מספר החברה" in question_for("defendant.identifier")
    assert "דוא״ל" in question_for("claimant.email")


def test_relative_date_and_vague_place_require_clarification():
    case = seed_case_from_opening(SAMPLE_OPENING)
    try:
        apply_chat_answer(case, "transaction_date", "לפני חצי שנה")
        assert False, "relative date should be rejected"
    except InterviewAnswerError:
        pass
    try:
        apply_chat_answer(case, "incident_place", "בחנות")
        assert False, "vague place should be rejected"
    except InterviewAnswerError:
        pass


def test_address_typo_is_not_silently_propagated():
    case = seed_case_from_opening(SAMPLE_OPENING)
    try:
        apply_chat_answer(case, "defendant.address", "ז׳בוטינסקי 1 רמת גם")
        assert False, "known city typo should be rejected"
    except InterviewAnswerError as exc:
        assert "רמת גן" in str(exc)


def test_ramat_gan_is_resolved_to_tel_aviv_court():
    case = seed_case_from_opening(SAMPLE_OPENING)
    case = apply_chat_answer(case, "incident_place", "סניף ברחוב ביאליק 1, רמת גן")
    case = apply_chat_answer(case, "defendant.address", "ז׳בוטינסקי 1, רמת גן")
    assert case.court_city == "תל אביב-יפו"
    assert "רמת גן" in case.jurisdiction_basis


def test_evidence_is_normalized_for_annex_list():
    case = apply_chat_answer(seed_case_from_opening(SAMPLE_OPENING), "evidence_confirmation", "יש לי קבלה")
    assert case.evidence[0].name == "העתק קבלה/חשבונית בגין העסקה"


def test_legal_retrieval_is_fact_and_remedy_specific():
    case = air_conditioner_sample()
    ids = {record.id for record in retrieve_authorities_for_case(case)}
    assert {"sale-law-11", "contracts-remedies-2", "contracts-remedies-9", "consumer-warranty-regulations"} <= ids


def test_gemini_update_shape_variations_do_not_crash_merge():
    case = seed_case_from_opening(SAMPLE_OPENING)
    merged = _merge_updates(
        case,
        [
            {
                "path": "prior_attempts",
                "value": [{"description": "שלוש פניות לחנות", "resolved": False}],
            },
            {
                "path": "damage_items",
                "value": [{"description": "עלות המזגן", "amount": 3200}],
            },
        ],
    )
    assert merged.prior_attempts == ["שלוש פניות לחנות"]
    assert merged.damage_items[0].label == "עלות המזגן"
    assert merged.damage_items[0].amount == Decimal("3200")


def test_invalid_gemini_update_is_ignored_without_losing_valid_updates():
    case = seed_case_from_opening(SAMPLE_OPENING)
    merged = _merge_updates(
        case,
        [
            {"path": "transaction_date", "value": {"unexpected": "object"}},
            {"path": "product_or_service_details", "value": "מזגן ביתי"},
            {"path": "category", "value": "unsupported"},
            {"path": "facts_confirmed", "value": True},
        ],
    )
    assert merged.transaction_date == ""
    assert merged.product_or_service_details == "מזגן ביתי"
    assert merged.category == ClaimCategory.CONSUMER_PRODUCT
    assert not merged.facts_confirmed


def test_natural_opening_recognizes_product_category_and_shekels_word():
    case = seed_case_from_opening("קניתי מזגן ב3200 שקל ואחרי שבוע הוא התקלקל")
    assert case.category == ClaimCategory.CONSUMER_PRODUCT
    assert case.product_or_service_details == "מזגן"
    assert case.claim_total == Decimal("3200")


def test_two_digit_year_is_accepted_and_normalized():
    case = apply_chat_answer(CaseData(), "incident_date", "18.08.26")
    assert case.incident_date == "18.08.2026"


def test_natural_phone_attempt_is_accepted_without_forcing_a_date():
    answer = "התקשרתי פעמיים והם אמרו שזה אני אשם"
    case = apply_chat_answer(CaseData(), "prior_attempts", answer)
    assert case.prior_attempts == [answer]


def test_prior_location_can_supply_referenced_business_name_and_address():
    case = apply_chat_answer(
        CaseData(), "incident_place", "תל אביב רחוב הארבעה 1 מחסני הזבלה"
    )
    case = apply_chat_answer(case, "defendant.full_name", "כתבתי")
    case = apply_chat_answer(case, "defendant.address", "כתבתי")
    assert case.defendant.full_name == "מחסני הזבלה"
    assert case.defendant.address == "תל אביב רחוב הארבעה 1"
    assert "כתבתי" not in case.model_dump_json()


def test_compound_store_place_splits_brand_address_and_preserves_all_roles():
    case = apply_chat_answer(
        CaseData(), "incident_place", "חנות KSP רחוב ירושלים 3 תל אביב"
    )
    case = apply_chat_answer(case, "defendant.full_name", "כתבתי")
    case = apply_chat_answer(case, "defendant_role", "יבואן / נותן שירות / מוכר")
    case = apply_chat_answer(case, "defendant.address", "אותה כתובת")

    assert case.defendant.full_name == "KSP"
    assert case.defendant.address == "רחוב ירושלים 3 תל אביב"
    assert case.defendant_role == "מוכר, יבואן ונותן שירות"


def test_same_address_reference_cannot_silently_become_empty():
    case = CaseData(incident_place="חנות KSP", defendant={"full_name": "חנות KSP"})

    try:
        apply_chat_answer(case, "defendant.address", "אותה כתובת")
        assert False, "an empty derived service address must be rejected"
    except InterviewAnswerError as exc:
        assert "לא הצלחתי להפיק כתובת" in str(exc)


def test_reference_suffix_is_removed_from_real_value():
    case = apply_chat_answer(CaseData(), "defendant.full_name", "מחסני הזבלה אמרתי לך")
    assert case.defendant.full_name == "מחסני הזבלה"


def test_damage_parser_accepts_bare_amounts_and_multiple_natural_formats():
    multiline = parse_damage_items("מחיר מוצר - 3200\nעגמת נפש 800")
    compact = parse_damage_items("3200 מוצר 800 עגמת נפש")
    assert [(item.label, item.amount) for item in multiline] == [
        ("השבת מחיר המוצר", Decimal("3200")),
        ("טרחה ועוגמת נפש", Decimal("800")),
    ]
    assert [(item.label, item.amount) for item in compact] == [
        ("השבת מחיר המוצר", Decimal("3200")),
        ("טרחה ועוגמת נפש", Decimal("800")),
    ]


def test_total_with_distress_is_split_using_known_purchase_price():
    items = parse_damage_items("4000 כי זה המחיר ועוד עגמת נפש", Decimal("3200"))
    assert [item.amount for item in items] == [Decimal("3200"), Decimal("800")]
    assert items[1].kind == "non_pecuniary"


def test_no_additional_damage_keeps_amount_from_opening():
    case = seed_case_from_opening("קניתי מזגן ב3200 שקל והוא התקלקל")
    case = apply_chat_answer(case, "damage_items", "רק מחיר המוצר")
    assert case.claim_total == Decimal("3200")


def test_damage_question_reuses_known_amount():
    case = seed_case_from_opening("קניתי מזגן ב3200 שקל והוא התקלקל")
    prompt = question_for("damage_items", case)
    assert "3,200 ₪" in prompt
    assert "מהתיאור הראשון" in prompt


def test_bare_confirmed_product_amount_gets_a_specific_damage_label():
    case = seed_case_from_opening("קניתי לפטופ ב4000 שקל והוא הגיע שבור")
    updated = apply_chat_answer(case, "damage_items", "4000")

    assert updated.damage_items[0].label == "השבת מחיר המוצר"
    assert updated.damage_items[0].amount == Decimal("4000")


def test_short_new_product_warranty_answer_is_formalized():
    case = apply_chat_answer(CaseData(), "warranty_details", "כן מוצר חדש")

    assert case.warranty_details == (
        "המוצר נרכש כחדש וקיימת אחריות, אך לא נמסרו תקופתה או פרטי מסמך האחריות"
    )


def test_streamlit_transcript_handles_natural_followups(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest
    from small_claims_agent.llm import GeminiAgentClient

    review_calls: list[tuple[str, str]] = []

    def fake_extract(self, user_text, case):
        return StateExtraction(
            inferred_category=classify_category(user_text, case.category),
            updates=[
                FieldUpdate(
                    path="event_summary",
                    value="Gemini invented a product failure that the user never described",
                    evidence_text="unsupported",
                )
            ],
        )

    def fake_review(self, case):
        review_calls.append((case.category.value, case.claimant.full_name))
        # Simulate the false-positive date warning seen in the real transcript.
        # The deterministic gate must discard it because 18 August is after 11 August.
        return InterviewConsistencyReview(
            verdict="clarify",
            issues=[
                ConsistencyIssue(
                    field="transaction_date",
                    message="possible date issue",
                    clarification_question="Are the dates correct?",
                )
            ],
        )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(GeminiAgentClient, "extract", fake_extract)
    monkeypatch.setattr(GeminiAgentClient, "review_interview", fake_review)
    messages = [
        "קניתי מזגן ב3200 שקל ואחרי שבוע הוא התקלקל, פניתי לחברה והם מתנערים",
        "11.08.2026",
        "18.08.26",
        "תל אבי רחוב הארבעה 1 מחסני הזבלה",
        "תל אביב רחוב הארבעה 1 מחסני הזבלה",
        "כתבתי",
        "מוכר",
        "לא ידוע",
        "כתובת העסק",
        "לא ידו",
        "באתי לכל משחק ליגה העונה, ובכולם הם הפסידו",
        "צעקתי במשחק שהם משחקים חרא",
        "התקשרתי אליהם פעמיים ואמרתי להם שהמזגן התקלקל",
        "הם אמרו שלא יתקנו אותו",
        "3200 מוצר 800 עגמת נפש",
        "החלפה",
        "החזר וביטול עסקה",
        "קיימת אחריות לשנה, יש לי קבלה ותעודת אחריות",
        "אין ראיות נוספות",
        "א ב יהושוע",
        "123456789",
        "המלך שלמה 89 בת ים",
        "0545123456",
        "אין",
    ]
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    for message in messages:
        app.chat_input[0].set_value(message).run()

    case = app.session_state["case"]
    assert not app.exception
    assert case.category == ClaimCategory.CONSUMER_PRODUCT
    assert case.event_summary == messages[0]
    assert case.product_or_service_details == "מזגן"
    assert case.incident_date == "18.08.2026"
    assert case.defendant.full_name == "מחסני הזבלה"
    assert case.defendant.address == "תל אביב רחוב הארבעה 1"
    assert case.prior_attempts == [
        "התקשרתי אליהם פעמיים ואמרתי להם שהמזגן התקלקל הם אמרו שלא יתקנו אותו"
    ]
    assert case.claim_total == Decimal("4000")
    assert app.session_state["awaiting_confirmation"]

    app.chat_input[0].set_value("תקן נתבע/ת: מחסני החשמל").run()
    assert app.session_state["case"].defendant.full_name == "מחסני החשמל"
    assert app.session_state["awaiting_confirmation"]
    app.chat_input[0].set_value("מאשר").run()
    assert app.session_state["ready"]
    assert review_calls == [("consumer_product", "א ב יהושוע")]


def test_streamlit_laptop_transcript_keeps_reused_defendant_address(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    messages = [
        (
            "קניתי לפטופ ב4000 שקל, פתחתי את הקופסה מהחנות וראיתי שהמסך של הלפטופ שבור, "
            "ניסיתי להחזיר את הלפטופ לחנות ולקבל את הכסף חזרה והם אמרו לי שאין החזרים "
            "ושאין סיכוי שהם מכרו לי ככה את הלפטופ ושאני שברתי אותו"
        ),
        "1.1.26",
        "1.1.26",
        "חנות KSP רחוב ירושלים 3 תל אביב",
        "כתבתי",
        "יבואן / נותן שירות / מוכר",
        "לא",
        "אותה כתובת",
        "לא",
        "השתמש במה שכתבתי",
        "4000",
        "כסף",
        "החלפה",
        "כן יש אחריות לשנתיים",
        "קבלה",
        "א ב",
        "123456789",
        "תל אביב 17 אילת",
        "0545232323",
        "אין",
    ]

    for message in messages:
        app.chat_input[0].set_value(message).run()

    case = app.session_state["case"]
    address_questions = [
        message["content"]
        for message in app.session_state["messages"]
        if message["role"] == "assistant"
        and message["content"].startswith("מה כתובת המסירה המלאה של הנתבע/ת?")
    ]

    assert not app.exception
    assert app.session_state["awaiting_confirmation"]
    assert case.defendant.full_name == "KSP"
    assert case.defendant.address == "רחוב ירושלים 3 תל אביב"
    assert case.defendant_role == "מוכר, יבואן ונותן שירות"
    assert len(address_questions) == 1


def test_compact_and_partially_separated_dates_are_normalized():
    case = apply_chat_answer(CaseData(), "transaction_date", "01012026")
    case = apply_chat_answer(case, "incident_date", "08.0126")
    assert case.transaction_date == "01.01.2026"
    assert case.incident_date == "08.01.2026"


def test_single_digit_and_slash_dates_are_normalized():
    case = apply_chat_answer(CaseData(), "transaction_date", "1.1.26")
    case = apply_chat_answer(case, "incident_date", "10/01/26")
    assert case.transaction_date == "01.01.2026"
    assert case.incident_date == "10.01.2026"
    assert "1.1.26" in question_for("transaction_date")
    assert "01/01/26" in question_for("transaction_date")


def test_other_alternative_remedy_is_preserved_verbatim_in_summary_and_draft():
    case = air_conditioner_sample()
    case = apply_chat_answer(case, "alternative_remedy", 'שיוציאו פוסט "יוסי המלך"')
    assert case.alternative_remedy == RemedyType.OTHER
    assert case.desired_non_monetary_relief == 'שיוציאו פוסט "יוסי המלך"'
    assert 'שיוציאו פוסט "יוסי המלך"' in build_confirmation_summary(case)

    records = retrieve_authorities_for_case(case)
    sections = DemoAgentClient().draft_sections(
        case, [record.model_dump(mode="json") for record in records]
    )
    markdown = compose_claim_markdown(case, sections, records)
    assert 'לחלופין בלבד, מבוקש שיוציאו פוסט "יוסי המלך"' in markdown


def test_statutory_warranty_source_is_conditional_not_removed_when_user_reports_none():
    case = air_conditioner_sample()
    case.warranty_details = "לא קיימת"
    case.product_or_service_details = "מזגן"
    records = retrieve_authorities_for_case(case)
    ids = {record.id for record in records}
    assert "consumer-warranty-regulations" in ids

    causes = controlled_cause_paragraphs(
        [record.model_dump(mode="json") for record in records], case
    )
    text = " ".join(causes)
    assert "ככל שמדובר במוצר חדש" in text
    assert "גם אם לא נמסרה תעודת אחריות" in text
    assert "פרטי האחריות שנמסרו" not in text

    review = DraftReview(
        verdict="revise",
        unsupported_assertions=["תקנות הגנת הצרכן לא חלות כי אין אחריות"],
        revision_instructions=["הסר את האזכור של תקנות הגנת הצרכן"],
    )
    cleaned = _sanitize_draft_review(review, case, ids)
    assert cleaned.verdict == "pass"
    assert not cleaned.revision_instructions


def test_quality_review_cannot_turn_privacy_placeholders_into_factual_findings():
    case = air_conditioner_sample()
    review = DraftReview(
        verdict="revise",
        factual_consistency_issues=[
            "בשם הנתבע נרשם [INCIDENT_PLACE] במקום [DEFENDANT_NAME].",
            "הטלפון לא נמסר אף שקיים [DEFENDANT_PHONE].",
        ],
        clarity_issues=["נפל שיבוש: נמצ[PLAINTIFF_NAME]תל אביב."],
        revision_instructions=["החלף את [PLAINTIFF_EMAIL] בדוא״ל שהוזן."],
    )

    cleaned = _sanitize_draft_review(review, case, set())

    assert cleaned.verdict == "pass"
    assert not cleaned.factual_consistency_issues
    assert not cleaned.clarity_issues
    assert not cleaned.revision_instructions


def test_local_draft_validation_catches_changed_party_details_before_llm_review():
    case = air_conditioner_sample()
    records = retrieve_authorities_for_case(case)
    sections = DemoAgentClient().draft_sections(
        case, [record.model_dump(mode="json") for record in records]
    )
    markdown = compose_claim_markdown(case, sections, records)
    corrupted = markdown.replace(case.defendant.full_name, "שם עסק שגוי")

    issues = validate_draft(corrupted, {record.id for record in records}, case)

    assert any("שם הנתבע/ת" in issue for issue in issues)


def test_placeholder_store_is_not_treated_as_a_real_location():
    case = CaseData(incident_place="חנות X")
    assert "incident_place" not in inferred_completed_fields(case)
    try:
        apply_chat_answer(CaseData(), "incident_place", "חנות X")
        assert False, "placeholder location should be rejected"
    except InterviewAnswerError:
        pass


def test_described_online_transaction_is_a_valid_incident_place():
    place = 'עסקה אינטרנטית באתר של החברה "COURSES"'
    case = apply_chat_answer(CaseData(), "incident_place", place)

    assert case.incident_place == place
    assert "incident_place" in inferred_completed_fields(case)


def test_latin_website_name_and_domain_are_valid_online_places():
    for place in ('אתר "COURSES"', "courses.co.il", "באתר COURSES"):
        case = apply_chat_answer(CaseData(), "incident_place", place)
        assert case.incident_place == place
        assert "incident_place" in inferred_completed_fields(case)


def test_telephone_transaction_is_a_valid_incident_place():
    for place in ("עסקה טלפונית", "ביצעתי את העסקה בטלפון", "רכשתי דרך המוקד"):
        case = apply_chat_answer(CaseData(), "incident_place", place)
        assert case.incident_place == place
        assert "incident_place" in inferred_completed_fields(case)


def test_bare_online_transaction_still_requests_channel_details():
    try:
        apply_chat_answer(CaseData(), "incident_place", "באינטרנט")
        assert False, "a bare online location should still be clarified"
    except InterviewAnswerError as exc:
        assert "שם החברה" in str(exc)


def test_online_transaction_cannot_be_reused_as_defendant_postal_address():
    case = CaseData(incident_place='עסקה אינטרנטית באתר של החברה "COURSES"')
    try:
        apply_chat_answer(case, "defendant.address", "אותה כתובת")
        assert False, "an online channel is not a service address"
    except InterviewAnswerError as exc:
        assert "כתובת הפיזית" in str(exc)


def test_telephone_transaction_cannot_be_reused_as_defendant_postal_address():
    case = CaseData(incident_place="עסקה טלפונית")
    try:
        apply_chat_answer(case, "defendant.address", "אותה כתובת")
        assert False, "a telephone channel is not a service address"
    except InterviewAnswerError as exc:
        assert "כתובת הפיזית" in str(exc)


def test_unknown_defendant_address_returns_actionable_lookup_guidance():
    case = CaseData(incident_place="אינטרנט, באתר COURSES")
    try:
        apply_chat_answer(case, "defendant.address", "לא יודע")
        assert False, "a website cannot replace the defendant service address"
    except InterviewAnswerError as exc:
        message = str(exc)
        assert "כתובת פיזית" in message
        assert "צור קשר" in message
        assert "רשם החברות" in message


def test_english_keyboard_name_is_rejected():
    try:
        apply_chat_answer(CaseData(), "defendant.full_name", "njxbh huxh")
        assert False, "Latin keyboard gibberish should be rejected"
    except InterviewAnswerError as exc:
        assert "מקלדת האנגלית" in str(exc)


def test_latin_defendant_business_brand_is_accepted():
    case = apply_chat_answer(CaseData(), "defendant.full_name", "COURSES")
    assert case.defendant.full_name == "COURSES"


def test_short_israeli_mobile_number_is_rejected():
    try:
        apply_chat_answer(CaseData(), "claimant.phone", "054512346")
        assert False, "Israeli mobile number must have ten digits"
    except InterviewAnswerError as exc:
        assert "10 ספרות" in str(exc)


def test_warranty_evidence_is_reused_and_comma_list_is_deduplicated():
    case = apply_chat_answer(
        CaseData(),
        "warranty_details",
        "קיימת אחריות לשנה, יש לי קבלה ותעודת אחריות",
    )
    assert [item.name for item in case.evidence] == [
        "העתק קבלה/חשבונית בגין העסקה",
        "העתק תעודת אחריות",
    ]
    case = apply_chat_answer(case, "evidence_confirmation", "קבלה, תעודת אחריות")
    assert len(case.evidence) == 2


def test_large_nonpecuniary_amount_requires_explicit_confirmation():
    try:
        parse_damage_items("מחיר מוצר 3200 עגמת נפש 4000", Decimal("3200"))
        assert False, "ambiguous unusually large distress item should require confirmation"
    except InterviewAnswerError as exc:
        assert "7,200" in str(exc)
    confirmed = parse_damage_items(
        "מחיר מוצר 3200 עגמת נפש 4000 בוודאות", Decimal("3200")
    )
    assert sum(item.amount for item in confirmed) == Decimal("7200")


def test_large_nonpecuniary_amount_requires_confirmation_without_opening_amount():
    try:
        parse_damage_items("מחיר מוצר 3500 עוגמת נפש 10000")
        assert False, "large distress item should be confirmed even without an opening amount"
    except InterviewAnswerError as exc:
        assert "13,500" in str(exc)
    confirmed = parse_damage_items("מחיר מוצר 3500 עוגמת נפש 10000 בוודאות")
    assert sum(item.amount for item in confirmed) == Decimal("13500")


def test_standalone_damage_confirmation_reuses_itemized_answer():
    original = "3500 מזגן 15000 עגמת נפש 50 נסיעות 450 הפסד יום עבודה"
    merged = merge_damage_confirmation([original], "בוודאות")
    assert merged == f"{original} בוודאות"
    items = parse_damage_items(merged)
    assert [item.amount for item in items] == [
        Decimal("3500"),
        Decimal("15000"),
        Decimal("50"),
        Decimal("450"),
    ]
    assert items[0].label == "השבת מחיר המוצר"
    assert items[2].kind == "consequential"
    assert items[3].kind == "consequential"
    assert items[3].label == "הפסד יום עבודה"
    assert "בוודאות" not in " ".join(item.label for item in items)
    assert merge_damage_confirmation([original], "כן") == f"{original} בוודאות"
    assert sum(item.amount for item in items) == Decimal("19000")


def test_english_keyboard_opening_is_converted_and_bare_price_is_captured():
    converted, changed = normalize_opening_keyboard_layout("ebh,h nzdi c3500")
    assert changed
    assert converted == "קניתי מזגן ב3500"
    case = seed_case_from_opening(converted)
    assert case.product_or_service_details == "מזגן"
    assert case.claim_total == Decimal("3500")


def test_space_separated_receipt_and_warranty_become_two_normalized_evidence_items():
    case = apply_chat_answer(CaseData(), "evidence_confirmation", "אחריות קבלה")
    assert [item.name for item in case.evidence] == [
        "העתק קבלה/חשבונית בגין העסקה",
        "העתק תעודת אחריות",
    ]


def test_receipt_does_not_discard_other_evidence_lines():
    case = apply_chat_answer(
        CaseData(),
        "evidence_confirmation",
        "קבלה\nפירוט כרטיס אשראי\nצילום המוצר\nתכתובת דוא״ל",
    )
    assert [item.name for item in case.evidence] == [
        "העתק קבלה/חשבונית בגין העסקה",
        "פירוט כרטיס אשראי",
        "צילום המוצר",
        "העתק תכתובת דוא״ל",
    ]


def test_invalid_model_draft_is_replaced_by_valid_local_draft():
    class InvalidDraftClient(DemoAgentClient):
        model_name = "invalid-test-model"

        def draft_sections(self, case, authorities):
            sections = super().draft_sections(case, authorities)
            sections.factual_background = list(case.prior_attempts)
            return sections

    case = air_conditioner_sample()
    graph = build_graph(InvalidDraftClient())
    state = initial_state(case)
    state["generate_requested"] = True
    result = graph.invoke(state, {"configurable": {"thread_id": "draft-fallback-test"}})

    assert result["draft_markdown"]
    assert not result["validation_errors"]
    assert any("טיוטת Gemini" in warning for warning in result["warnings"])


def test_natural_defendant_correction_commands_are_understood():
    assert correction_command("תקן נתבע") == ("defendant.full_name", "")
    assert correction_command("תקן נתבע מחסני הזבלה") == (
        "defendant.full_name",
        "מחסני הזבלה",
    )
    assert correction_command("תקן נתבע/ת: מחסני הזבלה") == (
        "defendant.full_name",
        "מחסני הזבלה",
    )


def test_defendant_address_correction_commands_target_the_address():
    assert correction_command("תקן מען") == ("defendant.address", "")
    assert correction_command("תקן מען הקרן 40 ירושלים") == (
        "defendant.address",
        "הקרן 40 ירושלים",
    )
    assert correction_command("תקן מען: כמו מקום") == ("defendant.address", "כמו מקום")
    case = CaseData(incident_place="הקרן 40 ירושלים")
    updated = apply_chat_answer(case, "defendant.address", "כמו מקום")
    assert updated.defendant.address == "הקרן 40 ירושלים"


def test_latin_only_address_is_rejected():
    try:
        apply_chat_answer(CaseData(), "defendant.address", "njkfdsa fds")
        assert False, "Latin keyboard gibberish should be rejected"
    except InterviewAnswerError as exc:
        assert "מקלדת האנגלית" in str(exc)


def test_recall_request_returns_existing_value_without_mutation():
    case = CaseData(transaction_date="01.02.2026")
    assert is_recall_request("מה הזנתי?")
    assert field_value_for_user(case, "transaction_date") == "01.02.2026"


def test_consistency_date_issue_needs_actual_chronology_conflict():
    valid = CaseData(transaction_date="01.02.2026", incident_date="10.02.2026")
    invalid = CaseData(transaction_date="10.02.2026", incident_date="01.02.2026")
    assert not consistency_issue_is_supported(valid, "transaction_date")
    assert consistency_issue_is_supported(invalid, "transaction_date")


def test_confirmation_summary_does_not_repeat_raw_litigation_request():
    case = seed_case_from_opening(SAMPLE_OPENING)
    summary = build_confirmation_summary(case)
    assert "אני רוצה לתבוע" not in summary
    assert "המזגן התקלקל אחרי שבועיים" in summary


def test_tel_aviv_typo_gets_specific_suggestion_but_correct_city_passes():
    try:
        apply_chat_answer(CaseData(), "incident_place", "תל אבי רחוב הארבעה 4")
        assert False, "misspelled Tel Aviv should be rejected"
    except InterviewAnswerError as exc:
        assert "תל אביב" in str(exc)
    case = apply_chat_answer(CaseData(), "incident_place", "תל אביב רחוב הארבעה 4")
    assert case.incident_place == "תל אביב רחוב הארבעה 4"


def test_future_transaction_and_incident_dates_are_rejected_immediately():
    for field in ("transaction_date", "incident_date"):
        try:
            apply_chat_answer(CaseData(), field, "1.1.30")
            assert False, "a future case date should not be accepted"
        except InterviewAnswerError as exc:
            assert "בעתיד" in str(exc)


def test_invalid_month_returns_the_specific_date_error():
    try:
        apply_chat_answer(CaseData(), "transaction_date", "15.15.20")
        assert False, "month 15 should be rejected"
    except InterviewAnswerError as exc:
        assert "החודש 15 אינו תקין" in str(exc)


def test_incident_date_cannot_precede_transaction_date():
    case = apply_chat_answer(CaseData(), "transaction_date", "10.1.26")
    try:
        apply_chat_answer(case, "incident_date", "5.1.26")
        assert False, "the incident cannot precede the transaction"
    except InterviewAnswerError as exc:
        assert "מוקדם מתאריך העסקה" in str(exc)


def test_business_address_reference_reuses_prior_location():
    case = CaseData(incident_place="תל אביב רחוב הארבעה 4")
    case = apply_chat_answer(case, "defendant.address", "כתובת העסק")
    assert case.defendant.address == "תל אביב רחוב הארבעה 4"


def test_short_same_address_reference_reuses_full_location():
    case = CaseData(incident_place="אצטדיון טדי, מועדון ספורט מכבי 1, ירושלים")
    case = apply_chat_answer(case, "defendant.address", "אותה")
    assert case.defendant.address == "אצטדיון טדי, מועדון ספורט מכבי 1, ירושלים"


def test_misspelled_unknown_is_accepted_for_optional_phone():
    case = apply_chat_answer(CaseData(), "defendant.phone", "לא ידו")
    assert case.defendant.phone == ""


def test_natural_blame_and_refusal_are_recognized_as_contact_outcome():
    first = "ההתקשרתי אליהם פעמיים והם מתנערים ואומרים שאני אשם"
    second = "התקשרתי אליהם פעמיים והם לא מוכנים לזכות"
    assert apply_chat_answer(CaseData(), "prior_attempts", first).prior_attempts == [first]
    assert apply_chat_answer(CaseData(), "prior_attempts", second).prior_attempts == [second]


def test_no_refunds_is_a_contact_outcome_and_fragments_can_accumulate():
    complete = "התקשרתי לחנות, ואמרו שאין החזרות או פיצויים"
    assert apply_chat_answer(CaseData(), "prior_attempts", complete).prior_attempts == [complete]

    case = CaseData(product_or_service_details="מזגן")
    assert prior_attempt_fragment_is_relevant("התקשרתי", case)
    combined = merge_partial_answer("התקשרתי", "אמרו שלא מחזירים")
    assert apply_chat_answer(case, "prior_attempts", combined).prior_attempts == [combined]


def test_in_person_double_charge_complaint_and_denial_are_accepted():
    complete = "אמרתי לו שהוא חייב אותי כפול והוא טען שלא"
    assert apply_chat_answer(CaseData(), "prior_attempts", complete).prior_attempts == [complete]

    partial = "הלכתי למסעדה ואמרתי לו שחייב כפול"
    try:
        apply_chat_answer(CaseData(), "prior_attempts", partial)
        assert False, "a complaint without the restaurant's response is incomplete"
    except InterviewAnswerError as exc:
        assert "התגובה שקיבלתם" in str(exc)


def test_previous_restaurant_name_can_be_reused_from_place():
    case = CaseData(incident_place="מסעדת יוסי ובניו, קניון מלחה ירושלים")
    updated = apply_chat_answer(case, "defendant.full_name", "כתבתי קודם")
    assert updated.defendant.full_name == "מסעדת יוסי ובניו"
    same = apply_chat_answer(case, "defendant.full_name", "אותו דבר")
    assert same.defendant.full_name == "מסעדת יוסי ובניו"


def test_claimant_saying_amarti_is_not_mistaken_for_defendant_response():
    case = CaseData(product_or_service_details="מזגן")
    incomplete = "התקשרתי אליהם פעמיים ואמרתי להם שהמזגן התקלקל"
    assert prior_attempt_fragment_is_relevant(incomplete, case)
    try:
        apply_chat_answer(case, "prior_attempts", incomplete)
        assert False, "the claimant's own statement is not the defendant's response"
    except InterviewAnswerError as exc:
        assert "התגובה שקיבלתם" in str(exc)

    complete = merge_partial_answer(incomplete, "הם אמרו שלא יתקנו אותו")
    updated = apply_chat_answer(case, "prior_attempts", complete)
    assert updated.prior_attempts == [complete]


def test_phone_contact_synonyms_are_recognized():
    case = CaseData(product_or_service_details="מזגן")
    variants = (
        "צלצלתי אליהם שבע פעמים ואמרו לי שלא יתקנו את המזגן",
        "חייגתי לחנות והם אמרו שלא יחזירו לי כסף",
        "טלפנתי לשירות והנציגה אמרה שאין אחריות",
        "דיברתי איתם בטלפון והם סירבו לתקן את התקלה",
    )
    for answer in variants:
        assert apply_chat_answer(case, "prior_attempts", answer).prior_attempts == [answer]


def test_verified_warranty_details_are_always_visible_in_draft():
    case = air_conditioner_sample()
    case.warranty_details = "יש אחריות, אמרתי להם בחנות"
    records = retrieve_authorities_for_case(case)
    sections = DemoAgentClient().draft_sections(
        case, [record.model_dump(mode="json") for record in records]
    )
    # Simulate Gemini omitting or paraphrasing the warranty incorrectly.
    sections.factual_background = [
        paragraph for paragraph in sections.factual_background if "אחריות" not in paragraph
    ]
    markdown = compose_claim_markdown(case, sections, records)
    assert "## אחריות ושירות לאחר מכירה" in markdown
    assert "יש אחריות, אמרתי להם בחנות" in markdown
    assert not validate_draft(markdown, {record.id for record in records}, case)


def test_short_restaurant_opening_is_a_complete_service_dispute():
    opening = "בעל מסעדה עקץ אותי וחייב אותי פעמיים בסכום כולל של 20 אלף"
    case = seed_case_from_opening(opening)
    assert opening_story_is_complete(opening)
    assert case.category == ClaimCategory.SERVICE_CONTRACT
    assert case.claim_total == Decimal("20000")


def test_rental_deposit_opening_is_complete_and_extracts_rented_subject():
    opening = "שכרתי דירה ושילמתי 5000 שקל פיקדון והמשכיר מסרב להחזיר"
    case = seed_case_from_opening(opening)
    assert opening_story_is_complete(opening)
    assert case.product_or_service_details == "דירה"
    assert case.category == ClaimCategory.LANDLORD_TENANT


def test_plural_damage_correction_command_is_understood():
    assert correction_command("תקן סכומים 10000") == ("damage_items", "10000")
    assert correction_command("תקן נזקים החזר חיוב כפול 10000") == (
        "damage_items",
        "החזר חיוב כפול 10000",
    )


def test_final_draft_contains_confirmed_case_description_dates_and_place():
    case = air_conditioner_sample()
    records = retrieve_authorities_for_case(case)
    sections = DemoAgentClient().draft_sections(
        case, [record.model_dump(mode="json") for record in records]
    )
    markdown = compose_claim_markdown(case, sections, records)

    assert "### תיאור המקרה" in markdown
    assert "כפי שאושר בראיון" not in markdown
    assert "שמסר/ה התובע/ת בראיון" not in markdown
    assert case.event_summary.rstrip(".") not in markdown
    assert "התובע/ת רכש/ה מהנתבע/ת מזגן" in markdown
    assert "לפי תיאור התובע/ת" in markdown
    factual_section = markdown.split("## הרקע העובדתי", 1)[1].split("## עילת התביעה", 1)[0]
    assert "1.6.2026" in factual_section
    assert "15.6.2026" in factual_section
    assert case.incident_place in factual_section
    assert not validate_draft(markdown, {record.id for record in records}, case)


def test_simple_first_person_defect_is_formalized_and_passes_final_validation():
    case = air_conditioner_sample()
    case.product_or_service_details = "מזגן"
    case.event_summary = "קניתי מזגן ב200 שקל והוא התקלקל אחרי יומיים"
    case.damage_items = [DamageItem(label="השבת מחיר המוצר", amount=Decimal("200"))]
    records = retrieve_authorities_for_case(case)
    sections = DemoAgentClient().draft_sections(
        case, [record.model_dump(mode="json") for record in records]
    )

    markdown = compose_claim_markdown(case, sections, records)
    description = markdown.split("### תיאור המקרה", 1)[1].split("### השתלשלות האירועים", 1)[0]

    assert "המזגן התקלקל אחרי יומיים" in description
    assert "קניתי" not in description
    assert not validate_draft(markdown, {record.id for record in records}, case)


def test_mixed_email_and_phone_attempt_keeps_channels_counts_and_response():
    case = CaseData(product_or_service_details="מזגן")
    attempt = "פניתי פעמיים במייל והתקשרתי שלוש פעמים ואמרו לי שהמזגן תקין"

    formal = formalize_resolution_attempt(attempt, case)

    assert "פעמיים בדוא״ל" in formal
    assert "שלוש פעמים בטלפון" in formal
    assert "המזגן תקין" in formal
    assert "פניתי" not in formal
    assert "התקשרתי" not in formal


def test_confirmation_accepts_completed_form_wording():
    from small_claims_agent.interview import confirmation_is_yes

    assert confirmation_is_yes("מאושר")
    assert confirmation_is_yes("מאושרת")
    assert confirmation_is_yes("מאושר!")


def test_laptop_case_description_is_formal_and_preserves_the_broken_screen_fact():
    case = air_conditioner_sample()
    case.product_or_service_details = "לפטופ"
    case.event_summary = (
        "קניתי לפטופ ב4000 שקל, פתחתי את הקופסה מהחנות וראיתי שהמסך של "
        "הלפטופ שבור, ניסיתי להחזיר את הלפטופ לחנות ולקבל את הכסף חזרה "
        "והם אמרו לי שאין החזרים ושאני שברתי אותו"
    )
    case.damage_items = [DamageItem(label="השבת מחיר המוצר", amount=Decimal("4000"))]
    records = retrieve_authorities_for_case(case)
    sections = DemoAgentClient().draft_sections(
        case, [record.model_dump(mode="json") for record in records]
    )
    markdown = compose_claim_markdown(case, sections, records)
    description = markdown.split("### תיאור המקרה", 1)[1].split("### השתלשלות האירועים", 1)[0]

    assert "התובע/ת רכש/ה מהנתבע/ת לפטופ" in description
    assert "מסך הלפטופ שבור" in description
    assert "קניתי" not in description
    assert "ניסיתי" not in description
    assert not validate_draft(markdown, {record.id for record in records}, case)


def test_prior_attempt_can_be_reused_from_opening_without_retyping():
    opening = (
        "קניתי לפטופ ב4000 שקל והמסך היה שבור, ניסיתי להחזיר את הלפטופ לחנות "
        "ולקבל את הכסף חזרה והם אמרו לי שאין החזרים ושאני שברתי אותו"
    )
    case = seed_case_from_opening(opening)
    extracted = extract_prior_attempt_from_opening(opening, case)

    assert extracted.startswith("ניסיתי להחזיר")
    updated = apply_chat_answer(case, "prior_attempts", "אמרתי כבר")
    assert updated.prior_attempts == [extracted]


def test_in_person_method_fragment_completes_a_pending_contact_response():
    case = CaseData(product_or_service_details="לפטופ")
    response = (
        "ניסיתי להחזיר את הלפטופ ולקבל את הכסף חזרה והם אמרו לי שאין החזרים "
        "ושאני שברתי אותו"
    )
    combined = merge_partial_answer(response, "הגעה לחנות")

    assert prior_attempt_fragment_is_relevant("הגעה לחנות", case, has_pending_fragment=True)
    assert apply_chat_answer(case, "prior_attempts", combined).prior_attempts == [combined]


def test_valid_user_revision_regenerates_different_pdf_bytes():
    case = air_conditioner_sample()
    records = retrieve_authorities_for_case(case)
    sections = DemoAgentClient().draft_sections(
        case, [record.model_dump(mode="json") for record in records]
    )
    original = compose_claim_markdown(case, sections, records)
    revised = original.replace("לפי תיאור התובע/ת", "לטענת התובע/ת")

    assert revised != original
    assert not validate_draft(revised, {record.id for record in records}, case)
    assert markdown_to_pdf_bytes(revised) != markdown_to_pdf_bytes(original)


def test_streamlit_allows_valid_draft_edit_and_pdf_refresh(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    case = air_conditioner_sample()
    records = retrieve_authorities_for_case(case)
    sections = DemoAgentClient().draft_sections(
        case, [record.model_dump(mode="json") for record in records]
    )
    markdown = compose_claim_markdown(case, sections, records)
    app.session_state["case"] = case
    app.session_state["started"] = True
    app.session_state["ready"] = True
    app.session_state["draft_markdown"] = markdown
    app.session_state["draft_editor"] = markdown
    app.session_state["draft_authority_ids"] = {record.id for record in records}
    app.session_state["draft_authorities"] = [record.model_dump(mode="json") for record in records]
    app.run()

    download_labels = {button.label for button in app.get("download_button")}
    assert "הורדת PDF של כתב התביעה" in download_labels
    assert "הורדת דו״ח מקורות כ-PDF" in download_labels

    editor = next(area for area in app.text_area if area.label == "עריכת טיוטת כתב התביעה")
    revised = markdown.replace("לפי תיאור התובע/ת", "לטענת התובע/ת")
    editor.set_value(revised).run()
    save_button = next(button for button in app.button if button.label == "שמירת שינויים ועדכון ה-PDF")
    save_button.click().run()

    assert not app.exception
    assert app.session_state["draft_revision_count"] == 1
    assert "לטענת התובע/ת" in app.session_state["draft_markdown"]


def test_court_draft_has_brief_legal_basis_but_not_full_source_catalog():
    case = air_conditioner_sample()
    records = retrieve_authorities_for_case(case)
    sections = DemoAgentClient().draft_sections(
        case, [record.model_dump(mode="json") for record in records]
    )
    markdown = compose_claim_markdown(case, sections, records)

    assert "עילת התביעה" in markdown
    assert "סעיף 11 לחוק המכר" in markdown
    assert "מקורות משפטיים שנבדקו" not in markdown
    assert "מקור רשמי" not in markdown
    assert not any(record.source_url in markdown for record in records)
    assert not any(record.quote_he in markdown for record in records)


def test_ai_source_report_contains_retrieved_sources_and_quality_findings():
    case = air_conditioner_sample()
    records = retrieve_authorities_for_case(case)
    review = {
        "verdict": "revise",
        "clarity_issues": ["יש לקצר משפט אחד."],
        "revision_instructions": ["בדקו את ניסוח הסעד החלופי."],
    }
    report = compose_source_report_markdown(
        case,
        records,
        review,
        ["אזהרת בדיקה לדוגמה"],
        "gemini-test-model",
    )

    assert "אינו מיועד להגשה לבית המשפט" in report
    assert "אחזור מובנה ומבוסס-כללים" in report
    assert "לא בוצע חיפוש וקטורי" in report
    assert "נדרשת בדיקה" in report
    assert "יש לקצר משפט אחד" in report
    assert "אזהרת בדיקה לדוגמה" in report
    for record in records:
        assert record.id in report
        assert record.quote_he in report
        assert record.source_url in report
    assert "נוסח מאוחד ציבורי (אינו מקור רשמי)" in report
    assert "תמצית שמורה" in report
    assert markdown_to_pdf_bytes(report, title="דו״ח מקורות ובקרת איכות").startswith(b"%PDF")


def test_unrelated_sports_comments_are_not_relevant_resolution_attempts():
    case = CaseData(product_or_service_details="מזגן")
    first = "באתי לכל משחק ליגה העונה, ובכולם הם הפסידו"
    second = "צעקתי במשחק שהם משחקים חרא"
    assert not prior_attempt_fragment_is_relevant(first, case)
    assert not prior_attempt_fragment_is_relevant(second, case)
    for message in (first, second):
        try:
            apply_chat_answer(case, "prior_attempts", message)
            assert False, "off-topic sports content must not enter case state"
        except InterviewAnswerError as exc:
            assert "למחלוקת הנוכחית" in str(exc)
    assert not case.prior_attempts


def test_partial_answer_merge_does_not_duplicate_repeated_message():
    answer = "התקשרתי אליהם פעמיים"
    assert merge_partial_answer(answer, answer) == answer
    assert merge_partial_answer(answer, "הם אומרים שאני אשם") == (
        "התקשרתי אליהם פעמיים הם אומרים שאני אשם"
    )


def test_llm_repair_proposal_must_pass_local_validation():
    class FakeClient:
        available = True

        def repair_answer(self, field, question, failed_answers, case):
            return AnswerRepair(
                action="accept",
                repaired_answer="01.01.2026",
                confidence=0.95,
            )

    outcome = attempt_llm_repair(
        FakeClient(),
        CaseData(),
        "transaction_date",
        "מה תאריך העסקה?",
        ["בתחילת ינואר", "הראשון בינואר"],
    )
    assert outcome and outcome.accepted
    assert outcome.case and outcome.case.transaction_date == "01.01.2026"
    assert outcome.assessment and outcome.assessment.source == "model"


def test_llm_repair_is_never_called_for_pii_fields():
    class FailIfCalled:
        available = True

        def repair_answer(self, *args, **kwargs):
            raise AssertionError("PII must not be sent to the model")

    outcome = attempt_llm_repair(
        FailIfCalled(),
        CaseData(),
        "claimant.full_name",
        "מה שמך?",
        ["כתבתי", "כבר כתבתי"],
    )
    assert outcome is None


def test_streamlit_uses_llm_repair_on_second_failed_non_pii_answer(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest
    from small_claims_agent.llm import GeminiAgentClient

    repair_calls: list[list[str]] = []

    def fake_extract(self, user_text, case):
        return StateExtraction(inferred_category=classify_category(user_text, case.category))

    def fake_repair(self, field, question, failed_answers, case):
        repair_calls.append(failed_answers)
        return AnswerRepair(
            action="accept",
            repaired_answer="01.01.2026",
            confidence=0.95,
        )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(GeminiAgentClient, "extract", fake_extract)
    monkeypatch.setattr(GeminiAgentClient, "repair_answer", fake_repair)

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    app.chat_input[0].set_value("קניתי מזגן ב3200 שקל והוא התקלקל").run()
    app.chat_input[0].set_value("בתחילת ינואר").run()
    assert not repair_calls
    app.chat_input[0].set_value("הראשון בינואר").run()

    assert repair_calls == [["בתחילת ינואר", "הראשון בינואר"]]
    assert app.session_state["case"].transaction_date == "01.01.2026"
    assert app.session_state["awaiting_field"] == "incident_date"


def test_reference_aliases_reuse_prior_business_address_and_attempt():
    case = CaseData(
        event_summary=(
            "קניתי מקרר ב4000 שקל והוא התקלקל אחרי שבוע. "
            "פניתי לחנות והם אמרו שאין אחריות"
        ),
        incident_place="חנות מחסני חשמל ירושלים רחוב המסגר 4",
        product_or_service_details="מקרר",
    )

    case = apply_chat_answer(case, "defendant.full_name", "כתבתי כבר")
    case = apply_chat_answer(case, "defendant.address", "כתבתי כבר")
    case = apply_chat_answer(case, "prior_attempts", "השתמש")

    assert case.defendant.full_name == "מחסני חשמל"
    assert case.defendant.address == "ירושלים רחוב המסגר 4"
    assert case.prior_attempts == ["פניתי לחנות והם אמרו שאין אחריות"]


def test_streamlit_merges_business_and_address_place_fragments(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    for message in (
        "קניתי מקרר ב4000 שקל ואחרי שבוע הוא הפסיק לעבוד",
        "1.2.26",
        "8.2.26",
        "חנות מחסני חשמל",
        "ירושלים רחוב המסגר 4",
        "כתבתי כבר",
    ):
        app.chat_input[0].set_value(message).run()

    case = app.session_state["case"]
    assert case.incident_place == "חנות מחסני חשמל ירושלים רחוב המסגר 4"
    assert case.defendant.full_name == "מחסני חשמל"
    assert app.session_state["awaiting_field"] == "defendant_role"
