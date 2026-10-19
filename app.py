from __future__ import annotations

import importlib
import json
import logging
import os
import re
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from small_claims_agent import interview as interview_module
from small_claims_agent import llm as llm_module

# Streamlit may hot-reload this file while retaining an older cached helper
# module. Reload once when a newly added public helper is missing.
if not all(
    hasattr(interview_module, name)
    for name in (
        "correction_command",
        "is_recall_request",
        "is_current_question_help_request",
        "is_current_question_reset_request",
        "build_repeated_failure_guidance",
        "field_value_for_user",
        "merge_damage_confirmation",
        "damage_confirmation_intent",
        "normalize_opening_keyboard_layout",
        "prior_attempt_fragment_is_relevant",
        "opening_story_missing_parts",
        "defendant_name_reference_candidate",
        "confirmation_is_no",
        "merge_incident_place_answer",
    )
):
    interview_module = importlib.reload(interview_module)
if getattr(interview_module, "HOT_RELOAD_VERSION", 0) < 6:
    interview_module = importlib.reload(interview_module)
if not hasattr(llm_module, "LLMGateway") or not hasattr(
    llm_module.LLMGateway, "review_draft_for_clarification"
):
    llm_module = importlib.reload(llm_module)

from small_claims_agent.graph import _sanitize_draft_review, build_graph, initial_state
from small_claims_agent.drafting import compose_source_report_markdown
from small_claims_agent.draft_clarification import sanitize_draft_clarifications
from small_claims_agent.interview import (
    InterviewAnswerError,
    SAMPLE_OPENING,
    WELCOME_MESSAGE,
    apply_chat_answer_with_assessment,
    build_confirmation_summary,
    build_live_draft_preview,
    build_repeated_failure_guidance,
    damage_confirmation_intent,
    confirmation_is_yes,
    correction_command,
    field_value_for_user,
    inferred_completed_fields,
    is_recall_request,
    is_current_question_help_request,
    is_current_question_reset_request,
    interview_progress,
    merge_incident_place_answer,
    merge_partial_answer,
    merge_damage_confirmation,
    next_interview_field,
    normalize_opening_keyboard_layout,
    opening_story_missing_parts,
    defendant_name_reference_candidate,
    confirmation_is_no,
    prior_attempt_fragment_is_relevant,
    question_for,
    seed_case_from_opening,
)
from small_claims_agent.llm import LLMGateway
from small_claims_agent.models import AuthorityRecord, CaseData
from small_claims_agent.pdf_export import markdown_to_pdf_bytes
from small_claims_agent.privacy import free_text_privacy_findings
from small_claims_agent.repair import (
    CONSISTENCY_FIELDS,
    attempt_llm_repair,
    consistency_issue_is_supported,
)
from small_claims_agent.validators import (
    SMALL_CLAIM_LIMIT_ILS,
    draft_cross_field_warnings,
    validate_case,
    validate_draft,
)


load_dotenv()
st.set_page_config(page_title="סוכן תביעות קטנות", page_icon="⚖️", layout="centered")
st.markdown(
    """
    <style>
    html, body, [class*="css"], .stApp { direction: rtl; text-align: right; }
    [data-testid="stChatMessage"] { direction: rtl; text-align: right; }
    [data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
    .privacy { background:#fff4ec; border-right:4px solid #d04a02; padding:12px; border-radius:6px; }
    .summary { background:#f5f7f8; padding:12px; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


LOG_PATH = Path(__file__).resolve().parent / "logs" / "interview.log"
PERSISTENT_DIAGNOSTICS_ENABLED = os.getenv("ENABLE_DIAGNOSTIC_FILE_LOG", "0").strip() == "1"
FINAL_FAILURE_GUIDANCE_THRESHOLD = 6


def _diagnostic_logger() -> logging.Logger:
    logger = logging.getLogger("small_claims_agent.interview")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler: logging.Handler = logging.NullHandler()
        if PERSISTENT_DIAGNOSTICS_ENABLED:
            try:
                LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(
                    LOG_PATH,
                    maxBytes=1_000_000,
                    backupCount=2,
                    encoding="utf-8",
                )
            except OSError:
                # Diagnostics must never prevent the interview from starting.
                handler = logging.NullHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


DIAGNOSTIC_LOGGER = _diagnostic_logger()


def record_diagnostic(event: str, **details) -> None:
    """Record parser decisions without storing party identification fields."""
    entry = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_id": st.session_state.get("diagnostic_session_id", "startup"),
        "event": event,
        **details,
    }
    if "diagnostics" in st.session_state:
        st.session_state.diagnostics.append(entry)
        st.session_state.diagnostics = st.session_state.diagnostics[-100:]
    DIAGNOSTIC_LOGGER.info(json.dumps(entry, ensure_ascii=False, default=str))


def reset_interview() -> None:
    client = LLMGateway()
    st.session_state.case = CaseData()
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.awaiting_field = None
    st.session_state.completed_fields = set()
    st.session_state.started = False
    st.session_state.ready = False
    st.session_state.awaiting_confirmation = False
    st.session_state.pending_field_confirmation = None
    st.session_state.preview_return_question = ""
    st.session_state.observations = {}
    st.session_state.partial_answers = {}
    st.session_state.draft_markdown = ""
    st.session_state.draft_editor = ""
    st.session_state.draft_review = {}
    st.session_state.draft_warnings = []
    st.session_state.draft_error = ""
    st.session_state.draft_edit_error = ""
    st.session_state.draft_authority_ids = set()
    st.session_state.draft_authorities = []
    st.session_state.draft_revision_count = 0
    st.session_state.draft_clarifications = []
    st.session_state.draft_clarification_index = 0
    st.session_state.draft_clarification_error = ""
    st.session_state.draft_clarification_skipped = False
    st.session_state.draft_clarification_changed = False
    st.session_state.model_status = client.model_name if client.available else "מצב מקומי (לא הוגדר מפתח ספק)"
    st.session_state.client = client
    st.session_state.failure_counts = {}
    st.session_state.failed_answers = {}
    st.session_state.failure_guidance_shown = set()
    st.session_state.consistency_checked = False
    st.session_state.consistency_review_count = 0
    st.session_state.diagnostic_session_id = str(uuid.uuid4())
    st.session_state.diagnostics = []
    st.session_state.graph = build_graph(client)
    record_diagnostic(
        "interview_reset",
        model_available=client.available,
        model=client.model_name,
        reason=None if client.available else "provider_api_key_missing",
    )


def enrich_opening_with_model(opening: str, local_case: CaseData) -> CaseData:
    if not st.session_state.client.available:
        record_diagnostic("opening_model_skipped", reason="missing_api_key")
        return local_case
    # Start from deterministic local captures so the LLM can enrich them but can
    # never erase the amount or category already recognized from the opening.
    state = initial_state(local_case)
    state["last_user_message"] = opening
    try:
        result = st.session_state.graph.invoke(
            state,
            {"configurable": {"thread_id": f"opening-{uuid.uuid4()}"}},
        )
    except Exception as exc:
        # Model enrichment is optional. The deterministic local capture is a
        # complete safe fallback and the conversation must always continue.
        st.session_state.model_status = f"מצב מקומי ({type(exc).__name__})"
        record_diagnostic(
            "opening_model_failed",
            **getattr(exc, "diagnostic", {"exception_type": type(exc).__name__}),
        )
        return local_case
    model_case = CaseData.model_validate(result["case"])
    # Keep deterministic local captures whenever the model left a field empty.
    # The LLM may classify the dispute, but it may never rewrite the user's story
    # or infer facts that were not captured deterministically from the opening.
    model_case.event_summary = local_case.event_summary
    model_case.transaction_date = local_case.transaction_date
    model_case.incident_date = local_case.incident_date
    model_case.incident_place = local_case.incident_place
    model_case.warranty_details = local_case.warranty_details
    model_case.defendant_role = local_case.defendant_role
    model_case.product_or_service_details = local_case.product_or_service_details
    # The raw opening is context, not a ready-made chronology or contact history.
    model_case.chronology = []
    model_case.prior_attempts = []
    model_case.damage_items = local_case.damage_items
    if model_case.category.value == "unsupported":
        model_case.category = local_case.category
    raw_model_status = str(result.get("model_status", st.session_state.client.model_name))
    fallback_used = bool(result.get("manual_mode")) or raw_model_status.startswith(
        ("All configured LLM providers failed", "Provider output failed")
    )
    if fallback_used:
        diagnostic = result.get("model_diagnostic") or {}
        category = diagnostic.get("category", raw_model_status.rsplit(":", 1)[-1].strip())
        st.session_state.model_status = f"מצב מקומי (LLM: {category})"
        record_diagnostic("opening_model_fallback", **diagnostic)
    else:
        for diagnostic in result.get("model_diagnostics", []):
            record_diagnostic(
                "llm_provider_failed",
                **{**diagnostic, "llm_operation": diagnostic.get("operation"), "operation": "opening_extract"},
            )
        st.session_state.model_status = raw_model_status
        record_diagnostic(
            "opening_model_completed",
            model_status=st.session_state.model_status,
            facts_overwritten=False,
        )
    return model_case


def run_consistency_checkpoint(case: CaseData) -> bool:
    """Return True when the interview was rerouted to a factual clarification."""
    if st.session_state.consistency_checked:
        return False
    client = st.session_state.client
    if not client.available or st.session_state.consistency_review_count >= 2:
        st.session_state.consistency_checked = True
        return False
    st.session_state.consistency_review_count += 1
    try:
        review = client.review_interview(case)
    except Exception as exc:
        # A consistency review is advisory. A temporary provider, quota or
        # hot-reload error must never destroy an otherwise valid interview.
        record_diagnostic(
            "consistency_review_skipped",
            **getattr(exc, "diagnostic", {"exception_type": type(exc).__name__}),
        )
        st.session_state.consistency_checked = True
        return False

    issue = next(
        (
            item
            for item in review.issues
            if item.field in CONSISTENCY_FIELDS
            and consistency_issue_is_supported(case, item.field)
        ),
        None,
    )
    if review.verdict == "clarify" and issue:
        st.session_state.completed_fields.discard(issue.field)
        st.session_state.awaiting_field = issue.field
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"בדיקת העקביות מצאה נקודה שכדאי לברר: {issue.clarification_question}",
            }
        )
        return True
    st.session_state.consistency_checked = True
    return False


def advance_interview() -> None:
    case = st.session_state.case
    completed = st.session_state.completed_fields
    validation = validate_case(case)

    if validation.errors:
        if case.claim_total > SMALL_CLAIM_LIMIT_ILS:
            completed.discard("damage_items")
            st.session_state.awaiting_field = "damage_items"
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": validation.errors[0] + " אנא פרטו מחדש את הסכומים שייכללו בתביעה.",
                }
            )
            return
        date_error = next(
            (
                error
                for error in validation.errors
                if "נמצא בעתיד" in error or "מוקדם ממועד העסקה" in error
            ),
            None,
        )
        if date_error:
            date_field = "transaction_date" if "מועד העסקה נמצא בעתיד" in date_error else "incident_date"
            completed.discard(date_field)
            st.session_state.awaiting_field = date_field
            st.session_state.messages.append(
                {"role": "assistant", "content": date_error + "\n\n" + question_for(date_field, case)}
            )
            return
        unresolved = next((error for error in validation.errors if "לא ניתן להסיק בבטחה" in error), None)
        if unresolved:
            completed.discard("incident_place")
            st.session_state.awaiting_field = "incident_place"
            st.session_state.messages.append(
                {"role": "assistant", "content": unresolved + "\n\n" + question_for("incident_place", case)}
            )
            return
        st.session_state.messages.append({"role": "assistant", "content": "\n".join(validation.errors)})
        return

    field = next_interview_field(case, completed)
    if field:
        st.session_state.awaiting_field = field
        record_diagnostic(
            "next_field",
            field=field,
            completed_count=len(completed),
            opening_missing=(opening_story_missing_parts(case.event_summary) if field == "event_summary" else []),
        )
        st.session_state.messages.append({"role": "assistant", "content": question_for(field, case)})
        return

    # Final LLM judge: all non-PII and PII fields have been collected, while
    # safe_case_context keeps identifying details out of the model request.
    if run_consistency_checkpoint(case):
        return

    st.session_state.awaiting_field = None
    st.session_state.awaiting_confirmation = True
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": build_confirmation_summary(case),
        }
    )


def append_live_draft_preview_to_chat() -> None:
    """Show every filled and missing interview field inside the conversation."""
    content = build_live_draft_preview(
        st.session_state.case,
        st.session_state.completed_fields,
        st.session_state.partial_answers,
    )
    record_diagnostic(
        "live_draft_preview",
        field=st.session_state.awaiting_field or "complete",
        completed_count=len(st.session_state.completed_fields),
    )
    st.session_state.messages.append({"role": "assistant", "content": content})
    if st.session_state.awaiting_confirmation:
        follow_up = "אם כל העובדות נכונות, כתבו ״מאשר/ת״. לתיקון, כתבו למשל ״תקן תאריך עסקה״."
    elif st.session_state.awaiting_field:
        follow_up = question_for(st.session_state.awaiting_field, st.session_state.case)
        pending = st.session_state.pending_field_confirmation
        if pending:
            follow_up = (
                f"זיהיתי שם אפשרי לנתבע/ת: ״{pending['candidate']}״. "
                "האם התכוונתם לשם הזה? כתבו ״כן״ כדי להשתמש בו, או ״לא״ ואז כתבו את השם הנכון."
            )
    else:
        follow_up = "אפשר להמשיך מהשאלה האחרונה או להתחיל מחדש."
    st.session_state.preview_return_question = follow_up
    st.session_state.messages.append({"role": "assistant", "content": follow_up})


def process_user_message(message: str) -> None:
    st.session_state.preview_return_question = ""
    st.session_state.messages.append({"role": "user", "content": message})
    current_field = "opening" if not st.session_state.started else (
        "confirmation" if st.session_state.awaiting_confirmation else st.session_state.awaiting_field or "complete"
    )
    record_diagnostic(
        "message_received",
        field=current_field,
        character_count=len(message),
        content_logged=False,
    )
    if st.session_state.get("draft_clarifications"):
        process_draft_clarification_answer(message)
        return
    if not st.session_state.started:
        opening, keyboard_converted = normalize_opening_keyboard_layout(message)
        if re.search(r"[A-Za-z]", message) and not re.search(r"[א-ת]", message) and not keyboard_converted:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "נראה שהתיאור נכתב באנגלית או כשהמקלדת הייתה על אנגלית. החליפו לעברית וכתבו שוב בקצרה מה קרה.",
                }
            )
            record_diagnostic("opening_rejected", reason="unrecognized_latin_text")
            return
        if keyboard_converted:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"זיהיתי שהטקסט הוקלד כשהמקלדת הייתה באנגלית, והמרתי אותו כך: ״{opening}״.",
                }
            )
        privacy_findings = free_text_privacy_findings(opening)
        if privacy_findings:
            labels = ", ".join(privacy_findings)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"זיהיתי בתיאור הראשוני פרטי זיהוי אפשריים: {labels}. "
                        "כדי שלא לשלוח אותם לספקי המודל, כתבו שוב רק מה קרה, מה נרכש או הוזמן, "
                        "כמה שולם ומה השתבש - בלי שם, מספר זהות, טלפון, דוא״ל או כתובת מגורים. "
                        "את הפרטים האלה אבקש בהמשך בשדות מקומיים נפרדים."
                    ),
                }
            )
            record_diagnostic("opening_rejected", reason="possible_pii", finding_count=len(privacy_findings))
            return
        local_case = seed_case_from_opening(opening)
        st.session_state.case = enrich_opening_with_model(opening, local_case)
        record_diagnostic(
            "opening_parsed",
            keyboard_converted=keyboard_converted,
            missing_parts=opening_story_missing_parts(st.session_state.case.event_summary),
            detected_product=st.session_state.case.product_or_service_details or "",
            detected_amount=str(st.session_state.case.claim_total),
            category=st.session_state.case.category.value,
        )
        st.session_state.started = True
        st.session_state.completed_fields.update(inferred_completed_fields(st.session_state.case))
        advance_interview()
        return

    if st.session_state.awaiting_confirmation:
        if confirmation_is_yes(message):
            data = st.session_state.case.model_dump(mode="python")
            data["facts_confirmed"] = True
            st.session_state.case = CaseData.model_validate(data)
            st.session_state.awaiting_confirmation = False
            st.session_state.ready = True
            validation = validate_case(st.session_state.case)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "העובדות אושרו. סכום התביעה הכספי הוא "
                        f"{validation.claim_total:,.0f} ₪, והאגרה המשוערת היא {validation.filing_fee:,.0f} ₪ בנפרד. "
                        "אפשר כעת ליצור את הטיוטה."
                    ),
                }
            )
            return
        command = correction_command(message)
        if command:
            correction, corrected_value = command
            if corrected_value:
                try:
                    updated, assessment = apply_chat_answer_with_assessment(
                        st.session_state.case, correction, corrected_value
                    )
                except InterviewAnswerError as exc:
                    st.session_state.messages.append({"role": "assistant", "content": str(exc)})
                    return
                data = updated.model_dump(mode="python")
                data["facts_confirmed"] = False
                st.session_state.case = CaseData.model_validate(data)
                st.session_state.observations[correction] = assessment.model_dump(mode="json")
                st.session_state.completed_fields.add(correction)
                st.session_state.draft_markdown = ""
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "התיקון נקלט. הנה הסיכום המעודכן:\n\n"
                        + build_confirmation_summary(st.session_state.case),
                    }
                )
                return
            st.session_state.awaiting_confirmation = False
            st.session_state.awaiting_field = correction
            st.session_state.completed_fields.discard(correction)
            st.session_state.messages.append(
                {"role": "assistant", "content": question_for(correction, st.session_state.case)}
            )
            return
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "לא הצלחתי לזהות אישור או שדה לתיקון. כתבו ״מאשר/ת״, או למשל ״תקן תאריך עסקה״.",
            }
        )
        return

    field = st.session_state.awaiting_field
    if not field:
        st.session_state.messages.append(
            {"role": "assistant", "content": "הראיון כבר הושלם. אפשר ליצור טיוטה או להתחיל מחדש."}
        )
        return
    parser_message_override = None
    pending_name = st.session_state.pending_field_confirmation
    if pending_name and field == "defendant.full_name":
        if confirmation_is_yes(message):
            parser_message_override = pending_name["candidate"]
            st.session_state.pending_field_confirmation = None
        elif confirmation_is_no(message):
            st.session_state.pending_field_confirmation = None
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "בסדר, לא אשתמש בשם שהצעתי. כתבו את השם המשפטי של הנתבע/ת כפי שמופיע בקבלה או בחשבונית.",
                }
            )
            return
        else:
            # A correction can be supplied directly or prefixed with a clear negation.
            parser_message_override = re.sub(
                r"^\s*לא\s*(?:,|[-:])?\s*(?:השם הנכון הוא|התכוונתי ל|זה)\s*",
                "",
                message,
                flags=re.IGNORECASE,
            ).strip()
            st.session_state.pending_field_confirmation = None
    if is_current_question_reset_request(message):
        st.session_state.partial_answers.pop(field, None)
        st.session_state.failure_counts.pop(field, None)
        st.session_state.failed_answers.pop(field, None)
        st.session_state.failure_guidance_shown.discard(field)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "איפסתי את התשובות החלקיות לשאלה הנוכחית.\n\n"
                + question_for(field, st.session_state.case),
            }
        )
        record_diagnostic("current_question_reset", field=field)
        return
    if is_current_question_help_request(message):
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": build_repeated_failure_guidance(
                    st.session_state.case,
                    field,
                    st.session_state.partial_answers.get(field, ""),
                    st.session_state.failure_counts.get(field, 0),
                ),
            }
        )
        record_diagnostic("current_question_help_shown", field=field)
        return
    if is_recall_request(message):
        append_live_draft_preview_to_chat()
        return
    if field == "defendant.full_name" and parser_message_override is None:
        candidate = defendant_name_reference_candidate(message, st.session_state.case)
        if candidate:
            st.session_state.pending_field_confirmation = {
                "field": field,
                "candidate": candidate,
            }
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"מהמקום שתיארתם זיהיתי שם אפשרי: ״{candidate}״. "
                        "האם התכוונתם לשם הזה? כתבו ״כן״ כדי להשתמש בו, או ״לא״ ואז כתבו את השם הנכון."
                    ),
                }
            )
            record_diagnostic("defendant_name_candidate_confirmation_requested", field=field)
            return
    parser_message = parser_message_override if parser_message_override is not None else message
    if field == "prior_attempts" and confirmation_is_yes(message):
        opening_attempt = interview_module.extract_prior_attempt_from_opening(
            st.session_state.case.event_summary, st.session_state.case
        )
        if opening_attempt:
            parser_message = "השתמש במה שכתבתי"
    if field == "event_summary" and st.session_state.partial_answers.get(field):
        parser_message = merge_partial_answer(st.session_state.partial_answers[field], message)
    elif field == "incident_place" and st.session_state.partial_answers.get(field):
        # Reuse a saved place as-is for reference replies; only merge a new
        # fragment when it actually adds missing location information.
        parser_message = merge_incident_place_answer(
            st.session_state.partial_answers[field], message
        )
    elif field == "prior_attempts" and st.session_state.partial_answers.get(field):
        pending_attempt = st.session_state.partial_answers[field]
        if prior_attempt_fragment_is_relevant(
            message, st.session_state.case, has_pending_fragment=True
        ):
            parser_message = merge_partial_answer(pending_attempt, message)
    if field == "damage_items":
        pending_damage_answers = st.session_state.failed_answers.get(field, [])
        confirmation_intent = damage_confirmation_intent(pending_damage_answers, message)
        if confirmation_intent == "no":
            st.session_state.failed_answers[field] = []
            st.session_state.failure_counts[field] = 0
            st.session_state.messages.append({
                "role": "assistant",
                "content": "בסדר. שלחו את רכיבי הסכום המתוקנים, עם סכום נפרד לכל רכיב.",
            })
            record_diagnostic("damage_breakdown_rejected_by_user")
            return
        confirmed = merge_damage_confirmation(pending_damage_answers, message)
        if confirmed:
            parser_message = confirmed
    repair_notice = ""
    try:
        updated, assessment = apply_chat_answer_with_assessment(
            st.session_state.case, field, parser_message
        )
    except InterviewAnswerError as exc:
        record_diagnostic(
            "answer_rejected",
            field=field,
            reason=str(exc),
            partial_before=bool(st.session_state.partial_answers.get(field)),
        )
        if field == "event_summary":
            st.session_state.partial_answers[field] = parser_message
        elif field == "incident_place" and "כללי מדי" in str(exc):
            # Preserve a useful business/channel fragment only when the user
            # was asked to add specificity. Do not retain a typo or malformed
            # address, otherwise the corrected answer would inherit the error.
            st.session_state.partial_answers[field] = parser_message
        elif field == "prior_attempts":
            pending_attempt = st.session_state.partial_answers.get(field, "")
            current_is_relevant = prior_attempt_fragment_is_relevant(
                message,
                st.session_state.case,
                has_pending_fragment=bool(pending_attempt),
            )
            if current_is_relevant:
                st.session_state.partial_answers[field] = merge_partial_answer(
                    pending_attempt, message
                )
            # Irrelevant fragments remain out of case state but reach the
            # shared repair path after two failures instead of looping locally.
        st.session_state.failure_counts[field] = st.session_state.failure_counts.get(field, 0) + 1
        answers = st.session_state.failed_answers.setdefault(field, [])
        answers.append(message)
        st.session_state.failed_answers[field] = answers[-2:]
        outcome = None
        if st.session_state.failure_counts[field] % 2 == 0:
            record_diagnostic(
                "model_repair_attempted",
                field=field,
                failure_count=st.session_state.failure_counts[field],
            )
            outcome = attempt_llm_repair(
                st.session_state.client,
                st.session_state.case,
                field,
                question_for(field, st.session_state.case),
                st.session_state.failed_answers[field],
            )
            if outcome and outcome.diagnostic:
                record_diagnostic("llm_provider_failed", field=field, **outcome.diagnostic)
        if outcome and outcome.accepted and outcome.case and outcome.assessment:
            updated = outcome.case
            assessment = outcome.assessment
            repair_notice = f"הבנתי את התשובה כך: {outcome.repaired_answer}"
            record_diagnostic("model_repair_accepted", field=field)
        else:
            clarification = outcome.clarification_question if outcome else str(exc)
            if (
                st.session_state.failure_counts[field] >= FINAL_FAILURE_GUIDANCE_THRESHOLD
                and field not in st.session_state.failure_guidance_shown
            ):
                clarification += "\n\n" + build_repeated_failure_guidance(
                    st.session_state.case,
                    field,
                    st.session_state.partial_answers.get(field, ""),
                    st.session_state.failure_counts[field],
                    str(exc),
                )
                st.session_state.failure_guidance_shown.add(field)
            if outcome:
                record_diagnostic("model_repair_requested_clarification", field=field)
            st.session_state.messages.append({"role": "assistant", "content": clarification})
            return
    data = updated.model_dump(mode="python")
    data["facts_confirmed"] = False
    st.session_state.case = CaseData.model_validate(data)
    st.session_state.observations[field] = assessment.model_dump(mode="json")
    st.session_state.partial_answers.pop(field, None)
    st.session_state.failure_counts.pop(field, None)
    st.session_state.failed_answers.pop(field, None)
    st.session_state.failure_guidance_shown.discard(field)
    st.session_state.completed_fields.add(field)
    record_diagnostic(
        "answer_accepted",
        field=field,
        source=assessment.source,
        confidence=assessment.confidence,
    )
    st.session_state.draft_markdown = ""
    if repair_notice:
        st.session_state.messages.append({"role": "assistant", "content": repair_notice})
    advance_interview()


def generate_draft(*, run_user_clarification_review: bool = True) -> bool:
    state = initial_state(st.session_state.case)
    state["generate_requested"] = True
    st.session_state.draft_error = ""
    try:
        result = st.session_state.graph.invoke(
            state,
            {"configurable": {"thread_id": f"draft-{uuid.uuid4()}"}},
        )
    except Exception as exc:
        st.session_state.draft_error = f"יצירת הטיוטה נכשלה ({type(exc).__name__}). נסו שוב; אם התקלה חוזרת, עברו למצב המקומי."
        record_diagnostic("draft_failed", error_type=type(exc).__name__)
        return False
    if result.get("validation_errors"):
        st.session_state.draft_error = "לא ניתן ליצור טיוטה: " + " ".join(result["validation_errors"])
        record_diagnostic("draft_rejected", validation_error_count=len(result["validation_errors"]))
        return False
    st.session_state.draft_markdown = result["draft_markdown"]
    st.session_state.draft_editor = result["draft_markdown"]
    st.session_state.draft_review = result.get("draft_review", {})
    st.session_state.draft_warnings = result.get("warnings", [])
    st.session_state.draft_authority_ids = {
        item["id"] for item in result.get("authorities", []) if item.get("id")
    }
    st.session_state.draft_authorities = result.get("authorities", [])
    st.session_state.model_status = result.get("model_status", st.session_state.model_status)
    for diagnostic in result.get("model_diagnostics", []):
        record_diagnostic("llm_provider_failed", **diagnostic)
    record_diagnostic(
        "draft_generated",
        model_status=st.session_state.model_status,
        warning_count=len(st.session_state.draft_warnings),
    )
    st.session_state.draft_clarifications = []
    st.session_state.draft_clarification_index = 0
    st.session_state.draft_clarification_error = ""
    if run_user_clarification_review and not st.session_state.draft_clarification_skipped:
        client = st.session_state.client
        if not client.available:
            st.session_state.draft_warnings = list(
                dict.fromkeys(
                    st.session_state.draft_warnings
                    + ["לא ניתן היה להפעיל מעבר LLM לשאלות הבהרה; הטיוטה נשענה על בדיקות מקומיות."]
                )
            )
            record_diagnostic("draft_clarification_review_skipped", reason="no_provider_available")
        else:
            try:
                clarification_review = client.review_draft_for_clarification(
                    st.session_state.case,
                    st.session_state.draft_markdown,
                    st.session_state.draft_authority_ids,
                )
                issues = sanitize_draft_clarifications(
                    clarification_review, st.session_state.case, st.session_state.draft_markdown
                )
                st.session_state.draft_clarifications = issues
                record_diagnostic("draft_clarification_review_completed", accepted_issue_count=len(issues))
                if issues:
                    st.session_state.draft_clarification_index = 0
                    st.session_state.messages.append(
                        {"role": "assistant", "content": _draft_clarification_prompt(issues[0])}
                    )
            except Exception as exc:
                st.session_state.draft_warnings = list(
                    dict.fromkeys(
                        st.session_state.draft_warnings
                        + ["מעבר ה־LLM להבהרות לא היה זמין; הטיוטה נבדקה בבדיקות מקומיות בלבד."]
                    )
                )
                record_diagnostic(
                    "draft_clarification_review_failed",
                    **getattr(exc, "diagnostic", {"exception_type": type(exc).__name__}),
                )
    if not st.session_state.draft_clarifications:
        st.session_state.draft_clarification_changed = False
    return True


def _draft_clarification_prompt(issue: dict[str, str]) -> str:
    return (
        "לפני יצירת קובץ ה־PDF, מצאתי אי־התאמה שכדאי לאמת:\n\n"
        f"בנתוני הראיון: ״{issue['case_excerpt']}״\n"
        f"בטיוטה: ״{issue['draft_excerpt']}״\n\n"
        f"{issue['clarification_question']}\n"
        "כתבו ״כן״ אם נתוני הראיון נכונים. אחרת כתבו ״לא״ ואז את הערך המתוקן."
    )


def process_draft_clarification_answer(message: str) -> None:
    issues = st.session_state.draft_clarifications
    index = st.session_state.draft_clarification_index
    issue = issues[index]
    answer = message.strip()
    changed = False
    if not confirmation_is_yes(answer):
        correction = re.sub(
            r"^\s*לא\s*(?:,|[-:])?\s*(?:הערך הנכון הוא|התכוונתי ל|צריך להיות)?\s*",
            "",
            answer,
            count=1,
        ).strip()
        if not correction or confirmation_is_no(answer):
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "כדי לתקן, כתבו ״לא״ ואחריו הערך הנכון או העובדה שצריך לעדכן.",
                }
            )
            return
        field = "evidence_confirmation" if issue["field"] == "evidence" else issue["field"]
        try:
            updated, assessment = apply_chat_answer_with_assessment(
                st.session_state.case, field, correction
            )
        except InterviewAnswerError as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"לא הצלחתי לעדכן את הפרט: {exc}\nכתבו ״כן״ אם נתוני הראיון נכונים, או ״לא״ ואחריו תיקון ברור."}
            )
            return
        data = updated.model_dump(mode="python")
        data["facts_confirmed"] = True
        st.session_state.case = CaseData.model_validate(data)
        st.session_state.observations[field] = assessment.model_dump(mode="json")
        st.session_state.completed_fields.add(field)
        changed = True
        st.session_state.draft_clarification_changed = True
        record_diagnostic("draft_clarification_corrected", field=field)
    else:
        record_diagnostic("draft_clarification_confirmed", field=issue["field"])

    index += 1
    st.session_state.draft_clarification_index = index
    if index < len(issues):
        st.session_state.messages.append(
            {"role": "assistant", "content": _draft_clarification_prompt(issues[index])}
        )
        return

    st.session_state.draft_clarifications = []
    st.session_state.draft_clarification_index = 0
    st.session_state.draft_markdown = ""
    st.session_state.draft_editor = ""
    if changed or st.session_state.draft_clarification_changed:
        st.session_state.draft_clarification_skipped = True
        data = st.session_state.case.model_dump(mode="python")
        data["facts_confirmed"] = False
        st.session_state.case = CaseData.model_validate(data)
        st.session_state.ready = False
        st.session_state.awaiting_confirmation = True
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "התיקון נקלט. נא לאשר מחדש את סיכום העובדות המעודכן לפני יצירת הטיוטה:\n\n"
                + build_confirmation_summary(st.session_state.case),
            }
        )
    else:
        st.session_state.draft_clarification_skipped = True
        st.session_state.messages.append(
            {"role": "assistant", "content": "תודה, נתוני הראיון אושרו. אני יוצר מחדש את הטיוטה המעודכנת."}
        )
        generate_draft(run_user_clarification_review=False)


def save_edited_draft(markdown: str) -> bool:
    """Validate a user revision, review it, and make it the source for the next PDF."""
    revised = markdown.strip()
    st.session_state.draft_edit_error = ""
    issues = validate_draft(
        revised,
        st.session_state.draft_authority_ids,
        st.session_state.case,
    )
    if issues:
        st.session_state.draft_edit_error = "לא ניתן לשמור את השינוי: " + " ".join(issues)
        record_diagnostic("draft_edit_rejected", validation_error_count=len(issues))
        return False

    try:
        review = st.session_state.client.review(
            st.session_state.case,
            revised,
            st.session_state.draft_authority_ids,
        )
        review = _sanitize_draft_review(
            review,
            st.session_state.case,
            st.session_state.draft_authority_ids,
        )
        st.session_state.draft_review = review.model_dump(mode="json")
        if review.verdict == "revise":
            st.session_state.draft_warnings = list(
                dict.fromkeys(
                    st.session_state.draft_warnings
                    + ["בקרת האיכות מצאה נקודות לבדיקה בטיוטה הערוכה; ההערות מוצגות למטה."]
                )
            )
    except Exception as exc:
        st.session_state.draft_warnings = list(
            dict.fromkeys(
                st.session_state.draft_warnings
                + ["השינוי עבר בדיקות מקומיות, אך בדיקת ה־LLM לא הייתה זמינה הפעם."]
            )
        )
        record_diagnostic(
            "draft_edit_review_failed",
            **getattr(exc, "diagnostic", {"exception_type": type(exc).__name__}),
        )

    st.session_state.draft_markdown = revised + "\n"
    st.session_state.draft_warnings = list(
        dict.fromkeys(st.session_state.draft_warnings + draft_cross_field_warnings(revised, st.session_state.case))
    )
    st.session_state.draft_revision_count += 1
    record_diagnostic(
        "draft_edit_saved",
        revision_count=st.session_state.draft_revision_count,
        character_count=len(revised),
        content_logged=False,
    )
    return True


if "messages" not in st.session_state:
    reset_interview()
if "pending_field_confirmation" not in st.session_state:
    st.session_state.pending_field_confirmation = None
if "preview_return_question" not in st.session_state:
    st.session_state.preview_return_question = ""
if "partial_answers" not in st.session_state:
    st.session_state.partial_answers = {}
if "failure_counts" not in st.session_state:
    st.session_state.failure_counts = {}
if "failed_answers" not in st.session_state:
    st.session_state.failed_answers = {}
if "failure_guidance_shown" not in st.session_state:
    st.session_state.failure_guidance_shown = set()
if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = []
if "diagnostic_session_id" not in st.session_state:
    st.session_state.diagnostic_session_id = str(uuid.uuid4())
if "draft_error" not in st.session_state:
    st.session_state.draft_error = ""
if "draft_editor" not in st.session_state:
    st.session_state.draft_editor = st.session_state.get("draft_markdown", "")
if "draft_edit_error" not in st.session_state:
    st.session_state.draft_edit_error = ""
if "draft_authority_ids" not in st.session_state:
    st.session_state.draft_authority_ids = set()
if "draft_authorities" not in st.session_state:
    st.session_state.draft_authorities = []
if "draft_revision_count" not in st.session_state:
    st.session_state.draft_revision_count = 0
if "draft_clarifications" not in st.session_state:
    st.session_state.draft_clarifications = []
if "draft_clarification_index" not in st.session_state:
    st.session_state.draft_clarification_index = 0
if "draft_clarification_error" not in st.session_state:
    st.session_state.draft_clarification_error = ""
if "draft_clarification_skipped" not in st.session_state:
    st.session_state.draft_clarification_skipped = False
if "draft_clarification_changed" not in st.session_state:
    st.session_state.draft_clarification_changed = False
if "consistency_checked" not in st.session_state:
    st.session_state.consistency_checked = False
if "consistency_review_count" not in st.session_state:
    st.session_state.consistency_review_count = 0
if "client" not in st.session_state:
    st.session_state.client = LLMGateway()

st.title("⚖️ סוכן AI להכנת טיוטת תביעה קטנה")
st.caption("כל איסוף המידע מתבצע בשיחה ממוקדת בעברית")
st.markdown(
    '<div class="privacy"><b>הדגמה לימודית בלבד.</b> אין להזין מידע אמיתי. '
    "תשובות לשאלות זיהוי נשמרות במצב המקומי ואינן נשלחות לספקי ה־LLM.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("מצב הראיון")
    completed, total = interview_progress(st.session_state.case, st.session_state.completed_fields)
    st.progress(completed / total, text=f"{completed} מתוך {total} שלבים")
    st.caption(f"מצב מודל: {st.session_state.model_status}")
    if st.button("התחלת תרחיש המזגן", use_container_width=True):
        reset_interview()
        process_user_message(SAMPLE_OPENING)
        st.rerun()
    if st.button("התחלה מחדש", use_container_width=True):
        reset_interview()
        st.rerun()

    if st.session_state.started:
        validation = validate_case(st.session_state.case)
        st.markdown('<div class="summary">', unsafe_allow_html=True)
        st.write(f"**סכום שנקלט:** {validation.claim_total:,.0f} ₪")
        if validation.claim_total > 0:
            st.write(f"**אגרה משוערת:** {validation.filing_fee:,.0f} ₪")
        else:
            st.write("**אגרה משוערת:** תחושב לאחר קליטת סכום")
        st.write(f"**סוג מחלוקת:** {st.session_state.case.category.value}")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("יומן אבחון", expanded=False):
        st.caption("תוכן השיחה, פרטי זיהוי ומפתח API אינם נרשמים. מוצגים סוג שגיאה, שלבי החריגה, קוד HTTP והצעת בדיקה.")
        st.json(st.session_state.diagnostics[-25:])
        diagnostic_json = json.dumps(
            st.session_state.diagnostics,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        st.download_button(
            "הורדת יומן האבחון",
            data=diagnostic_json.encode("utf-8"),
            file_name="interview-diagnostics.json",
            mime="application/json",
            use_container_width=True,
        )
        if PERSISTENT_DIAGNOSTICS_ENABLED:
            st.caption(f"רישום מתמשך מקומי פעיל: {LOG_PATH}")
        else:
            st.caption("רישום מתמשך לקובץ כבוי כברירת מחדל; היומן נשמר בזיכרון ההפעלה בלבד.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.started and not st.session_state.draft_markdown and not st.session_state.draft_clarifications:
    if st.button("📄 הצגת טיוטת הביניים", use_container_width=True):
        append_live_draft_preview_to_chat()
        st.rerun()
if st.session_state.get("preview_return_question"):
    st.info(f"להמשך הראיון — {st.session_state.preview_return_question}")

user_message = st.chat_input(
    "כתבו תשובה לשאלה האחרונה..." if st.session_state.started else "ספרו בקצרה מה קרה...",
    disabled=st.session_state.ready and not bool(st.session_state.draft_clarifications),
)
if user_message:
    process_user_message(user_message)
    st.rerun()

if st.session_state.draft_clarifications:
    current_issue = st.session_state.draft_clarifications[st.session_state.draft_clarification_index]
    st.info(
        f"נדרשת הבהרה {st.session_state.draft_clarification_index + 1} "
        f"מתוך {len(st.session_state.draft_clarifications)} לפני הפקת ה־PDF. "
        "השיבו בצ׳אט: ״כן״ אם נתוני הראיון נכונים, או ״לא״ ואחריו התיקון."
    )

if st.session_state.ready and not st.session_state.draft_markdown and not st.session_state.draft_clarifications:
    if st.session_state.draft_error:
        st.error(st.session_state.draft_error)
    if st.button("יצירת טיוטת כתב תביעה", type="primary", use_container_width=True):
        with st.spinner("מכין טיוטה ובודק אותה..."):
            generate_draft()
        st.rerun()

if st.session_state.draft_markdown and not st.session_state.draft_clarifications:
    markdown = st.session_state.draft_markdown
    st.divider()
    st.subheader("טיוטה מוכנה לעריכה")
    for warning in st.session_state.draft_warnings:
        if "מעבר" in warning and "הבהר" in warning:
            st.warning(warning)
    st.caption(
        "אפשר לערוך את תוכן ה-Markdown למטה. לאחר השמירה יבוצעו בדיקות מחדש, "
        "וקובץ ה-PDF להורדה ייווצר מהגרסה המעודכנת."
    )
    st.text_area(
        "עריכת טיוטת כתב התביעה",
        key="draft_editor",
        height=560,
    )
    if st.button("שמירת שינויים ועדכון ה-PDF", type="primary", use_container_width=True):
        with st.spinner("בודק את השינויים ומעדכן את הטיוטה..."):
            save_edited_draft(st.session_state.draft_editor)
        st.rerun()
    if st.session_state.draft_edit_error:
        st.error(st.session_state.draft_edit_error)
    if st.session_state.draft_revision_count:
        st.success(f"נשמרו {st.session_state.draft_revision_count} תיקונים. ה-PDF מעודכן לגרסה האחרונה.")
    with st.expander("תצוגה מקדימה של הגרסה שתישמר ב-PDF", expanded=True):
        st.markdown(markdown)
    with st.expander("בקרת איכות"):
        review = st.session_state.draft_review
        st.write("סטטוס: " + ("עבר" if review.get("verdict") == "pass" else "נדרשת בדיקה"))
        for warning in st.session_state.draft_warnings:
            st.warning(warning)
        for issue in review.get("revision_instructions", []):
            st.write(f"- {issue}")
    claim_pdf_bytes = markdown_to_pdf_bytes(markdown, title="טיוטת כתב תביעה")
    st.markdown("### מסמך להגשה לאחר בדיקת המשתמש")
    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        "הורדת Markdown",
        data=markdown.encode("utf-8"),
        file_name="small_claim_draft.md",
        mime="text/markdown",
        use_container_width=True,
    )
    download_col2.download_button(
        "הורדת PDF של כתב התביעה",
        data=claim_pdf_bytes,
        file_name="small_claim_court_draft.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    source_report_markdown = compose_source_report_markdown(
        st.session_state.case,
        [AuthorityRecord.model_validate(item) for item in st.session_state.draft_authorities],
        st.session_state.draft_review,
        st.session_state.draft_warnings,
        st.session_state.model_status,
    )
    source_report_pdf = markdown_to_pdf_bytes(
        source_report_markdown,
        title="דו״ח מקורות ובקרת איכות",
    )
    with st.expander("דו״ח מקורות ובקרת איכות של הסוכן - אופציונלי", expanded=False):
        st.warning("דו״ח זה מיועד לשקיפות ולהדגמת מנגנון האחזור ואינו מיועד להגשה לבית המשפט.")
        st.markdown(source_report_markdown)
        report_col1, report_col2 = st.columns(2)
        report_col1.download_button(
            "הורדת דו״ח מקורות כ-Markdown",
            data=source_report_markdown.encode("utf-8"),
            file_name="ai_sources_and_quality_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        report_col2.download_button(
            "הורדת דו״ח מקורות כ-PDF",
            data=source_report_pdf,
            file_name="ai_sources_and_quality_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
