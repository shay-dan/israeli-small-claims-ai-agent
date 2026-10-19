from __future__ import annotations

import json
import argparse
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from dotenv import load_dotenv
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from small_claims_agent.courts import apply_jurisdiction, resolve_jurisdiction
from small_claims_agent.drafting import compose_claim_markdown, compose_source_report_markdown
from small_claims_agent.graph import build_graph, initial_state
from small_claims_agent.interview import (
    InterviewAnswerError,
    apply_chat_answer,
    build_live_draft_preview,
    correction_command,
    inferred_completed_fields,
    is_recall_request,
    merge_damage_confirmation,
    merge_partial_answer,
    next_interview_field,
    normalize_opening_keyboard_layout,
    opening_story_is_complete,
    parse_damage_items,
    prior_attempt_fragment_is_relevant,
    question_for,
    seed_case_from_opening,
)
from small_claims_agent.legal import classify_category, retrieve_authorities_for_case
from small_claims_agent.llm import DemoAgentClient, LLMGateway, ModelUnavailable
from small_claims_agent.models import CaseData, ClaimCategory, DamageItem, EvidenceItem, Party, RemedyType
from small_claims_agent.pdf_export import markdown_to_pdf_bytes
from small_claims_agent.privacy import redact_text, safe_case_context
from small_claims_agent.sample_data import air_conditioner_sample
from small_claims_agent.validators import SMALL_CLAIM_LIMIT_ILS, calculate_filing_fee, validate_case, validate_draft


REPORT_DIR = ROOT / "output" / "smoke"
PDF_DIR = REPORT_DIR / "pdf"
REPORT_PATH = ROOT / "output" / "smoke_test_report.md"
JSON_PATH = ROOT / "output" / "smoke_test_results.json"


@dataclass
class Result:
    section: str
    test_id: str
    description: str
    status: str
    expected: str
    actual: str
    duration_ms: int


class SmokeRunner:
    def __init__(self) -> None:
        self.results: list[Result] = []

    def check(
        self,
        section: str,
        test_id: str,
        description: str,
        expected: str,
        operation: Callable[[], Any],
        predicate: Callable[[Any], bool] = bool,
    ) -> Any:
        started = time.perf_counter()
        try:
            value = operation()
            passed = predicate(value)
            status = "PASS" if passed else "FAIL"
            actual = _short(value)
        except Exception as exc:  # every failure belongs in the report
            value = None
            status = "FAIL"
            actual = f"{type(exc).__name__}: {exc}"
        self.results.append(
            Result(
                section=section,
                test_id=test_id,
                description=description,
                status=status,
                expected=expected,
                actual=actual,
                duration_ms=round((time.perf_counter() - started) * 1000),
            )
        )
        return value

    def expected_error(
        self,
        section: str,
        test_id: str,
        description: str,
        expected_substring: str,
        operation: Callable[[], Any],
    ) -> None:
        def invoke() -> str:
            try:
                operation()
            except InterviewAnswerError as exc:
                return str(exc)
            return "NO ERROR"

        self.check(
            section,
            test_id,
            description,
            f"InterviewAnswerError containing {expected_substring!r}",
            invoke,
            lambda value: expected_substring in value,
        )

    def write(self, metadata: dict[str, Any]) -> None:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(
            json.dumps(
                {"metadata": metadata, "results": [asdict(item) for item in self.results]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        total = len(self.results)
        passed = sum(item.status == "PASS" for item in self.results)
        failed = sum(item.status == "FAIL" for item in self.results)
        skipped = sum(item.status == "SKIP" for item in self.results)
        lines = [
            "# Comprehensive Smoke Test Report",
            "",
            f"- Generated: {metadata['generated_at']}",
            f"- Python: `{metadata['python']}`",
            f"- LLM provider configured: `{metadata['llm_provider_configured']}`",
            f"- Total checks: **{total}**",
            f"- Passed: **{passed}**",
            f"- Failed: **{failed}**",
            f"- Skipped: **{skipped}**",
            f"- Overall: **{'PASS' if failed == 0 else 'FAIL'}**",
            "",
            "## Summary by section",
            "",
            "| Section | Passed | Failed | Skipped | Total |",
            "|---|---:|---:|---:|---:|",
        ]
        sections = list(dict.fromkeys(item.section for item in self.results))
        for section in sections:
            rows = [item for item in self.results if item.section == section]
            section_passed = sum(item.status == "PASS" for item in rows)
            section_failed = sum(item.status == "FAIL" for item in rows)
            section_skipped = sum(item.status == "SKIP" for item in rows)
            lines.append(f"| {section} | {section_passed} | {section_failed} | {section_skipped} | {len(rows)} |")
        failures = [item for item in self.results if item.status == "FAIL"]
        lines.extend(["", "## Failures requiring attention", ""])
        if not failures:
            lines.append("No failures detected.")
        for item in failures:
            lines.extend(
                [
                    f"### {item.test_id} — {item.description}",
                    "",
                    f"- Expected: {item.expected}",
                    f"- Actual: {item.actual}",
                    "",
                ]
            )
        lines.extend(["", "## Detailed execution log", ""])
        for section in sections:
            lines.extend([f"### {section}", "", "| ID | Status | Description | Expected | Actual | ms |", "|---|---|---|---|---|---:|"])
            for item in (row for row in self.results if row.section == section):
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            _cell(item.test_id),
                            item.status,
                            _cell(item.description),
                            _cell(item.expected),
                            _cell(item.actual),
                            str(item.duration_ms),
                        )
                    )
                    + " |"
                )
            lines.append("")
        lines.extend(
            [
                "## Artifact inventory",
                "",
                f"- Machine-readable results: `{JSON_PATH.relative_to(ROOT).as_posix()}`",
                f"- Smoke PDFs: `{PDF_DIR.relative_to(ROOT).as_posix()}`",
                "- Rendered PNGs and visual findings are appended by the outer QA runner.",
                "",
            ]
        )
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _short(value: Any, limit: int = 420) -> str:
    if isinstance(value, CaseData):
        value = {
            "category": value.category.value,
            "amount": str(value.claim_total),
            "next": next_interview_field(value, inferred_completed_fields(value)),
        }
    elif hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _complete_case(kind: str) -> CaseData:
    if kind == "product":
        return air_conditioner_sample()
    if kind == "restaurant":
        case = CaseData(
            claimant=Party(
                full_name="דמות בדיקה",
                identifier="123456789",
                address="רחוב הדוגמה 1 ירושלים",
                phone="0501234567",
            ),
            defendant=Party(
                full_name="מסעדת בדיקה בע״מ",
                address="קניון מלחה ירושלים",
                legal_form="company",
            ),
            category=ClaimCategory.SERVICE_CONTRACT,
            transaction_date="01.01.2026",
            incident_date="01.01.2026",
            incident_place="מסעדת בדיקה, קניון מלחה ירושלים",
            product_or_service_details="ארוחה במסעדה",
            defendant_role="נותן שירות",
            event_summary="כרטיס האשראי חויב פעמיים בסכום כולל של 20,000 ₪ בגין אותה ארוחה.",
            prior_attempts=["פניתי במסעדה והנתבע טען שלא בוצע חיוב כפול."],
            damage_items=[DamageItem(label="השבת החיוב הכפול העודף", amount=Decimal("10000"))],
            primary_remedy=RemedyType.REFUND,
            alternative_remedy=RemedyType.NONE,
            evidence=[EvidenceItem(name="פירוט חיובי אשראי")],
            confirmed_no_more_evidence=True,
            facts_confirmed=True,
        )
        return apply_jurisdiction(case)
    if kind == "course":
        case = CaseData(
            claimant=Party(
                full_name="דמות בדיקה",
                identifier="123456789",
                address="רחוב הדוגמה 2 תל אביב",
                phone="0501234567",
            ),
            defendant=Party(
                full_name="COURSES LTD",
                address="רחוב המסחר 8 תל אביב",
                legal_form="company",
            ),
            category=ClaimCategory.SERVICE_CONTRACT,
            transaction_date="01.01.2026",
            incident_date="03.01.2026",
            incident_place="courses.co.il",
            product_or_service_details="קורס ללימודי הייטק",
            defendant_role="נותן שירות",
            event_summary="שילמתי 20,000 ₪ עבור קורס וביקשתי לבטל בתוך התקופה שהוצגה לי, אך הבקשה נדחתה.",
            prior_attempts=["שלחתי הודעה באתר ונאמר לי שלא ניתן לבטל לאחר כניסה לתכנים."],
            damage_items=[DamageItem(label="השבת מחיר הקורס", amount=Decimal("20000"))],
            primary_remedy=RemedyType.CANCELLATION,
            alternative_remedy=RemedyType.NONE,
            evidence=[EvidenceItem(name="קבלה"), EvidenceItem(name="תנאי ההרשמה")],
            confirmed_no_more_evidence=True,
            facts_confirmed=True,
        )
        return apply_jurisdiction(case)
    raise ValueError(kind)


def run_openings(runner: SmokeRunner) -> None:
    cases = [
        ("O01", "קניתי מזגן ב3500 שקל ואחרי שבוע הפסיק לעבוד", True, "3500", "consumer_product"),
        ("O02", "קניתי כדורגל ב10 שקל ואחרי יום התפנצ'ר", True, "10", "consumer_product"),
        ("O03", "קניתי אופניים ב1200 שקל ואחרי יומיים השרשרת נשברה", True, "1200", "consumer_product"),
        ("O04", "שילמתי 20000 שקל על קורס ללימודים בהייטק ואחרי יומיים ביקשתי לבטל וסירבו להחזיר", True, "20000", "service_contract"),
        ("O05", "שילמתי 4500 שקל על סדנת צילום אבל ביקשתי לבטל וסירבו להחזיר", True, "4500", "service_contract"),
        ("O06", "בעל מסעדה עקץ אותי וחייב אותי פעמים בסכום כולל של 20 אלף", True, "20000", "service_contract"),
        ("O07", "קיבלתי שירות וחויבתי פעמיים בלי סיבה ושילמתי עשר אלף שקל", True, "10000", "service_contract"),
        ("O08", "קניתי מקרר והוא התקלקל", False, "0", "consumer_product"),
        ("O09", "קניתי מקרר ב5000 שקל", False, "5000", "consumer_product"),
        ("O10", "שילמתי 500 שקל אבל הייתה תקלה", False, "500", "unsupported"),
        ("O11", "הזמנתי צילום אירוע ב6000 שקל והשירות לא סופק", True, "6000", "service_contract"),
        ("O12", "שכרתי דירה ושילמתי 5000 שקל פיקדון והמשכיר מסרב להחזיר", True, "5000", "landlord_tenant"),
    ]
    for test_id, text, complete, amount, category in cases:
        def operation(text=text) -> dict[str, Any]:
            case = seed_case_from_opening(text)
            return {
                "complete": opening_story_is_complete(text),
                "amount": str(case.claim_total),
                "category": case.category.value,
                "product": case.product_or_service_details,
            }

        runner.check(
            "Opening stories",
            test_id,
            text,
            f"complete={complete}, amount={amount}, category={category}",
            operation,
            lambda value, complete=complete, amount=amount, category=category: value["complete"] == complete
            and value["amount"] == amount
            and value["category"] == category,
        )
    keyboard_cases = [("ebh,h nzdi", "קניתי מזגן", True), ("hello world", "hello world", False)]
    for index, (text, converted, changed) in enumerate(keyboard_cases, start=13):
        runner.check(
            "Opening stories",
            f"O{index:02d}",
            f"Keyboard-layout conversion: {text}",
            f"{converted}, changed={changed}",
            lambda text=text: normalize_opening_keyboard_layout(text),
            lambda value, converted=converted, changed=changed: value == (converted, changed),
        )


def run_dates(runner: SmokeRunner) -> None:
    valid = [
        ("D01", "1.1.26", "01.01.2026"),
        ("D02", "01/01/26", "01.01.2026"),
        ("D03", "01.01.2026", "01.01.2026"),
        ("D04", "2026-01-01", "01.01.2026"),
        ("D05", "01012026", "01.01.2026"),
        ("D06", "010126", "01.01.2026"),
        ("D07", "29.2.24", "29.02.2024"),
    ]
    for test_id, answer, expected in valid:
        runner.check(
            "Dates",
            test_id,
            f"Accept {answer}",
            expected,
            lambda answer=answer: apply_chat_answer(CaseData(), "transaction_date", answer).transaction_date,
            lambda value, expected=expected: value == expected,
        )
    invalid = [
        ("D08", "15.15.20", "החודש 15"),
        ("D09", "31.2.26", "אינו קיים"),
        ("D10", "29.2.25", "אינו קיים"),
        ("D11", "1.1.30", "בעתיד"),
        ("D12", "אתמול", "תאריך עמום"),
        ("D13", "0101", "מבנה התאריך"),
    ]
    for test_id, answer, message in invalid:
        runner.expected_error(
            "Dates",
            test_id,
            f"Reject {answer}",
            message,
            lambda answer=answer: apply_chat_answer(CaseData(), "transaction_date", answer),
        )
    case = apply_chat_answer(CaseData(), "transaction_date", "10.1.26")
    runner.expected_error(
        "Dates",
        "D14",
        "Incident before transaction",
        "מוקדם מתאריך העסקה",
        lambda: apply_chat_answer(case, "incident_date", "5.1.26"),
    )
    runner.check(
        "Dates",
        "D15",
        "Incident same day as transaction",
        "accepted",
        lambda: apply_chat_answer(case, "incident_date", "10.1.26").incident_date,
        lambda value: value == "10.01.2026",
    )


def run_places_and_parties(runner: SmokeRunner) -> None:
    accepted = [
        "תל אביב רחוב הארבעה 4",
        "מסעדת יוסי ובניו, קניון מלחה ירושלים",
        'אתר "COURSES"',
        "courses.co.il",
        "https://courses.co.il/register",
        "עסקה טלפונית",
        "ביצעתי את העסקה בטלפון",
        "רכשתי דרך המוקד",
    ]
    for index, place in enumerate(accepted, start=1):
        runner.check(
            "Places and parties",
            f"P{index:02d}",
            f"Accept place: {place}",
            "accepted unchanged",
            lambda place=place: apply_chat_answer(CaseData(), "incident_place", place).incident_place,
            lambda value, place=place: value == place,
        )
    rejected = [
        ("בחנות", "כללי מדי"),
        ("באינטרנט", "שם החברה"),
        ("תל אבי רחוב הארבעה 4", "תל אביב"),
        ("njkfdsa fds", "מקלדת האנגלית"),
    ]
    for index, (place, message) in enumerate(rejected, start=9):
        runner.expected_error(
            "Places and parties",
            f"P{index:02d}",
            f"Reject place: {place}",
            message,
            lambda place=place: apply_chat_answer(CaseData(), "incident_place", place),
        )
    physical = CaseData(incident_place="תל אביב רחוב הארבעה 4")
    runner.check(
        "Places and parties",
        "P13",
        "Reuse physical transaction address",
        "full prior address",
        lambda: apply_chat_answer(physical, "defendant.address", "אותה").defendant.address,
        lambda value: value == physical.incident_place,
    )
    remote = CaseData(incident_place="courses.co.il")
    runner.expected_error(
        "Places and parties",
        "P14",
        "Do not reuse website as postal address",
        "כתובת הפיזית",
        lambda: apply_chat_answer(remote, "defendant.address", "אותה כתובת"),
    )
    runner.expected_error(
        "Places and parties",
        "P15",
        "Unknown defendant address gives lookup guidance",
        "רשם החברות",
        lambda: apply_chat_answer(remote, "defendant.address", "לא יודע"),
    )
    party_checks = [
        ("P16", "defendant.full_name", "COURSES", "COURSES"),
        ("P17", "defendant.identifier", "לא", ""),
        ("P18", "defendant.phone", "לא ידוע", ""),
        ("P19", "claimant.phone", "0545363636", "0545363636"),
        ("P20", "claimant.email", "אין", ""),
    ]
    for test_id, field, answer, expected in party_checks:
        runner.check(
            "Places and parties",
            test_id,
            f"Normalize {field}: {answer}",
            repr(expected),
            lambda field=field, answer=answer: _party_value(apply_chat_answer(CaseData(), field, answer), field),
            lambda value, expected=expected: value == expected,
        )
    invalid_party = [
        ("P21", "defendant.full_name", "njxbh huxh", "מקלדת האנגלית"),
        ("P22", "claimant.identifier", "545", "9 ספרות"),
        ("P23", "claimant.phone", "054512346", "10 ספרות"),
        ("P24", "claimant.email", "abc", "אינה תקינה"),
    ]
    for test_id, field, answer, message in invalid_party:
        runner.expected_error(
            "Places and parties",
            test_id,
            f"Reject {field}: {answer}",
            message,
            lambda field=field, answer=answer: apply_chat_answer(CaseData(), field, answer),
        )
    restaurant = CaseData(incident_place="מסעדת יוסי ובניו, קניון מלחה ירושלים")
    for index, answer in enumerate(("כתבתי קודם", "אותו דבר"), start=25):
        def require_explicit_name_confirmation(answer=answer):
            try:
                return apply_chat_answer(
                    restaurant, "defendant.full_name", answer
                ).defendant.full_name
            except InterviewAnswerError as exc:
                return str(exc)

        runner.check(
            "Places and parties",
            f"P{index:02d}",
            f"Require confirmation before reusing restaurant name via {answer}",
            "explicitly asks whether the candidate is מסעדת יוסי ובניו",
            require_explicit_name_confirmation,
            lambda value: "מסעדת יוסי ובניו" in value and "האם התכוונתם" in value,
        )


def _party_value(case: CaseData, field: str) -> str:
    side, attr = field.split(".")
    return getattr(getattr(case, side), attr)


def run_contact_attempts(runner: SmokeRunner) -> None:
    accepted = [
        "התקשרתי לחנות ואמרו שאין החזרות",
        "צלצלתי אליהם ואמרו לי שלא יתקנו",
        "חייגתי לחנות והם סירבו להחזיר כסף",
        "שלחתי מייל והם השיבו שלא יזכו אותי",
        "כתבתי להם בוואטסאפ והם טענו שאני אשם",
        "הגעתי למסעדה ואמרתי לו שחייב פעמיים והוא טען שלא",
        "אמרתי לו שהוא חייב אותי כפול והוא טען שלא",
        "פניתי שלוש פעמים בטלפון והם לא ענו",
    ]
    for index, answer in enumerate(accepted, start=1):
        runner.check(
            "Contact attempts",
            f"C{index:02d}",
            answer,
            "accepted as one resolution attempt",
            lambda answer=answer: apply_chat_answer(CaseData(), "prior_attempts", answer).prior_attempts,
            lambda value, answer=answer: value == [answer],
        )
    incomplete = [
        ("C09", "התקשרתי", "התגובה שקיבלתם"),
        ("C10", "הלכתי למסעדה ואמרתי לו שחייב כפול", "התגובה שקיבלתם"),
        ("C11", "הם אמרו שאין החזרות", "אמצעי הפנייה"),
    ]
    for test_id, answer, message in incomplete:
        runner.expected_error(
            "Contact attempts",
            test_id,
            answer,
            message,
            lambda answer=answer: apply_chat_answer(CaseData(), "prior_attempts", answer),
        )
    voucher_case = CaseData(product_or_service_details="מקרר")
    voucher_attempt = "הלכתי לחנות והם הסכימו רק לתת לי שובר על 200 שקל ולא לתקן או להחליף"
    runner.check(
        "Contact attempts",
        "C15",
        "Walk-in contact with a partial voucher offer is accepted",
        "one resolution attempt",
        lambda: apply_chat_answer(voucher_case, "prior_attempts", voucher_attempt).prior_attempts,
        lambda value: value == [voucher_attempt],
    )
    opening_with_attempt = seed_case_from_opening(
        "קניתי מקרר ב4000 שקל והוא התקלקל אחרי יומיים. הלכתי לחנות והם "
        "הסכימו רק לתת לי שובר על 200 שקל ולא לתקן או להחליף"
    )
    runner.check(
        "Contact attempts",
        "C16",
        "Opening contact with a partial remedy can be reused",
        "attempt extracted and reused",
        lambda: apply_chat_answer(opening_with_attempt, "prior_attempts", "השתמש במה שכתבתי").prior_attempts,
        lambda value: len(value) == 1 and "שובר" in value[0] and "הלכתי" in value[0],
    )
    case = CaseData(product_or_service_details="מזגן")
    combined = merge_partial_answer("התקשרתי", "אמרו שלא מחזירים")
    runner.check(
        "Contact attempts",
        "C12",
        "Combine method-only and outcome-only fragments",
        combined,
        lambda: apply_chat_answer(case, "prior_attempts", combined).prior_attempts,
        lambda value: value == [combined],
    )
    runner.check(
        "Contact attempts",
        "C13",
        "Method-only fragment remains relevant for accumulation",
        "True",
        lambda: prior_attempt_fragment_is_relevant("התקשרתי", case),
        lambda value: value is True,
    )
    runner.check(
        "Contact attempts",
        "C14",
        "Sports complaint unrelated to product is rejected",
        "False",
        lambda: prior_attempt_fragment_is_relevant("צעקתי במשחק שהם משחקים גרוע", case),
        lambda value: value is False,
    )


def run_damages_remedies_evidence(runner: SmokeRunner) -> None:
    parsed = [
        ("M01", "מחיר מוצר 3200", [("3200", "direct")]),
        ("M02", "מחיר מוצר 3200, עוגמת נפש 800", [("3200", "direct"), ("800", "non_pecuniary")]),
        ("M03", "נסיעות 50; הפסד יום עבודה 450", [("50", "consequential"), ("450", "consequential")]),
    ]
    for test_id, answer, expected in parsed:
        runner.check(
            "Damages, remedies, evidence",
            test_id,
            f"Parse damages: {answer}",
            str(expected),
            lambda answer=answer: [(str(item.amount), item.kind) for item in parse_damage_items(answer)],
            lambda value, expected=expected: value == expected,
        )
    runner.expected_error(
        "Damages, remedies, evidence",
        "M04",
        "Large non-pecuniary amount pauses for explicit confirmation",
        "כתבו ״כן״ או ״לא״",
        lambda: parse_damage_items("מחיר מוצר 3200 עוגמת נפש 4000", Decimal("3200")),
    )
    runner.check(
        "Damages, remedies, evidence",
        "M05",
        "Natural yes confirms prior non-pecuniary itemization",
        "total 7200",
        lambda: merge_damage_confirmation(["מחיר מוצר 3200 עוגמת נפש 4000"], "כן"),
        lambda value: bool(value)
        and sum((item.amount for item in parse_damage_items(value, Decimal("3200"))), Decimal("0"))
        == Decimal("7200"),
    )
    duplicate = seed_case_from_opening("בעל מסעדה חייב אותי פעמים בסכום כולל של 20 אלף וקיבלתי שירות מזלזל")
    runner.check(
        "Damages, remedies, evidence",
        "M06",
        "Explicit 10000 replaces inferred 20000 duplicate-charge total",
        "10000 with excess-charge label",
        lambda: apply_chat_answer(duplicate, "damage_items", "10000"),
        lambda value: value.claim_total == Decimal("10000") and "החיוב הכפול העודף" in value.damage_items[0].label,
    )
    product = seed_case_from_opening("קניתי מזגן ב3200 שקל והוא התקלקל")
    runner.check(
        "Damages, remedies, evidence",
        "M07",
        "Only product price retains opening amount",
        "3200",
        lambda: apply_chat_answer(product, "damage_items", "רק מחיר המוצר").claim_total,
        lambda value: value == Decimal("3200"),
    )
    remedy_cases = [
        ("M08", "primary_remedy", "כסף", RemedyType.REFUND),
        ("M09", "primary_remedy", "ביטול עסקה והחזר", RemedyType.CANCELLATION),
        ("M10", "primary_remedy", "תיקון", RemedyType.REPAIR),
        ("M11", "primary_remedy", "החלפה", RemedyType.REPLACEMENT),
        ("M12", "primary_remedy", "שיוציאו הודעת התנצלות", RemedyType.OTHER),
        ("M13", "alternative_remedy", "אין", RemedyType.NONE),
    ]
    for test_id, field, answer, expected in remedy_cases:
        runner.check(
            "Damages, remedies, evidence",
            test_id,
            f"Remedy {field}: {answer}",
            expected.value,
            lambda field=field, answer=answer: getattr(apply_chat_answer(CaseData(), field, answer), field),
            lambda value, expected=expected: value == expected,
        )
    case = CaseData(primary_remedy=RemedyType.REFUND)
    runner.expected_error(
        "Damages, remedies, evidence",
        "M14",
        "Alternative remedy cannot duplicate primary",
        "שונה מהסעד העיקרי",
        lambda: apply_chat_answer(case, "alternative_remedy", "כסף"),
    )
    runner.check(
        "Damages, remedies, evidence",
        "M15",
        "Other remedy wording is retained",
        "שיוציאו הודעת התנצלות",
        lambda: apply_chat_answer(CaseData(), "primary_remedy", "שיוציאו הודעת התנצלות").desired_non_monetary_relief,
        lambda value: value == "שיוציאו הודעת התנצלות",
    )
    warranty = apply_chat_answer(CaseData(), "warranty_details", "אחריות לשנה, יש לי קבלה ותעודת אחריות")
    runner.check(
        "Damages, remedies, evidence",
        "M16",
        "Warranty answer seeds receipt and certificate evidence",
        "2 evidence items",
        lambda: [item.name for item in warranty.evidence],
        lambda value: len(value) == 2,
    )
    runner.check(
        "Damages, remedies, evidence",
        "M17",
        "Evidence list splits receipt and warranty",
        "deduplicated evidence",
        lambda: apply_chat_answer(warranty, "evidence_confirmation", "קבלה, תעודת אחריות").evidence,
        lambda value: len(value) == 2,
    )


def run_preview_corrections_privacy(runner: SmokeRunner) -> None:
    case = seed_case_from_opening("קניתי מזגן ב6500 שקל והוא התקלקל אחרי שבוע")
    preview = build_live_draft_preview(case, inferred_completed_fields(case), {"prior_attempts": "התקשרתי"})
    checks = [
        ("R01", "Live preview has draft heading", "טיוטת ביניים"),
        ("R02", "Live preview shows filled product", "מזגן"),
        ("R03", "Live preview shows inferred total", "6,500 ₪"),
        ("R04", "Live preview shows blank fields", "⬜ [טרם מולא]"),
        ("R05", "Live preview shows pending fragment", "נקלט חלקית; טרם אושר"),
    ]
    for test_id, description, token in checks:
        runner.check(
            "Preview, corrections, privacy",
            test_id,
            description,
            token,
            lambda token=token: token in preview,
            lambda value: value is True,
        )
    for index, phrase in enumerate(("מה כן יש לך?", "מה כבר רשום?", "מה נקלט עד עכשיו?"), start=6):
        runner.check(
            "Preview, corrections, privacy",
            f"R{index:02d}",
            f"Recall command: {phrase}",
            "True",
            lambda phrase=phrase: is_recall_request(phrase),
            lambda value: value is True,
        )
    corrections = [
        ("R09", "תקן תאריך עסקה 1.2.26", "transaction_date"),
        ("R10", "תקן מען נתבע רחוב הקרן 40 ירושלים", "defendant.address"),
        ("R11", "תקן סכומים 10000", "damage_items"),
    ]
    for test_id, command, field in corrections:
        runner.check(
            "Preview, corrections, privacy",
            test_id,
            command,
            field,
            lambda command=command: correction_command(command),
            lambda value, field=field: bool(value) and value[0] == field,
        )
    pii_case = _complete_case("course")
    raw = "דמות בדיקה 123456789 0501234567 test@example.com רחוב הדוגמה 2 תל אביב"
    redacted = redact_text(raw, pii_case)
    runner.check(
        "Preview, corrections, privacy",
        "R12",
        "PII redaction masks name, ID, phone, email, address",
        "no raw PII",
        lambda: redacted,
        lambda value: not any(token in value for token in ("דמות בדיקה", "123456789", "0501234567", "test@example.com", "רחוב הדוגמה")),
    )
    safe = json.dumps(safe_case_context(pii_case), ensure_ascii=False)
    runner.check(
        "Preview, corrections, privacy",
        "R13",
        "Safe model context contains no party PII",
        "placeholders only",
        lambda: safe,
        lambda value: "123456789" not in value and "דמות בדיקה" not in value and "[PLAINTIFF_NAME]" in value,
    )


def run_validation_legal_drafts_pdfs(runner: SmokeRunner) -> None:
    runner.check(
        "Validation and fees",
        "V01",
        "Minimum fee",
        "50",
        lambda: calculate_filing_fee(Decimal("3200")),
        lambda value: value == Decimal("50"),
    )
    runner.check(
        "Validation and fees",
        "V02",
        "One-percent fee",
        "200",
        lambda: calculate_filing_fee(Decimal("20000")),
        lambda value: value == Decimal("200"),
    )
    at_limit = _complete_case("product")
    at_limit.damage_items = [DamageItem(label="סעד", amount=SMALL_CLAIM_LIMIT_ILS)]
    runner.check(
        "Validation and fees",
        "V03",
        "Exact small-claim ceiling accepted",
        "no ceiling error",
        lambda: validate_case(at_limit).errors,
        lambda value: not any("עולה על תקרת" in item for item in value),
    )
    over_limit = deepcopy(at_limit)
    over_limit.damage_items = [DamageItem(label="סעד", amount=SMALL_CLAIM_LIMIT_ILS + 1)]
    runner.check(
        "Validation and fees",
        "V04",
        "Above ceiling rejected",
        "ceiling error",
        lambda: validate_case(over_limit).errors,
        lambda value: any("עולה על תקרת" in item for item in value),
    )
    company_claimant = _complete_case("product")
    company_claimant.claimant.legal_form = "company"
    runner.check(
        "Validation and fees",
        "V05",
        "Company claimant rejected",
        "company eligibility error",
        lambda: validate_case(company_claimant).errors,
        lambda value: any("חברה בע״מ" in item for item in value),
    )
    for index, kind in enumerate(("product", "restaurant", "course"), start=6):
        case = _complete_case(kind)
        runner.check(
            "Validation and fees",
            f"V{index:02d}",
            f"Complete {kind} case validates",
            "ready=True",
            lambda case=case: validate_case(case).ready,
            lambda value: value is True,
        )
    legal_cases = [("L01", "product"), ("L02", "restaurant"), ("L03", "course")]
    for test_id, kind in legal_cases:
        case = _complete_case(kind)
        runner.check(
            "Legal retrieval",
            test_id,
            f"Retrieve authority IDs for {kind}",
            "non-empty, unique IDs",
            lambda case=case: [item.id for item in retrieve_authorities_for_case(case)],
            lambda value: bool(value) and len(value) == len(set(value)),
        )
    no_warranty = _complete_case("product")
    no_warranty.warranty_details = ""
    runner.check(
        "Legal retrieval",
        "L04",
        "Electrical-product regulations do not depend on a supplied warranty document",
        "consumer-warranty-regulations present",
        lambda: [item.id for item in retrieve_authorities_for_case(no_warranty)],
        lambda value: "consumer-warranty-regulations" in value,
    )
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for index, kind in enumerate(("product", "restaurant", "course"), start=1):
        case = _complete_case(kind)
        records = retrieve_authorities_for_case(case)
        sections = DemoAgentClient().draft_sections(case, [item.model_dump(mode="json") for item in records])
        markdown = compose_claim_markdown(case, sections, records)
        allowed = {item.id for item in records}
        runner.check(
            "Draft and PDF",
            f"G{index:02d}a",
            f"{kind} draft passes deterministic validator",
            "no issues",
            lambda markdown=markdown, allowed=allowed, case=case: validate_draft(markdown, allowed, case),
            lambda value: value == [],
        )
        expected_total = f"{case.claim_total:,.0f} ₪"
        runner.check(
            "Draft and PDF",
            f"G{index:02d}b",
            f"{kind} draft contains exact total",
            expected_total,
            lambda markdown=markdown: markdown,
            lambda value, expected_total=expected_total: expected_total in value,
        )
        runner.check(
            "Draft and PDF",
            f"G{index:02d}c",
            f"{kind} draft contains mandatory sections and no missing placeholder",
            "all headings, no [טרם מולא]",
            lambda markdown=markdown: markdown,
            lambda value: all(
                heading in value
                for heading in ("## פרטי הצדדים", "## הרקע העובדתי", "## עילת התביעה", "## הסעד המבוקש", "## רשימת נספחים")
            )
            and "[טרם מולא]" not in value,
        )
        runner.check(
            "Draft and PDF",
            f"G{index:02d}c2",
            f"{kind} court draft excludes full source catalogue",
            "brief legal basis only; no source URLs or quotations",
            lambda markdown=markdown: markdown,
            lambda value, records=records: "מקורות משפטיים שנבדקו" not in value
            and "מקור רשמי" not in value
            and all(item.source_url not in value and item.quote_he not in value for item in records),
        )
        pdf_path = PDF_DIR / f"smoke_{kind}.pdf"
        pdf_path.write_bytes(markdown_to_pdf_bytes(markdown))
        reader = PdfReader(str(pdf_path))
        runner.check(
            "Draft and PDF",
            f"G{index:02d}d",
            f"{kind} PDF structure",
            "%PDF, at least one page, >5KB",
            lambda pdf_path=pdf_path, reader=reader: {
                "header": pdf_path.read_bytes()[:4].decode("ascii"),
                "pages": len(reader.pages),
                "bytes": pdf_path.stat().st_size,
            },
            lambda value: value["header"] == "%PDF" and value["pages"] >= 1 and value["bytes"] > 5000,
        )
        source_report = compose_source_report_markdown(
            case,
            records,
            {"verdict": "pass"},
            [],
            "smoke-test-local-drafter",
        )
        runner.check(
            "Draft and PDF",
            f"G{index:02d}e",
            f"{kind} source/QC report is transparent and complete",
            "not-for-filing label and every retrieved record",
            lambda source_report=source_report: source_report,
            lambda value, records=records: "אינו מיועד להגשה לבית המשפט" in value
            and "לא בוצע חיפוש וקטורי" in value
            and all(
                item.id in value and item.quote_he in value and item.source_url in value
                for item in records
            ),
        )
        report_pdf_path = PDF_DIR / f"smoke_{kind}_sources.pdf"
        report_pdf_path.write_bytes(
            markdown_to_pdf_bytes(source_report, title="דו״ח מקורות ובקרת איכות")
        )
        report_reader = PdfReader(str(report_pdf_path))
        runner.check(
            "Draft and PDF",
            f"G{index:02d}f",
            f"{kind} source/QC PDF structure",
            "%PDF, at least one page, >5KB",
            lambda report_pdf_path=report_pdf_path, report_reader=report_reader: {
                "header": report_pdf_path.read_bytes()[:4].decode("ascii"),
                "pages": len(report_reader.pages),
                "bytes": report_pdf_path.stat().st_size,
            },
            lambda value: value["header"] == "%PDF" and value["pages"] >= 1 and value["bytes"] > 5000,
        )


def run_streamlit(runner: SmokeRunner) -> None:
    from streamlit.testing.v1 import AppTest

    app_path = ROOT / "app.py"
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
        app = AppTest.from_file(app_path, default_timeout=30).run()
        runner.check(
            "Streamlit UI",
            "U01",
            "App starts without exception",
            "no exception",
            lambda: list(app.exception),
            lambda value: not value,
        )
        runner.check(
            "Streamlit UI",
            "U02",
            "Chat input is present",
            "one chat input",
            lambda: len(app.chat_input),
            lambda value: value == 1,
        )
        app.chat_input[0].set_value(
            "בעל מסעדה עקץ אותי בכסף והביא לי שירות מזלזל וניצל את התמימות שלי "
            "חייב אותי פעמים בסכום כולל של 20 אלף"
        ).run()
        runner.check(
            "Streamlit UI",
            "U03",
            "Restaurant opening advances to transaction date",
            "transaction_date",
            lambda: app.session_state["awaiting_field"],
            lambda value: value == "transaction_date",
        )
        preview_button = next((button for button in app.button if button.label == "📄 הצגת טיוטת הביניים"), None)
        runner.check(
            "Streamlit UI",
            "U04",
            "Live draft button is in main app",
            "button present",
            lambda: preview_button is not None,
            lambda value: value is True,
        )
        if preview_button:
            preview_button.click().run()
            content = "\n".join(
                item["content"] for item in app.session_state["messages"][-3:]
            )
            runner.check(
                "Streamlit UI",
                "U05",
                "Live preview contains filled and empty fields",
                "20,000 and blank marker",
                lambda content=content: content,
                lambda value: "20,000 ₪" in value and "⬜ [טרם מולא]" in value,
            )
        app.chat_input[0].set_value("1.1.30").run()
        runner.check(
            "Streamlit UI",
            "U06",
            "Future date stays on same field with specific message",
            "transaction_date + future warning",
            lambda: {
                "field": app.session_state["awaiting_field"],
                "message": app.session_state["messages"][-1]["content"],
            },
            lambda value: value["field"] == "transaction_date" and "בעתיד" in value["message"],
        )
        app.chat_input[0].set_value("1.1.26").run()
        runner.check(
            "Streamlit UI",
            "U07",
            "Corrected date advances",
            "incident_date",
            lambda: app.session_state["awaiting_field"],
            lambda value: value == "incident_date",
        )


def run_live_gateway(runner: SmokeRunner) -> None:
    load_dotenv(ROOT / ".env")
    client = LLMGateway()
    if not client.available:
        runner.results.append(
            Result("Live LLM gateway", "A00", "Provider configuration", "SKIP", "at least one provider configured", "no provider key available", 0)
        )
        return
    opening = "שילמתי 20000 שקל על קורס ללימודים בהייטק וביקשתי לבטל אחרי יומיים אך סירבו"
    base_case = seed_case_from_opening(opening)
    runner.check(
        "Live LLM gateway",
        "A01",
        "Structured extraction returns schema-valid output",
        "StateExtraction",
        lambda: client.extract(opening, base_case),
        lambda value: value is not None and hasattr(value, "updates"),
    )
    runner.check(
        "Live LLM gateway",
        "A02",
        "Repair call returns schema-valid action without inventing amount",
        "accept or clarify; no invented amount",
        lambda: client.repair_answer(
            "prior_attempts",
            question_for("prior_attempts", base_case),
            ["התקשרתי", "אמרו שלא"],
            base_case,
        ),
        lambda value: value.action in {"accept", "clarify"} and "999" not in value.repaired_answer,
    )
    complete = _complete_case("restaurant")
    runner.check(
        "Live LLM gateway",
        "A03",
        "Consistency judge returns schema-valid verdict",
        "pass or clarify",
        lambda: client.review_interview(complete),
        lambda value: value.verdict in {"pass", "clarify"},
    )
    records = retrieve_authorities_for_case(complete)
    sections = runner.check(
        "Live LLM gateway",
        "A04",
        "Draft sections return structured Hebrew content",
        "non-empty factual background",
        lambda: client.draft_sections(complete, [item.model_dump(mode="json") for item in records]),
        lambda value: bool(value.factual_background) and bool(value.resolution_attempts),
    )
    if sections:
        sections.cause_of_action = DemoAgentClient().draft_sections(
            complete, [item.model_dump(mode="json") for item in records]
        ).cause_of_action
        markdown = compose_claim_markdown(complete, sections, records)
        runner.check(
            "Live LLM gateway",
            "A05",
            "Live narrative preserves explicit 10000 claim total",
            "10,000 ₪ and not 20,000 as claim total",
            lambda: markdown,
            lambda value: bool(re.search(r"סכום התביעה[^\n]*10,000 ₪", value))
            and bool(re.search(r"סה״כ סעד כספי מבוקש[^\n]*10,000 ₪", value)),
        )


def run_pytest(runner: SmokeRunner) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        timeout=180,
    )
    runner.check(
        "Automated regression suite",
        "T01",
        "Run complete pytest suite",
        "exit code 0",
        lambda: {"exit_code": completed.returncode},
        lambda value: value["exit_code"] == 0,
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the deterministic and optional live comprehensive smoke suite.")
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip the nested pytest run when the regression suite was run separately.",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip direct live gateway checks; provider-specific live checks can be run with live_provider_matrix.py.",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    runner = SmokeRunner()
    if args.skip_pytest:
        runner.results.append(
            Result("Automated regression suite", "T01", "Run complete pytest suite", "SKIP", "already run separately", "skipped by CLI", 0)
        )
    else:
        run_pytest(runner)
    run_openings(runner)
    run_dates(runner)
    run_places_and_parties(runner)
    run_contact_attempts(runner)
    run_damages_remedies_evidence(runner)
    run_preview_corrections_privacy(runner)
    run_validation_legal_drafts_pdfs(runner)
    run_streamlit(runner)
    if args.skip_live:
        runner.results.append(
            Result("Live LLM gateway", "A00", "Live provider calls", "SKIP", "live checks requested", "skipped by CLI", 0)
        )
    else:
        run_live_gateway(runner)
    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version.split()[0],
        "llm_provider_configured": any(os.getenv(name) for name in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY")),
    }
    runner.write(metadata)
    passed = sum(item.status == "PASS" for item in runner.results)
    failed = sum(item.status == "FAIL" for item in runner.results)
    skipped = sum(item.status == "SKIP" for item in runner.results)
    print(f"SMOKE total={len(runner.results)} passed={passed} failed={failed} skipped={skipped}")
    print(REPORT_PATH)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
