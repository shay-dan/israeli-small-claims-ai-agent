from __future__ import annotations

import json
from pathlib import Path

from .models import AuthorityRecord, CaseData, ClaimCategory, RemedyType


KB_PATH = Path(__file__).with_name("legal_kb.json")


def load_authorities(path: Path = KB_PATH) -> list[AuthorityRecord]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [AuthorityRecord.model_validate(record) for record in records]


def retrieve_authorities(category: ClaimCategory, path: Path = KB_PATH) -> list[AuthorityRecord]:
    if category == ClaimCategory.UNSUPPORTED:
        return []
    return [record for record in load_authorities(path) if category in record.categories]


def retrieve_authorities_for_case(case: CaseData, path: Path = KB_PATH) -> list[AuthorityRecord]:
    """Return only authorities triggered by the verified facts and requested remedies."""
    records = {record.id: record for record in load_authorities(path)}
    selected: list[str] = []
    if case.category == ClaimCategory.CONSUMER_PRODUCT:
        selected.extend(["sale-law-11", "contracts-remedies-2"])
        described_product = f"{case.product_or_service_details} {case.event_summary}".lower()
        electrical_product = any(
            word in described_product
            for word in (
                "מזגן",
                "מקרר",
                "מכונת כביסה",
                "מייבש כביסה",
                "מדיח",
                "טלוויזיה",
                "מחשב",
                "מחשב נייד",
                "לפטופ",
                "טאבלט",
                "תנור",
                "שואב אבק",
                "טוסטר",
                "קומקום חשמלי",
                "מיקרוגל",
                "בלנדר",
                "מיקסר",
                "מכונת קפה",
                "כיריים",
                "מכשיר חשמלי",
                "מכשיר אלקטרוני",
            )
        )
        transaction_amount = next(
            (item.amount for item in case.damage_items if item.kind == "direct"),
            None,
        )
        if electrical_product and transaction_amount is not None and transaction_amount > 150:
            selected.append("consumer-warranty-regulations")
    elif case.category == ClaimCategory.SERVICE_CONTRACT:
        selected.append("contracts-remedies-2")
    elif case.category == ClaimCategory.LANDLORD_TENANT:
        rental_text = f"{case.product_or_service_details} {case.event_summary}".lower()
        has_residential_premises = any(
            word in rental_text for word in ("דירה", "דירת מגורים", "שכירות למגורים")
        )
        if (
            has_residential_premises
            and any(word in rental_text for word in ("פיקדון", "ערובה"))
            and any(
                phrase in rental_text
                for phrase in ("לא החזיר", "מסרב להחזיר", "סירב להחזיר", "השבת הפיקדון")
            )
        ):
            selected.append("rent-loan-25i-e")
        if any(
            word in rental_text
            for word in ("ליקוי", "פגם", "תיקון", "לתקן", "תיקן", "תיקנה", "תיקנו", "לא תוקן")
        ):
            selected.append("rent-loan-7")
        selected.append("contracts-remedies-2")
    else:
        return []

    if (
        case.primary_remedy in {RemedyType.REFUND, RemedyType.CANCELLATION}
        or case.alternative_remedy in {RemedyType.REFUND, RemedyType.CANCELLATION}
    ):
        selected.append("contracts-remedies-9")
    if any(item.kind in {"consequential", "other"} for item in case.damage_items):
        selected.append("contracts-remedies-10")
    if any(item.kind == "non_pecuniary" for item in case.damage_items):
        selected.append("contracts-remedies-13")
    return [records[record_id] for record_id in dict.fromkeys(selected) if record_id in records]


def classify_category(text: str, current: ClaimCategory = ClaimCategory.UNSUPPORTED) -> ClaimCategory:
    if current != ClaimCategory.UNSUPPORTED:
        return current
    normalized = text.lower()
    if any(
        word in normalized
        for word in (
            "שכירות",
            "משכיר",
            "שוכר",
            "שכר דירה",
            "פיקדון",
            "דירה שכורה",
            "דירה מושכרת",
            "שכרתי דירה",
            "שכרתי את הדירה",
        )
    ):
        return ClaimCategory.LANDLORD_TENANT
    if any(
        word in normalized
        for word in (
            "שירות",
            "ספק",
            "בעל מקצוע",
            "מנוי",
            "קורס",
            "לימוד",
            "הכשרה",
            "סדנ",
            "הדרכה",
            "ייעוץ",
            "תוכן דיגיטלי",
            "מסעדה",
            "ארוחה",
            "מלון",
            "מוסך",
            "מספרה",
            "הובלה",
            "ניקיון",
            "צילום",
        )
    ):
        return ClaimCategory.SERVICE_CONTRACT
    if "שכרתי" in normalized:
        return ClaimCategory.LANDLORD_TENANT
    if any(word in normalized for word in ("מזגן", "מוצר", "קניתי", "רכשתי", "חנות")):
        return ClaimCategory.CONSUMER_PRODUCT
    return ClaimCategory.UNSUPPORTED
