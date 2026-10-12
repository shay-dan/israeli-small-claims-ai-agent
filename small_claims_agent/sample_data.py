from __future__ import annotations

from .courts import apply_jurisdiction
from .models import CaseData


def air_conditioner_sample() -> CaseData:
    case = CaseData.model_validate(
        {
            "claimant": {
                "full_name": "דנה ישראלי (דמות בדיונית)",
                "identifier": "000000000",
                "address": "רחוב הדוגמה 12, תל אביב-יפו",
                "phone": "050-0000000",
                "email": "demo@example.com",
                "legal_form": "individual"
            },
            "defendant": {
                "full_name": "אלפא מוצרי חשמל בע״מ (חברה בדיונית)",
                "identifier": "510000000",
                "address": "רחוב המסחר 8, תל אביב-יפו",
                "phone": "03-0000000",
                "email": "",
                "legal_form": "company"
            },
            "category": "consumer_product",
            "transaction_date": "2026-06-01",
            "incident_date": "2026-06-15",
            "incident_place": "חנות הנתבעת בתל אביב-יפו",
            "product_or_service_details": "מזגן ביתי מדגם Demo-3200",
            "warranty_details": "אחריות יצרן לשנה לפי תעודת האחריות",
            "defendant_role": "מוכר",
            "event_summary": "מזגן שנרכש תמורת 3,200 ₪ הפסיק לפעול כשבועיים לאחר הרכישה, והנתבעת לא תיקנה אותו ולא השיבה את התמורה חרף שלוש פניות.",
            "chronology": [
                "ביום 1.6.2026 רכשה התובעת מהנתבעת מזגן בסכום של 3,200 ₪ וקיבלה קבלה.",
                "ביום 15.6.2026 חדל המזגן לפעול במהלך שימוש רגיל.",
                "בימים 16.6.2026, 20.6.2026 ו-27.6.2026 פנתה התובעת לנתבעת וביקשה תיקון או השבת התמורה.",
                "הנתבעת סירבה לתקן את המזגן או להשיב את התמורה."
            ],
            "prior_attempts": [
                "שלוש פניות טלפוניות ומתועדות בימים 16.6.2026, 20.6.2026 ו-27.6.2026.",
                "דרישה כתובה מיום 28.6.2026 שלא נענתה."
            ],
            "desired_non_monetary_relief": "לחלופין, החלפת המזגן במוצר תקין, לפי שיקול דעת בית המשפט.",
            "primary_remedy": "cancellation",
            "alternative_remedy": "replacement",
            "damage_items": [
                {"label": "השבת מחיר המזגן", "amount": "3200", "kind": "direct"},
                {"label": "בדיקת טכנאי פרטית", "amount": "350", "kind": "consequential"},
                {"label": "טרחה ועוגמת נפש", "amount": "500", "kind": "non_pecuniary"}
            ],
            "evidence": [
                {"name": "קבלה מיום 1.6.2026", "description": "הוכחת הרכישה והמחיר"},
                {"name": "תכתובת עם שירות הלקוחות", "description": "תיעוד הפניות והסירוב"},
                {"name": "חשבונית טכנאי", "description": "הוצאה בסך 350 ₪"}
            ],
            "confirmed_no_more_evidence": True,
            "prior_small_claims_count_this_year": 0,
            "facts_confirmed": True
        }
    )
    return apply_jurisdiction(case)
