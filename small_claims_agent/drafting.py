from __future__ import annotations

import re
from decimal import Decimal

from .models import AuthorityRecord, CaseData, ClaimCategory, DraftSections, RemedyType
from .validators import calculate_filing_fee


def money(value: Decimal) -> str:
    return f"{value:,.0f} ₪"


def approved_case_description(case: CaseData) -> str:
    """Return the confirmed opening account without conversational litigation requests."""
    description = " ".join(case.event_summary.split()).strip(" \"'״׳")
    description = re.sub(
        r"(?:[,.]\s*)?(?:אני|אנחנו)\s+(?:רוצה|רוצים|מבקש|מבקשת|מבקשים)\s+לתבוע(?:\s+אות\w*)?[.!]?\s*$",
        "",
        description,
    ).strip(" ,.;")
    return description


def _definite_subject(product: str) -> str:
    """Return a restrained Hebrew subject without changing the supplied product name."""
    product = product.strip()
    if not product:
        return "המוצר או השירות"
    if product.startswith("מכונת "):
        return "מכונת ה" + product[len("מכונת "):]
    return product if product.startswith("ה") else f"ה{product}"


def _time_after_event(text: str) -> str:
    """Extract a user-supplied relative time without changing its factual wording."""
    match = re.search(
        r"(?P<connector>אחרי|לאחר|כעבור)\s+"
        r"(?P<duration>(?:פחות\s+מ|כ-?)?(?:\d+|יום|יומיים|שלושה?|שלוש|שבוע(?:יים)?|חודש(?:יים)?)"
        r"(?:\s+(?:ימים|שבועות|חודשים))?)",
        text,
    )
    return f" {match.group('connector')} {match.group('duration').strip()}" if match else ""


def _formal_first_person_defect(text: str, product: str) -> str:
    """Formalize common defect statements without copying first-person prose."""
    if not re.search(
        r"(?:^|\s)(?:אני|קניתי|רכשתי|שילמתי|פתחתי|ראיתי|גיליתי)(?=\s|$|[,.;])",
        text,
    ):
        return ""
    subject = _definite_subject(product)
    timing = _time_after_event(text)
    if re.search(r"התקלקל|התקלקלה", text):
        return f"{subject} התקלקל{timing}."
    if re.search(r"(?:הפסיק|הפסיקה|חדל|חדלה)\s+(?:לעבוד|לפעול)|לא\s+(?:עובד|פועל)", text):
        return f"{subject} הפסיק לפעול{timing}."
    if re.search(r"התפוצץ|התפוצצה|התפוצצו", text):
        if re.search(r"התפוצצה", text):
            return f"{subject} התפוצצה{timing}."
        if re.search(r"התפוצצו", text):
            return f"{subject} התפוצצו{timing}."
        return f"{subject} התפוצץ{timing}."
    if re.search(r"התפנצ['׳]ר", text):
        return f"{subject} התפנצ׳ר{timing}."
    if re.search(r"(?:שבור|שבורה|נשבר|פגום|פגומה|סדוק|סדוקה|קרוע|קרועה)", text):
        return f"{subject} נמסר כשהוא פגום."
    return ""


def formal_event_fact(case: CaseData) -> str:
    """Turn the central user-supplied incident into restrained third-person prose."""
    raw = approved_case_description(case)
    product = case.product_or_service_details or "המוצר או השירות"

    discovered = re.search(
        r"(?:פתחתי[^,.;]{0,80}?(?:וראיתי|וגיליתי)|ראיתי|גיליתי|התברר)\s+ש(.+?)"
        r"(?=\s*,?\s*(?:ניסיתי|פניתי|ביקשתי|דרשתי)|[.;]|$)",
        raw,
    )
    if discovered:
        detail = discovered.group(1).strip(" ,.;")
        detail = re.sub(r"^המסך\s+של\s+ה", "מסך ה", detail)
        return f"בעת פתיחת המוצר או בדיקתו התברר כי {detail}."

    if re.search(r"(?:חויב|חוייב|חייב).{0,30}(?:פעמיים|כפול)", raw):
        return "לטענת התובע/ת, נגבה ממנו/ה חיוב כפול בגין אותה עסקה."
    if re.search(r"(?:פיקדון).{0,50}(?:לא\s+החזיר|מסרב\s+להחזיר|סירב\s+להחזיר)", raw):
        return "לאחר סיום השכירות, המשכיר לא השיב לתובע/ת את הפיקדון ששולם."
    if re.search(r"(?:ביקשתי|ביקשה|ביקשו).{0,25}(?:לבטל|ביטול)", raw) or "התחרטתי" in raw:
        return "התובע/ת ביקש/ה לבטל את העסקה, אך בקשת הביטול נדחתה."
    if re.search(r"(?:לא\s+סופק|לא\s+בוצע|לא\s+(?:הגיע|הגיעו|הופיע|הופיעו|התייצב|התייצבו))", raw):
        return f"לטענת התובע/ת, {product} לא סופק או לא בוצע בהתאם למוסכם."

    formal_first_person = _formal_first_person_defect(raw, product)
    if formal_first_person:
        return f"לפי תיאור התובע/ת, {formal_first_person}"

    defect_text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", raw)
    defect_clause = re.search(
        r"([^,.;]{0,90}(?:התקלקל|התקלקלה|לא\s+עובד|לא\s+פועל|הפסיק\s+לעבוד|"
        r"הפסיקה\s+לעבוד|הפסיק\s+לפעול|הפסיקה\s+לפעול|חדל\s+לפעול|חדלה\s+לפעול|"
        r"נשבר|שבור(?:ה|ים|ות)?|סדוק(?:ה|ים|ות)?|"
        r"פג(?:ום|ומה|ומים|ומות)|קרוע(?:ה|ים|ות)?|התפוצץ|התפוצצה|התפוצצו|התפנצ['׳]ר)[^,.;]{0,60})",
        defect_text,
    )
    if defect_clause:
        detail = defect_clause.group(1).strip(" ,.;")
        detail = re.sub(r"\s+שנרכש\s+תמורת\s+[\d,]+\s*(?:₪|שקל(?:ים)?)\s*", " ", detail)
        detail = re.sub(
            r"^([א-ת][א-ת״׳'\-]*)\s+(?=(?:הפסיק|הפסיקה|חדל|חדלה|נשבר|התפוצץ|התפוצצה|התפוצצו|התפנצ))",
            lambda match: (match.group(1) if match.group(1).startswith("ה") else "ה" + match.group(1)) + " ",
            detail,
        )
        if re.search(r"(?:^|\s)(?:אני|קניתי|רכשתי|שילמתי|פתחתי|ראיתי|גיליתי)(?:\s|$)", detail):
            detail = re.sub(r"^.*?(?=\b(?:ואחרי|אחרי|לאחר)\b)", "", detail)
        detail = re.sub(r"^(?:ואחרי|אחרי|לאחר)\s+", "לאחר ", detail)
        detail = re.sub(r"\b(?:הוא|היא)\b", product, detail, count=1)
        if re.search(r"(?:^|\s)(?:אני|קניתי|רכשתי|שילמתי|פתחתי|ראיתי|גיליתי)(?:\s|$)", detail):
            if re.search(r"התפוצץ|התפוצצה|התפוצצו", detail):
                detail = f"{_definite_subject(product)} התפוצצה במהלך השימוש"
            elif re.search(r"התפנצ['׳]ר", detail):
                detail = f"{product} התפנצ׳ר במהלך השימוש"
            elif re.search(
                r"(?:הפסיק|הפסיקה|חדל|חדלה)\s+(?:לעבוד|לפעול)|לא\s+(?:עובד|פועל)",
                detail,
            ):
                detail = f"{product} הפסיק לפעול"
            elif re.search(r"(?:שבור|שבורה|נשבר|פגום|פגומה|סדוק|סדוקה)", detail):
                detail = f"{product} נמסר כשהוא פגום"
            else:
                return f"לפי תיאור התובע/ת, התגלה פגם ב{product}."
        chronology_text = " ".join(case.chronology)
        if "שימוש רגיל" in chronology_text and "שימוש רגיל" not in detail:
            detail = detail.rstrip(" ,.;") + " במהלך שימוש רגיל"
        return f"לפי תיאור התובע/ת, {detail}."

    return f"לפי תיאור התובע/ת, התגלעה מחלוקת בנוגע ל{product} ולאופן קיום העסקה."


def formal_case_description(case: CaseData) -> str:
    """Build a concise court-style account without copying conversational prose."""
    product = case.product_or_service_details or "המוצר או השירות"
    if case.category == ClaimCategory.LANDLORD_TENANT:
        opening = f"התובע/ת התקשר/ה עם הנתבע/ת בעסקת שכירות בנוגע ל{product}."
    elif case.category == ClaimCategory.SERVICE_CONTRACT:
        opening = f"התובע/ת התקשר/ה עם הנתבע/ת בעסקה לקבלת {product}."
    else:
        opening = f"התובע/ת רכש/ה מהנתבע/ת {product}."
    return f"{opening} {formal_event_fact(case)}"


def formalize_resolution_attempt(attempt: str, case: CaseData) -> str:
    """Formalize common first-person contact descriptions without adding facts."""
    text = " ".join(attempt.split()).strip(" .")
    if not re.search(r"(?:פניתי|התקשרתי|צלצלתי|חייגתי|שלחתי|כתבתי|הגעתי|באתי|הלכתי|חזרתי|ניסיתי)", text):
        return text + ("" if text.endswith(".") else ".")

    frequency = r"(?:פעם|פעמיים|(?:אחת|אחד|שתיים|שתי|שניים|שלוש|שלושה|ארבע|ארבעה|חמש|חמישה|שש|שישה|שבע|שבעה|שמונה|תשע|תשעה|עשר|עשרה|\d+)\s+פעמים)"

    def nearby_frequency(channel_pattern: str) -> str:
        after = re.search(
            rf"(?:{channel_pattern})(?:\s+(?:ל[א-ת]+|ב[א-ת]+|אל[א-ת]*)){{0,2}}\s+(?P<count>{frequency})",
            text,
        )
        if after:
            return after.group("count")
        before = re.search(rf"(?P<count>{frequency})\s+(?:ב)?(?:{channel_pattern})", text)
        return before.group("count") if before else ""

    channels: list[str] = []
    if re.search(r"(?:מייל|דוא[״\"']?ל)", text):
        count = nearby_frequency(r"מייל|דוא[״\"']?ל")
        channels.append(f"{count + ' ' if count else ''}בדוא״ל")
    if re.search(r"(?:התקשרתי|צלצלתי|חייגתי|טלפנתי|בטלפון|טלפון)", text):
        count = nearby_frequency(r"התקשרתי|צלצלתי|חייגתי|טלפנתי|בטלפון|טלפון")
        channels.append(f"{count + ' ' if count else ''}בטלפון")
    if re.search(r"(?:וואטסאפ|ווטסאפ|מסרון|הודעה)", text):
        count = nearby_frequency(r"וואטסאפ|ווטסאפ|מסרון|הודעה")
        channels.append(f"{count + ' ' if count else ''}בהודעה")
    if re.search(r"(?:הגעתי|באתי|ניגשתי|הלכתי|חזרתי|הגעה|פיזית)(?:\s+(?:לחנות|לסניף|לשם))?", text):
        count = nearby_frequency(r"הגעתי|באתי|ניגשתי|הלכתי|חזרתי|הגעה|פיזית")
        channels.append(f"{count + ' ' if count else ''}באופן פיזי")

    if channels:
        method = "התובע/ת פנה/פנתה לנתבע/ת " + " וכן ".join(channels)
    else:
        method = "התובע/ת פנה/פנתה לנתבע/ת"

    request = ""
    if re.search(r"(?:להחזיר|החזר|הכסף\s+חזרה|זיכוי)", text):
        product = case.product_or_service_details or "המוצר"
        request = f" וביקש/ה להשיב את המוצר ({product}) ולקבל החזר כספי"
    elif re.search(r"(?:לתקן|תיקון)", text):
        request = f" וביקש/ה לתקן את {case.product_or_service_details or 'המוצר או השירות'}"

    outcome_parts: list[str] = []
    if re.search(r"(?:אין\s+החזר|סירב|סירבו|לא\s+מחזיר|לא\s+מוכנ)", text):
        outcome_parts.append("הבקשה נדחתה")
    if re.search(r"גירש|הוציאו אותי|סילקו אותי", text):
        outcome_parts.append("התובע/ת גורש/ה מן המקום במהלך הפנייה")
    if re.search(r"(?:שאני|שהתובע).{0,15}(?:שבר|אשם|גרם)", text):
        outcome_parts.append("הנתבע/ת טען/ה כי התובע/ת אחראי/ת לנזק")
    if not outcome_parts:
        reported = re.search(r"(?:^|\s)ו?(?:הם\s+)?אמרו(?:\s+לי)?\s+ש(.+)$", text)
        if reported:
            reported_text = reported.group(1).strip(" ,.;")
            if reported_text and not re.search(r"(?:^|\s)(?:אני|אנחנו|שלי|שלנו)(?=\s|$)", reported_text):
                outcome_parts.append(f"בתגובה נמסר כי {reported_text}")
    outcome = "; " + " וכן ".join(outcome_parts) if outcome_parts else ""
    return f"{method}{request}{outcome}."


def controlled_relief_explanation(case: CaseData) -> str:
    """Describe requested relief from state only, without model-added amounts."""
    labels = {
        RemedyType.UNDECIDED: "סעד שטרם הוגדר",
        RemedyType.NONE: "ללא סעד נוסף",
        RemedyType.REFUND: "השבת התמורה הכספית",
        RemedyType.CANCELLATION: "ביטול העסקה והשבת התמורה",
        RemedyType.REPAIR: "תיקון המוצר או השירות",
        RemedyType.REPLACEMENT: "החלפת המוצר במוצר תקין",
        RemedyType.OTHER: case.desired_non_monetary_relief or "סעד אחר שטרם פורט",
    }
    parts = [f"כסעד עיקרי, מבוקש {labels[case.primary_remedy]}."]
    if case.alternative_remedy not in {RemedyType.NONE, RemedyType.UNDECIDED}:
        parts.append(f"לחלופין בלבד, מבוקש {labels[case.alternative_remedy]}.")
    parts.append(
        f"הסכום הכספי הכולל המבוקש הוא {money(case.claim_total)}, בהתאם לרכיבים המפורטים להלן. "
        "סכום זה כולל את כל רכיבי ההשבה, הפיצוי וההוצאות ואינו מתווסף אליהם כסכום נוסף."
    )
    return " ".join(parts)


def compose_claim_markdown(
    case: CaseData,
    sections: DraftSections,
    authorities: list[AuthorityRecord],
) -> str:
    defendant_id = case.defendant.identifier or "לא נמסר"
    lines = [
        f"# בית המשפט לתביעות קטנות ב{case.court_city}",
        "",
        "## כתב תביעה",
        "",
        "> טיוטה לבדיקה ולעריכת המשתמש בלבד - אינה ייעוץ משפטי ואינה מסמך שהוגש לבית המשפט.",
        "",
        "## פרטי הצדדים",
        "",
        f"**התובע/ת:** {case.claimant.full_name}, ת״ז {case.claimant.identifier}",
        f"כתובת: {case.claimant.address} | טלפון: {case.claimant.phone} | דוא״ל: {case.claimant.email or 'לא נמסר'}",
        "",
        f"**הנתבע/ת:** {case.defendant.full_name}, מזהה {defendant_id}",
        f"כתובת למסירה: {case.defendant.address} | טלפון: {case.defendant.phone or 'לא נמסר'}",
        "",
        f"**סכום התביעה:** {money(case.claim_total)}",
        f"**אגרה משוערת (נפרדת מסכום התביעה):** {money(calculate_filing_fee(case.claim_total))}",
        "",
        "## סמכות מקומית",
        "",
        f"לבית המשפט הנכבד סמכות מקומית מאחר ש{case.jurisdiction_basis}",
        "",
        "## הרקע העובדתי",
        "",
    ]
    case_description = formal_case_description(case)
    if case_description:
        lines.extend(
            [
                "### תיאור המקרה",
                "",
                case_description,
                "",
                "### השתלשלות האירועים",
                "",
            ]
        )
    lines.extend(f"{index}. {paragraph}" for index, paragraph in enumerate(sections.factual_background, 1))
    if case.warranty_details:
        warranty_text = case.warranty_details.strip().rstrip(".")
        lines.extend(
            [
                "",
                "## אחריות ושירות לאחר מכירה",
                "",
                f"לפי המידע והמסמכים שנמסרו בנוגע לאחריות: {warranty_text}.",
            ]
        )
    lines.extend(["", "## עילת התביעה", ""])
    lines.extend(f"- {paragraph}" for paragraph in sections.cause_of_action)

    lines.extend(["", "## ניסיונות ליישוב המחלוקת", ""])
    lines.extend(f"- {attempt}" for attempt in sections.resolution_attempts)
    lines.extend(
        ["", "## הסעד המבוקש", "", controlled_relief_explanation(case), "", "### רכיבים כספיים", ""]
    )
    for item in case.damage_items:
        lines.append(f"- {item.label}: {money(item.amount)}")
    lines.extend(["", f"**סה״כ סעד כספי מבוקש: {money(case.claim_total)}**", "", "## רשימת נספחים", ""])
    if case.evidence:
        lines.extend(
            f"- נספח {index}: {item.name}" + (f" - {item.description}" if item.description else "")
            for index, item in enumerate(case.evidence, 1)
        )
    else:
        lines.append("- התובע אישר כי אין בידיו נספחים נוספים בשלב זה.")

    if case.witnesses:
        lines.extend(["", "## עדים", ""])
        lines.extend(
            f"- {w.full_name}, טלפון {w.phone or 'לא נמסר'}, דוא״ל {w.email or 'לא נמסר'}"
            for w in case.witnesses
        )

    lines.extend(
        [
            "",
            "## הצהרה",
            "",
            "אני מצהיר/ה כי העובדות המפורטות בטיוטה זו נכונות לפי מיטב ידיעתי.",
            "",
            "תאריך: ________________    חתימה: ________________",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def compose_source_report_markdown(
    case: CaseData,
    authorities: list[AuthorityRecord],
    draft_review: dict | None = None,
    warnings: list[str] | None = None,
    model_status: str = "",
) -> str:
    """Build a transparent RAG/QC report that is separate from the court filing."""
    review = draft_review or {}
    warning_items = warnings or []
    verdict = review.get("verdict", "not_available")
    verdict_label = {
        "pass": "עבר",
        "revise": "נדרשת בדיקה",
        "not_available": "לא זמין",
    }.get(verdict, verdict)
    deterministic_review = model_status.startswith("deterministic-demo")
    lines = [
        "# דו״ח מקורות ובקרת איכות של סוכן AI",
        "",
        "> מסמך נלווה לצורכי שקיפות ובקרה בלבד. אינו חלק מכתב התביעה ואינו מיועד להגשה לבית המשפט.",
        "",
        "## תקציר התיק ששימש לאחזור",
        "",
        f"- **סוג מחלוקת:** {case.category.value}",
        f"- **מוצר או שירות:** {case.product_or_service_details}",
        f"- **סכום תביעה:** {money(case.claim_total)}",
        f"- **סעד עיקרי:** {case.primary_remedy.value}",
        f"- **מצב יצירת הטיוטה:** {model_status or 'לא נמסר'}",
        "",
        "## שיטת האחזור",
        "",
        "המערכת השתמשה באחזור מובנה ומבוסס-כללים מתוך מאגר משפטי סגור. "
        "הבחירה נעשתה לפי סוג המחלוקת, מאפייני המוצר או השירות, הסעדים ורכיבי הנזק. "
        "לא בוצע חיפוש וקטורי או חיפוש חי במאגר משפטי חיצוני.",
        "",
        "## מקורות שאוחזרו",
        "",
    ]
    if authorities:
        for index, authority in enumerate(authorities, 1):
            source_label = {
                "official_primary": "מקור חקיקה רשמי",
                "official_guidance": "הסבר ממשלתי רשמי",
                "public_consolidation": "נוסח מאוחד ציבורי (אינו מקור רשמי)",
            }[authority.source_kind]
            excerpt_label = "ציטוט שמור" if authority.excerpt_kind == "quote" else "תמצית שמורה"
            lines.extend(
                [
                    f"### מקור {index}: {authority.law_name}, סעיף {authority.section}",
                    "",
                    f"- **מזהה במאגר:** `{authority.id}`",
                    f"- **קטגוריות:** {', '.join(item.value for item in authority.categories)}",
                    f"- **תאריך תחילה הרשום במאגר:** {authority.effective_date}",
                    f"- **נבדק לאחרונה במאגר:** {authority.verified_at}",
                    f"- **סוג מקור:** {source_label}",
                    f"- **{excerpt_label}:** {authority.quote_he}",
                    f"- **קישור למקור:** {authority.source_url}",
                    *(
                        [f"- **קישור אימות רשמי:** {authority.verification_url}"]
                        if authority.verification_url
                        else []
                    ),
                    "",
                ]
            )
    else:
        lines.extend(["לא אותרו מקורות משפטיים מתאימים במאגר המובנה.", ""])

    lines.extend(
        [
        "## בקרת איכות",
        "",
        (
            "בדוגמה הדטרמיניסטית הסטטוס שלהלן התקבל מאימות מקומי מבוסס-כללים, "
            "ולא מקריאה למודל שפה."
            if deterministic_review
            else "הסטטוס שלהלן משלב אימות מקומי ובקרת מודל בהתאם לזמינות המודל."
        ),
        "",
        f"- **סטטוס:** {verdict_label}",
        ]
    )
    review_groups = (
        ("בעיות עקביות עובדתית", review.get("factual_consistency_issues", [])),
        ("טענות שאינן נתמכות", review.get("unsupported_assertions", [])),
        ("בעיות בהירות", review.get("clarity_issues", [])),
        ("הוראות תיקון", review.get("revision_instructions", [])),
        ("אזהרות מערכת", warning_items),
    )
    has_review_items = False
    for heading, items in review_groups:
        if not items:
            continue
        has_review_items = True
        lines.extend(["", f"### {heading}", ""])
        lines.extend(f"- {item}" for item in items)
    if not has_review_items:
        lines.extend(["", "לא נרשמו הערות או אזהרות נוספות בבקרת האיכות."])

    lines.extend(
        [
            "",
            "## מגבלות",
            "",
            "- המאגר הוא מאגר הדגמה סגור ואינו מתעדכן אוטומטית.",
            "- תמצית שמורה אינה ציטוט מילולי. סוג המקור וסוג הקטע מסומנים ליד כל רשומה.",
            "- אחזור מקור אינו קביעה שהמקור חל על המקרה או שהתביעה מוצדקת.",
            "- יש לבדוק את נוסח הדין העדכני ואת התאמת הטיוטה לפני שימוש או הגשה.",
        ]
    )
    return "\n".join(lines).strip() + "\n"
