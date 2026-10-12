from __future__ import annotations

import re
from dataclasses import dataclass

from .models import CaseData


@dataclass(frozen=True)
class JurisdictionResolution:
    court_city: str
    basis: str
    locality: str


# Curated MVP lookup. A locality not listed here is never guessed: the interview
# asks for a clearer locality and the draft remains blocked.
LOCALITY_TO_COURT = {
    "תל אביב-יפו": "תל אביב-יפו",
    "תל אביב": "תל אביב-יפו",
    "רמת גן": "תל אביב-יפו",
    "גבעתיים": "תל אביב-יפו",
    "ירושלים": "ירושלים",
    "חיפה": "חיפה",
    "באר שבע": "באר שבע",
    "ראשון לציון": "ראשון לציון",
    "פתח תקווה": "פתח תקווה",
    "נתניה": "נתניה",
    "כפר סבא": "כפר סבא",
    "חדרה": "חדרה",
    "אשדוד": "אשדוד",
    "אשקלון": "אשקלון",
    "אילת": "אילת",
    "נצרת": "נצרת",
    "עכו": "עכו",
    "טבריה": "טבריה",
    "צפת": "צפת",
    "קריית גת": "קריית גת",
}


ONLINE_PLACE_MARKERS = (
    "אינטרנט",
    "מקוונת",
    "מקוון",
    "אונליין",
    "אתר",
    "באתר",
    "דרך האתר",
    "באפליקציה",
)

TELEPHONE_PLACE_MARKERS = (
    "בטלפון",
    "טלפונית",
    "טלפוני",
    "שיחת טלפון",
    "מוקד טלפוני",
    "דרך המוקד",
)

GENERIC_ONLINE_PLACES = {
    "באינטרנט",
    "באונליין",
    "אונליין",
    "באתר",
    "דרך האתר",
    "עסקה אינטרנטית",
    "עסקה מקוונת",
    "עסקה אינטרנטית באתר",
    "עסקה מקוונת באתר",
}


def is_online_transaction_place(text: str) -> bool:
    """Accept a described remote channel, while rejecting a bare 'online'."""
    normalized = " ".join(text.lower().split()).strip(" .,-")
    if re.search(r"(?:https?://|www\.)?[^\s.]+\.(?:co\.il|org\.il|ac\.il|gov\.il|com|net|org|il)(?:[/\s]|$)", normalized):
        return True
    if normalized in GENERIC_ONLINE_PLACES:
        return False
    if not any(marker in normalized for marker in ONLINE_PLACE_MARKERS):
        return False
    # A channel plus an identifying site/company token is enough. Latin brand
    # names are legitimate here and must not be treated as keyboard gibberish.
    without_channel = re.sub(
        r"\b(?:עסקה|אינטרנטית|מקוונת|מקוון|אונליין|דרך|באתר|אתר|של|החברה|באפליקציה)\b",
        " ",
        normalized,
    ).strip(' \"״׳')
    return len(without_channel) >= 2


def is_vague_online_place(text: str) -> bool:
    normalized = " ".join(text.lower().split()).strip(" .,-")
    return any(marker in normalized for marker in ONLINE_PLACE_MARKERS) and not is_online_transaction_place(normalized)


def is_telephone_transaction_place(text: str) -> bool:
    normalized = " ".join(text.lower().split()).strip(" .,-")
    return any(marker in normalized for marker in TELEPHONE_PLACE_MARKERS)


def is_remote_transaction_place(text: str) -> bool:
    return is_online_transaction_place(text) or is_telephone_transaction_place(text)


def locality_from_text(text: str) -> str:
    normalized = " ".join(text.replace("–", "-").split())
    # Prefer the longest key so "תל אביב-יפו" wins over "תל אביב".
    for locality in sorted(LOCALITY_TO_COURT, key=len, reverse=True):
        if locality in normalized:
            return locality
    return ""


def resolve_jurisdiction(case: CaseData) -> JurisdictionResolution | None:
    transaction_locality = locality_from_text(case.incident_place)
    defendant_locality = locality_from_text(case.defendant.address)
    locality = transaction_locality or defendant_locality
    if not locality:
        return None

    court_city = LOCALITY_TO_COURT[locality]
    reasons: list[str] = []
    if transaction_locality:
        reasons.append(f"העסקה או המעשה נושא התביעה התרחשו ב{transaction_locality}")
    if defendant_locality:
        reasons.append(f"מען הנתבע/ת או מקום העסק נמצא ב{defendant_locality}")
    basis = " וכן ".join(reasons) + f", המצויה בתחום השיפוט של בית המשפט לתביעות קטנות ב{court_city}."
    return JurisdictionResolution(court_city=court_city, basis=basis, locality=locality)


def apply_jurisdiction(case: CaseData) -> CaseData:
    resolved = resolve_jurisdiction(case)
    data = case.model_dump(mode="python")
    if resolved:
        data["court_city"] = resolved.court_city
        data["jurisdiction_basis"] = resolved.basis
    else:
        data["court_city"] = ""
        data["jurisdiction_basis"] = ""
    return CaseData.model_validate(data)
