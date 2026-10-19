from __future__ import annotations

import re
from dataclasses import dataclass

from .models import CaseData


@dataclass(frozen=True)
class JurisdictionResolution:
    court_city: str
    basis: str
    locality: str


@dataclass(frozen=True)
class ParsedAddress:
    street: str = ""
    house_number: str = ""
    locality: str = ""


# Curated MVP lookup. A locality not listed here is never guessed: the interview
# asks for a clearer locality and the draft remains blocked.
LOCALITY_TO_COURT = {
    "תל אביב-יפו": "תל אביב-יפו",
    "תל אביב": "תל אביב-יפו",
    "חולון": "תל אביב-יפו",
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


def _locality_match(text: str) -> tuple[str, int, int] | None:
    normalized = " ".join(text.replace("–", "-").split())
    # Match whole locality names, not substrings (e.g. אילת inside a street
    # name), then prefer the last locality in a conventional street/city
    # address. Longest match breaks ties such as תל אביב / תל אביב-יפו.
    matches: list[tuple[int, int, str]] = []
    for locality in LOCALITY_TO_COURT:
        pattern = rf"(?<![א-ת])(?:[בלמכוה])?{re.escape(locality)}(?![א-ת])"
        for match in re.finditer(pattern, normalized):
            matches.append((match.start(), len(locality), locality))
    if not matches:
        return None
    last_position = max(start for start, _, _ in matches)
    start, _, locality = max(
        (item for item in matches if item[0] == last_position),
        key=lambda item: item[1],
    )
    pattern = rf"(?<![א-ת])(?:[בלמכוה])?{re.escape(locality)}(?![א-ת])"
    match = next(match for match in re.finditer(pattern, normalized) if match.start() == start)
    return locality, match.start(), match.end()


def locality_from_text(text: str) -> str:
    match = _locality_match(text)
    return match[0] if match else ""


def parse_address(text: str) -> ParsedAddress:
    """Parse common Israeli street/number/city formats without altering source text."""
    normalized = " ".join(text.replace("–", "-").split()).strip(" ,.-")
    city_match = _locality_match(normalized)
    locality = city_match[0] if city_match else ""
    without_city = normalized
    if city_match:
        _, start, end = city_match
        without_city = (normalized[:start] + " " + normalized[end:]).strip(" ,.-")

    number_match = re.search(r"(?<!\d)(\d{1,5}[א-ת]?)(?!\d)", without_city)
    if not number_match:
        return ParsedAddress(locality=locality)
    before_number = without_city[:number_match.start()].strip(" ,.-")
    # In combined business/address strings, use the text after the last street
    # designator so the business name does not leak into the parsed street.
    designator = list(re.finditer(r"(?:רחוב|שדרות|שד[׳']?|דרך)\s+", before_number))
    if designator:
        before_number = before_number[designator[-1].end():]
    else:
        before_number = re.sub(
            r"^(?:חנות|בית העסק|החברה|חברת|סניף)\s+",
            "",
            before_number,
        )
    return ParsedAddress(
        street=before_number.strip(" ,.-"),
        house_number=number_match.group(1),
        locality=locality,
    )


def resolve_jurisdiction(case: CaseData) -> JurisdictionResolution | None:
    transaction_locality = parse_address(case.incident_place).locality
    defendant_locality = parse_address(case.defendant.address).locality
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
