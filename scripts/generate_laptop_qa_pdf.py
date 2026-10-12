from decimal import Decimal
from pathlib import Path
import sys


root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from small_claims_agent.drafting import compose_claim_markdown
from small_claims_agent.legal import retrieve_authorities_for_case
from small_claims_agent.llm import DemoAgentClient
from small_claims_agent.models import CaseData, ClaimCategory, DamageItem, EvidenceItem, Party, RemedyType
from small_claims_agent.pdf_export import markdown_to_pdf_bytes


target = root / "tmp" / "pdfs" / "laptop_formal_claim.pdf"
target.parent.mkdir(parents=True, exist_ok=True)

case = CaseData(
    claimant=Party(
        full_name="דמות בדיקה",
        identifier="123456789",
        address="רחוב הקרן 8 ירושלים",
        phone="0545363636",
    ),
    defendant=Party(
        full_name="KSP",
        address="רחוב השלום 1 תל אביב",
        legal_form="company",
    ),
    category=ClaimCategory.CONSUMER_PRODUCT,
    court_city="תל אביב-יפו",
    jurisdiction_basis=(
        "העסקה והאירוע נושא התביעה התרחשו בתל אביב, המצויה בתחום השיפוט של בית המשפט."
    ),
    transaction_date="01.01.2026",
    incident_date="01.01.2026",
    incident_place="רחוב השלום 1 תל אביב",
    product_or_service_details="לפטופ",
    defendant_role="מוכר",
    event_summary=(
        "קניתי לפטופ ב4000 שקל, פתחתי את הקופסה מהחנות וראיתי שהמסך של הלפטופ שבור, "
        "ניסיתי להחזיר את הלפטופ לחנות ולקבל את הכסף חזרה והם אמרו לי שאין החזרים "
        "ושאין סיכוי שהם מכרו לי ככה את הלפטופ ושאני שברתי אותו"
    ),
    prior_attempts=[
        "ניסיתי להחזיר את הלפטופ לחנות ולקבל את הכסף חזרה והם אמרו לי שאין החזרים "
        "ושאני שברתי אותו"
    ],
    damage_items=[DamageItem(label="השבת מחיר המוצר", amount=Decimal("4000"))],
    primary_remedy=RemedyType.REFUND,
    alternative_remedy=RemedyType.REPAIR,
    warranty_details="המוצר נרכש כחדש וקיימת אחריות, אך לא נמסרו פרטים נוספים.",
    evidence=[EvidenceItem(name="העתק קבלה/חשבונית בגין העסקה")],
    confirmed_no_more_evidence=True,
    facts_confirmed=True,
)

records = retrieve_authorities_for_case(case)
sections = DemoAgentClient().draft_sections(
    case, [record.model_dump(mode="json") for record in records]
)
markdown = compose_claim_markdown(case, sections, records)
target.write_bytes(markdown_to_pdf_bytes(markdown, title="טיוטת כתב תביעה"))
print(target.as_posix().encode("unicode_escape").decode("ascii"))
