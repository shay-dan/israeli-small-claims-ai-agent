"""Run bounded live structured-output checks against every configured LLM provider.

The report intentionally contains no API keys, prompts, or generated text.
Each configured provider receives six small requests to cover the production
operations without invoking the application UI or changing project data.
"""

from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from small_claims_agent.drafting import compose_claim_markdown
from small_claims_agent.draft_clarification import sanitize_draft_clarifications
from small_claims_agent.interview import question_for
from small_claims_agent.legal import retrieve_authorities_for_case
from small_claims_agent.llm import (
    DemoAgentClient,
    GeminiAgentClient,
    ModelUnavailable,
    OpenAICompatibleAgentClient,
)
from small_claims_agent.models import CaseData
from small_claims_agent.sample_data import air_conditioner_sample

_LIVE_CASE = None
_LIVE_DRAFT = ""


def _configured_providers():
    providers = [
        OpenAICompatibleAgentClient(
            "groq",
            os.getenv("GROQ_API_KEY", ""),
            os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            timeout_seconds=8,
        ),
        OpenAICompatibleAgentClient(
            "openrouter",
            os.getenv("OPENROUTER_API_KEY", ""),
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            os.getenv("OPENROUTER_MODEL", "openrouter/free"),
            timeout_seconds=8,
        ),
        # Gemini enforces a 10-second minimum manually set deadline.
        GeminiAgentClient(timeout_seconds=15, max_retries=0),
    ]
    return providers


def _call(provider, operation: str, fn) -> dict[str, object]:
    try:
        result = fn()
        if operation == "extract":
            detail = {
                "updates": len(result.updates),
                "category": result.inferred_category.value,
            }
        elif operation == "draft_sections":
            detail = {
                "background_items": len(result.factual_background),
                "attempt_items": len(result.resolution_attempts),
            }
        elif operation == "review_draft_for_clarification":
            safe_issues = sanitize_draft_clarifications(result, _LIVE_CASE, _LIVE_DRAFT)
            detail = {
                "provider_issue_count": len(result.issues),
                "accepted_issue_count": len(safe_issues),
                "accepted_fields": [item["field"] for item in safe_issues],
                "accepted_issues": safe_issues,
            }
        elif operation == "review":
            detail = {"verdict": result.verdict}
        elif operation == "review_interview":
            detail = {"verdict": result.verdict, "issue_count": len(result.issues)}
        elif operation == "repair_answer":
            detail = {"action": result.action, "confidence": result.confidence}
        else:
            detail = {}
        return {
            "provider": getattr(provider, "provider", "gemini"),
            "model": provider.model_name,
            "operation": operation,
            "status": "passed",
            **detail,
        }
    except Exception as exc:
        diagnostic = getattr(exc, "diagnostic", {})
        return {
            "provider": getattr(provider, "provider", "gemini"),
            "model": provider.model_name,
            "operation": operation,
            "status": "failed",
            "category": diagnostic.get("category", type(exc).__name__),
            "http_status": diagnostic.get("http_status"),
            "retryable": diagnostic.get("retryable"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("groq", "openrouter", "gemini"),
        help="Test only one provider (all configured providers run by default).",
    )
    parser.add_argument(
        "--operation",
        choices=(
            "extract", "review_interview", "draft_sections", "review",
            "review_draft_for_clarification", "repair_answer",
        ),
        help="Test one operation (all supported operations run by default).",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    providers = [
        provider
        for provider in _configured_providers()
        if args.provider is None or getattr(provider, "provider", "gemini") == args.provider
    ]
    case = air_conditioner_sample()
    records = retrieve_authorities_for_case(case)
    authority_data = [record.model_dump(mode="json") for record in records]
    authority_ids = {record.id for record in records}
    deterministic_sections = DemoAgentClient().draft_sections(case, authority_data)
    draft = compose_claim_markdown(case, deterministic_sections, records)
    global _LIVE_CASE, _LIVE_DRAFT
    _LIVE_CASE, _LIVE_DRAFT = case, draft
    opening = "קניתי אוזניות ב500 שקל ואחרי שעה הן הפסיקו לעבוד"
    operations = []

    def record(provider, operation, fn):
        result = _call(provider, operation, fn)
        operations.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)

    for provider in providers:
        if not provider.available:
            operations.append(
                {
                    "provider": getattr(provider, "provider", "gemini"),
                    "model": provider.model_name,
                    "status": "skipped_missing_key",
                }
            )
            continue
        checks = [
            ("extract", lambda: provider.extract(opening, CaseData())),
            ("review_interview", lambda: provider.review_interview(case)),
            ("draft_sections", lambda: provider.draft_sections(case, authority_data)),
            ("review", lambda: provider.review(case, draft, authority_ids)),
            (
                "review_draft_for_clarification",
                lambda: provider.review_draft_for_clarification(case, draft, authority_ids),
            ),
            (
                "repair_answer",
                lambda: provider.repair_answer(
                    "prior_attempts",
                    question_for("prior_attempts", case),
                    ["צלצלתי לחנות", "אמרו שלא מחזירים כסף"],
                    case,
                ),
            ),
        ]
        for operation, call in checks:
            if args.operation is None or args.operation == operation:
                record(provider, operation, call)

    passed = sum(item.get("status") == "passed" for item in operations)
    failed = sum(item.get("status") == "failed" for item in operations)
    skipped = sum(item.get("status") == "skipped_missing_key" for item in operations)
    print(
        json.dumps(
            {
                "summary": {"passed": passed, "failed": failed, "skipped": skipped},
                "providers": [
                    {
                        "provider": getattr(provider, "provider", "gemini"),
                        "model": provider.model_name,
                        "configured": provider.available,
                    }
                    for provider in providers
                ],
                "operations": operations,
                "status": "passed" if failed == 0 and skipped == 0 else "completed_with_failures",
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
