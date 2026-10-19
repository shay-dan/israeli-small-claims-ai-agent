from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from small_claims_agent.llm import LLMGateway, ModelUnavailable
from small_claims_agent.models import CaseData


FICTIONAL_OPENING = "קניתי כדורגל ב-10 שקלים, ואחרי יום הוא התפוצץ."


def safe_error(exc: Exception, provider_name: str) -> dict:
    diagnostic = getattr(exc, "diagnostic", {}) or {}
    return {
        "provider": diagnostic.get("provider", provider_name),
        "model": diagnostic.get("model"),
        "status": "failed",
        "category": diagnostic.get("category", type(exc).__name__),
        "http_status": diagnostic.get("http_status"),
    }


def main() -> int:
    load_dotenv(ROOT / ".env")
    gateway = LLMGateway()
    if not gateway.available:
        print(json.dumps({"status": "skipped", "reason": "no_provider_keys_configured"}))
        return 2

    results: list[dict] = []
    attempted: set[str] = set()
    try:
        extraction = gateway.extract(FICTIONAL_OPENING, CaseData())
        successful_provider = gateway.last_provider or "unknown"
        attempted.update(item.get("provider") for item in gateway.last_attempts)
        attempted.add(successful_provider)
        results.extend(
            {
                "provider": item.get("provider"),
                "model": item.get("model"),
                "status": "fallback_triggered",
                "category": item.get("category"),
                "http_status": item.get("http_status"),
            }
            for item in gateway.last_attempts
        )
        results.append({
            "provider": successful_provider,
            "model": gateway.last_model,
            "status": "passed",
            "structured_output": type(extraction).__name__,
            "updates": len(extraction.updates),
        })
    except ModelUnavailable as exc:
        diagnostic = exc.diagnostic
        attempts = diagnostic.get("provider_attempts", [])
        attempted.update(item.get("provider") for item in attempts)
        results.extend({**item, "status": "failed"} for item in attempts)
        if not attempts:
            results.append(safe_error(exc, "gateway"))

    # The gateway stops after the first successful provider. Probe any remaining
    # configured providers once so this smoke check validates every available key.
    for provider in gateway.providers:
        name = getattr(provider, "provider", "gemini")
        if name in attempted:
            continue
        try:
            extraction = provider.extract(FICTIONAL_OPENING, CaseData())
            results.append({
                "provider": name,
                "model": provider.model_name,
                "status": "passed",
                "structured_output": type(extraction).__name__,
                "updates": len(extraction.updates),
                "probe": "provider_not_reached_after_gateway_success",
            })
        except Exception as exc:  # report only allowlisted diagnostic metadata
            results.append(safe_error(exc, name))

    passed = sum(item.get("status") == "passed" for item in results)
    failed = sum(item.get("status") == "failed" for item in results)
    print(json.dumps({
        "status": "passed" if passed and not failed else "failed",
        "configured_provider_count": len(gateway.providers),
        "checked_provider_count": len(results),
        "provider_results": results,
        "gateway_success_provider": gateway.last_provider,
        "gateway_success_model": gateway.last_model,
        "privacy": "fictional text only; no prompts, completions, API keys, or case PII logged",
    }, ensure_ascii=False, indent=2))
    return 0 if passed and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
