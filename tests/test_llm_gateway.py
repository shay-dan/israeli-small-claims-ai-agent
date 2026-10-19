from __future__ import annotations

import pytest
import httpx
import json

from small_claims_agent.llm import (
    LLMGateway,
    ModelUnavailable,
    OpenAICompatibleAgentClient,
    _openai_compatible_diagnostic,
)
from small_claims_agent.models import StateExtraction


class StubProvider:
    def __init__(self, name, outcome=None, error=None):
        self.provider = name
        self.model_name = f"{name}-test-model"
        self.available = True
        self.outcome = outcome or StateExtraction()
        self.error = error
        self.calls = 0

    def extract(self, *_args):
        self.calls += 1
        if self.error:
            raise self.error
        return self.outcome


def failure(category, retryable):
    return ModelUnavailable(
        f"test failure: {category}",
        diagnostic={"category": category, "retryable": retryable},
    )


def test_gateway_uses_providers_in_order_and_stops_on_first_success():
    groq = StubProvider("groq", error=failure("quota_or_rate_limit", True))
    router = StubProvider("openrouter")
    gemini = StubProvider("gemini")
    gateway = LLMGateway([groq, router, gemini])

    result = gateway.extract("fictional input", None)

    assert isinstance(result, StateExtraction)
    assert (groq.calls, router.calls, gemini.calls) == (1, 1, 0)
    assert gateway.last_provider == "openrouter"
    assert gateway.last_model == "openrouter-test-model"
    assert gateway.last_attempts == [
        {"provider": "groq", "model": "groq-test-model", "category": "quota_or_rate_limit", "http_status": None}
    ]


def test_gateway_uses_gemini_as_third_provider_after_two_provider_errors():
    groq = StubProvider("groq", error=failure("timeout", True))
    router = StubProvider("openrouter", error=failure("provider_error", True))
    gemini = StubProvider("gemini")
    gateway = LLMGateway([groq, router, gemini])

    gateway.extract("fictional input", None)

    assert [groq.calls, router.calls, gemini.calls] == [1, 1, 1]
    assert gateway.last_provider == "gemini"


def test_gateway_does_not_fail_over_for_invalid_request():
    groq = StubProvider("groq", error=failure("invalid_request", False))
    router = StubProvider("openrouter")
    gemini = StubProvider("gemini")
    gateway = LLMGateway([groq, router, gemini])

    with pytest.raises(ModelUnavailable) as error:
        gateway.extract("fictional input", None)

    assert error.value.diagnostic["category"] == "invalid_request"
    assert router.calls == gemini.calls == 0


def test_gateway_fails_over_when_provider_returns_invalid_structured_output():
    groq = StubProvider("groq", error=failure("structured_output_invalid", True))
    router = StubProvider("openrouter", error=failure("structured_output_invalid", True))
    gemini = StubProvider("gemini")
    gateway = LLMGateway([groq, router, gemini])

    result = gateway.extract("fictional input", None)

    assert isinstance(result, StateExtraction)
    assert [groq.calls, router.calls, gemini.calls] == [1, 1, 1]
    assert gateway.last_provider == "gemini"


def test_gateway_reports_provider_attempts_without_exception_payloads():
    providers = [
        StubProvider("groq", error=failure("timeout", True)),
        StubProvider("openrouter", error=failure("quota_or_rate_limit", True)),
        StubProvider("gemini", error=failure("provider_error", True)),
    ]
    gateway = LLMGateway(providers)

    with pytest.raises(ModelUnavailable) as error:
        gateway.extract("fictional input", None)

    diagnostic = error.value.diagnostic
    assert diagnostic["category"] == "all_providers_failed"
    assert [item["provider"] for item in diagnostic["provider_attempts"]] == [
        "groq", "openrouter", "gemini"
    ]
    assert "fictional input" not in repr(diagnostic)


def test_gateway_with_no_keys_is_unavailable_without_network_calls():
    gateway = LLMGateway([])

    assert gateway.available is False
    with pytest.raises(ModelUnavailable) as error:
        gateway.extract("fictional input", None)
    assert error.value.diagnostic["category"] == "no_provider_configured"


def test_openai_compatible_adapter_sends_strict_schema_and_parses_response():
    captured = {}

    def respond(request):
        captured.update(json.loads(request.content))
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "updates": [],
                        "ambiguities": [],
                        "contradictions": [],
                        "candidate_question_topics": [],
                        "inferred_category": "unsupported",
                    }),
                },
                "finish_reason": "stop",
            }],
        }
        return httpx.Response(200, json=body)

    provider = OpenAICompatibleAgentClient("groq", "fake-token", "https://example.invalid/v1", "test-model")
    from openai import OpenAI

    provider._client = OpenAI(
        api_key="fake-token",
        base_url="https://example.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    result = provider._structured_call("fictional prompt", StateExtraction)

    assert isinstance(result, StateExtraction)
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True


def test_openai_compatible_diagnostic_classifies_retryable_provider_errors_safely():
    class RateLimitError(Exception):
        status_code = 429

    diagnostic = _openai_compatible_diagnostic(
        RateLimitError("private key=never-log-this"), "groq", "test-model", "extract"
    )

    assert diagnostic["category"] == "quota_or_rate_limit"
    assert diagnostic["retryable"] is True
    assert diagnostic["provider"] == "groq"
    assert "never-log-this" not in repr(diagnostic)


def test_openai_compatible_output_validation_error_is_retryable_but_bad_request_is_not():
    class ResponseValidationError(Exception):
        pass

    class BadRequestResponseValidationError(ResponseValidationError):
        status_code = 400

    output_error = _openai_compatible_diagnostic(
        ResponseValidationError("invalid generated output"), "openrouter", "test-model", "draft_sections"
    )
    request_error = _openai_compatible_diagnostic(
        BadRequestResponseValidationError("invalid request schema"), "openrouter", "test-model", "draft_sections"
    )

    assert output_error["category"] == "structured_output_invalid"
    assert output_error["retryable"] is True
    assert request_error["category"] == "invalid_request"
    assert request_error["retryable"] is False
