from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from small_claims_agent.graph import build_graph, initial_state
from small_claims_agent.interview import question_for
from small_claims_agent.llm import GeminiAgentClient
from small_claims_agent.models import CaseData
from small_claims_agent.sample_data import air_conditioner_sample


def main() -> None:
    load_dotenv(ROOT / ".env")
    client = GeminiAgentClient()
    report: dict[str, object] = {
        "available": client.available,
        "model": client.model_name,
    }
    if not client.available:
        report["status"] = "skipped_missing_api_key"
        print(json.dumps(report, ensure_ascii=True))
        return

    opening = "קניתי אוזניות ב500 שקל ואחרי שעה הן הפסיקו לעבוד"
    extraction = client.extract(opening, CaseData())
    report["extraction"] = {
        "updates": len(extraction.updates),
        "category": extraction.inferred_category.value,
        "questions": len(extraction.candidate_question_topics),
    }

    case = air_conditioner_sample()
    state = initial_state(case)
    state["generate_requested"] = True
    graph_result = build_graph(client).invoke(
        state,
        {"configurable": {"thread_id": "live-gemini-deep-smoke"}},
    )
    report["draft_graph"] = {
        "valid": not graph_result.get("validation_errors"),
        "draft_chars": len(graph_result.get("draft_markdown", "")),
        "authorities": len(graph_result.get("authorities", [])),
        "review_verdict": graph_result.get("draft_review", {}).get("verdict", "missing"),
        "warnings": len(graph_result.get("warnings", [])),
        "model_status": graph_result.get("model_status", ""),
        "review": graph_result.get("draft_review", {}),
        "warning_messages": graph_result.get("warnings", []),
    }

    repair = client.repair_answer(
        "prior_attempts",
        question_for("prior_attempts", case),
        ["צלצלתי לחנות", "אמרו שלא מחזירים כסף"],
        case,
    )
    report["repair"] = {
        "action": repair.action,
        "confidence": repair.confidence,
        "has_repaired_answer": bool(repair.repaired_answer),
        "has_clarification": bool(repair.clarification_question),
    }
    report["status"] = "passed"
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
