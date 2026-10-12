from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_deep_smoke import SCENARIOS, _completed_case
from small_claims_agent.graph import build_graph, initial_state
from small_claims_agent.llm import DemoAgentClient
from small_claims_agent.pdf_export import markdown_to_pdf_bytes


def main() -> None:
    destination = ROOT / "tmp" / "pdfs" / "deep_smoke"
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    graph = build_graph(DemoAgentClient())
    for scenario in (SCENARIOS[0], SCENARIOS[1], SCENARIOS[4]):
        case = _completed_case(scenario)
        state = initial_state(case)
        state["generate_requested"] = True
        result = graph.invoke(
            state,
            {"configurable": {"thread_id": "pdf-" + scenario["id"]}},
        )
        if result.get("validation_errors"):
            raise RuntimeError(result["validation_errors"])
        target = destination / f"{scenario['id']}.pdf"
        target.write_bytes(markdown_to_pdf_bytes(result["draft_markdown"]))
        outputs.append(str(target))
    print(json.dumps(outputs, ensure_ascii=True))


if __name__ == "__main__":
    main()
