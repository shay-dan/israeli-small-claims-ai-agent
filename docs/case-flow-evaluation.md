# End-to-end case-flow evaluation

The synthetic, reusable conversations are stored in `tests/fixtures/case_flow_eval_cases.json`.
They are replayed through the same local interview answer handler, missing-field progression,
case validation, deterministic authority retrieval, and local draft-generation graph used by
the application. The fixture contains eight complete scenarios covering consumer appliances,
an online course, a partial duplicate-charge claim, and residential rental disputes. Separate
guardrail tests cover an incomplete opening and a claim above the small-claims limit.

Run this suite with:

```powershell
python -m pytest tests/test_case_flow_eval.py -q
```

The suite uses synthetic values and `DemoAgentClient`; it makes no external provider calls.
It checks required draft content, forbidden unsupported phrases, preserved party details,
claim totals, selected source IDs, and that incomplete or ineligible cases do not produce a
draft. It does not establish legal correctness and is not a substitute for review by a legal
professional. The separate retrieval-only baseline is documented in
`docs/legal-retrieval-evaluation.md`.
