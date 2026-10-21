# Legal retrieval evaluation set

The reusable, synthetic baseline cases are in `tests/fixtures/legal_retrieval_eval_cases.json`.
They contain no real claimant or case data. Each case starts after interview extraction and
category classification, and records the authority IDs expected from the deterministic
retriever. The set is intended for regression testing and later expansion into an LLM/RAG
evaluation; it is not a legal opinion or proof that an authority applies to any real case.

Run the saved baseline with:

```powershell
python -m pytest tests/test_legal_retrieval_eval.py -q
```

The evaluation currently tests source-selection behavior and important exclusions. It does
not yet evaluate extraction quality, factual grounding in generated drafts, legal accuracy,
or answer quality. Those should be added as separate labeled checks rather than folded into
a single score.
