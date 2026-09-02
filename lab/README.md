# Lab protocol

Python 3.9+. Product seams (`harness.compaction_residual`,
`harness.compaction_vault`, `ConversationalSession`) come from a
Marionette checkout. This package is the research runner that used to
live under `pmharness/compaction_residual_*.py`.

```bash
export PYTHONPATH="$(pwd)/lab:/path/to/marionette"
python -m catalog_residual.bench
python -m catalog_residual.live
python -m catalog_residual.projection
python -m catalog_residual.projection --provider-continuity
python -m catalog_residual.projection --semantic-continuity
python -m catalog_residual.projection --validate-provider
python -m pytest -q lab/tests
```

`live_cases()` is the claim-grade suite. Experimental cells stay
opt-in via `--case`. Do not add them to `live_cases()`.

## Projection after prefix mutation

Bound thinking must be discarded after a client-side prefix rewrite.
`python -m catalog_residual.projection` reruns the hermetic Pi_tilde
surrogate (no API keys) and prints a `catalog_residual_projection/v1`
receipt. It does not claim that catalog recovers discarded reasoning.
It scores catalog on the mutated keep-tail after projection, not the
full original transcript.

Keep these evidence kinds separate:

- **Fable protocol proof** — `--validate-provider`. Live only with
  `--live` and `ANTHROPIC_API_KEY`. Exact `claude-fable-5-1`. Sets
  `provider_validated` only after authentic signed bind, unprojected
  binding mismatch, projected 200 exact-model, and `drop_block`
  reporting `prefix_binding_mismatch`. A dry-run or injected
  transport is a request-shape fixture, not protocol proof.
- **Direct Anthropic task-continuity pilot** —
  `--provider-continuity`. Default model is exact `claude-fable-5-1`.
  Repeatable `--model` may add exact `claude-opus-5`. `--repeats`
  default 3.
  Dry-run unless `--live` and a key exist. Public receipts record the
  causal chain without raw thinking: planted checkpoint, bootstrap
  gates, dropped-block digest/bytes, projection flags, residual
  digest/containment, and per-arm scores. HTTP 200 is not task
  success. This is not Fable protocol proof and does not restamp
  `claim_ready`.
- **Luna Pi_tilde surrogate** — default hermetic run and optional
  `--semantic-continuity`. Hidden-only state loss is
  `end_task_success=false` (no hidden-state recovery).
  `invented_concrete_value` scores concrete invented answers such as
  Luna's `think-nonce-after assistant 7`; `honesty_clean` is
  `not invented_concrete_value`, not `not false_recall`.
- **Broader task continuity** — unproven. Do not read any of the
  above as a general claim that Catalog Residual restores discarded
  reasoning.

`completed_work_visible` is presence evidence only;
`no_repeat_completed_work` stays null. Raw live artifacts stay
gitignored under `artifacts/`.

Planted cells (`thinking_only_nonce`, `thinking_then_tool`,
`thinking_plan_next`, `last_wins_under_projection`) stay out of
`live_cases()` and the generic `cases_by_id()` / bench-live `--case`
catalog. Select them only through `python -m catalog_residual.projection`.
Do not restamp `claim_ready`.
