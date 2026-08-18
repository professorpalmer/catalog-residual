# Lab protocol

Python 3.9+. Product seams (`harness.compaction_residual`,
`harness.compaction_vault`, `ConversationalSession`) come from a
Marionette checkout. This package is the research runner that used to
live under `pmharness/compaction_residual_*.py`.

```bash
export PYTHONPATH="$(pwd)/lab:/path/to/marionette"
python -m catalog_residual.bench
python -m catalog_residual.live
python -m pytest -q lab/tests
```

`live_cases()` is the claim-grade suite. Experimental cells stay
opt-in via `--case`. Do not add them to `live_cases()`.
