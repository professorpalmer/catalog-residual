# Extractive Residuals After Compaction

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Paper: CC BY 4.0](https://img.shields.io/badge/paper%20%26%20data-CC%20BY%204.0-lightgrey.svg)](LICENSE)

A lab study of compaction residuals in a coding harness: whether a
bounded extractive handle catalog, plus a last-wins selected story and a
session-scoped SQLite FTS vault, can be a factory-default residual
without claiming that summarization is obsolete.

> **Status: public lab notebook.** Findings include results that cut
> against early slogans. Hybrid+stems wins the claim-grade residual
> ranking. Catalog is the factory default because compact writes a
> selected story, not because FTS became a summarizer. We do not
> restamp `claim_ready` on experimental cells.

**Site.** [professorpalmer.github.io/catalog-residual](https://professorpalmer.github.io/catalog-residual/)

**Paper.** [`paper/paper.pdf`](paper/paper.pdf) (same file as
[`docs/paper.pdf`](docs/paper.pdf)).

**Product.** The shipping seams live in
[Marionette v0.9.245+](https://github.com/professorpalmer/marionette).
This repo is the research surface that used to hang off that tree.

## The question

After a long chat, the harness rewrites the compacted middle. The usual
residual is a paid paragraph. We measure four arms (summary, hybrid,
catalog, off) with deterministic substring oracles — no LLM-as-judge —
on two OpenRouter models (`gpt-5.6-luna`, `gpt-5.6-sol`), $n=3$.

## Headline findings (calibrated)

- **Complementarity, not a knockout.** On the claim-grade
  $11\times4\times3\times2$ factorial (264 trials, \$1.23), summary
  keeps designed catalog-miss nonces and drops generic policy stems;
  catalog keeps stems and drops nonce prose. Hybrid+stems is the only
  compact residual that keeps both on `residual_recall_round1`
  (B $60/60$ fact-bearing vs A $42/60$ vs C $54/60$).
  `claim_ready=true` on that merge. That ranking is not a default flip.
- **Dump-and-query is complementary.** Vault uniquely recovers buried
  prose when the later ask overlaps and the 256KB peek sidecar has
  already dropped the middle (Luna and Sol $3/3$ vault-on vs $0/3$
  vault-off). It does not select. Recap and paraphrase failed
  catalog+vault until compact wrote a selected story.
- **`miss_plan` died on purpose.** Empty FTS dumping the compact-time
  user extract poisoned unrelated asks. Empty FTS is now empty.
- **Factory-flip gate passed on both models.** 16 experimental cases,
  A vs C, 96 trials each (Luna \$0.082, Sol \$0.830). Factory residual
  in Marionette v0.9.244 is `catalog`. Summary and hybrid stay Settings
  opt-ins. v0.9.245 paints last-wins compact receipts and vault-cite
  chips; that is a transcript surface, not a residual-algorithm change.
- **Last-wins is part of the residual.** Topic overlap $0.6$ after
  polarity stripping; one-word acks (`Reversed.`, `Noted.`) never enter
  the story or the summarizer input. Summarizer last-wins then reached
  A reversal $2/3$ on both models (lexical misses, not rollback). We
  did not loosen the oracle to force $3/3$.

Not tweet-safe: stop summarizing, universal win, ``we solved recap.''

## Layout

| Path | What |
| --- | --- |
| `paper/paper.tex` | LaTeX preprint |
| `docs/` | GitHub Pages (`index.html` + `paper.pdf`) |
| `data/published_tables.json` | Sanitized aggregates from live receipts |
| `notes/COMPACTION_RESIDUAL_LAB.md` | Chronological lab notebook |
| `lab/catalog_residual/` | Battery, hermetic bench, live runner |
| `lab/tests/` | Protocol tests (need Marionette on `PYTHONPATH`) |

Raw live receipts stay out of git. They are large and contain residual
text. Regenerate with the runner.

## Reproduction

Needs a Marionette checkout (product `harness/` seams) and this lab on
`PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/catalog-residual/lab:/path/to/marionette
python -m catalog_residual.bench          # hermetic, no keys
python -m catalog_residual.live           # dry-run
python -m catalog_residual.projection     # Pi_tilde surrogate, no keys
python -m pytest -q lab/tests
```

Live replay is explicit:

```bash
python -m catalog_residual.live --live \
  --driver openrouter:openai/gpt-5.6-luna \
  --suite all --repeats 3
```

`--suite` is `core` (default), `holdout`, or `all`. `--case` selects
experimental cells. Do not add those cells to `live_cases()`.

Build the PDF (needs [tectonic](https://tectonic-typesetting.github.io/)):

```bash
sh paper/build.sh
```

## License

Code MIT. Paper and published tables CC BY 4.0.
