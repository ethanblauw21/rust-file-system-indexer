# ADR-005: Magnet-Doc Suppression — Diagnosed, Not Built (Skip)

**Status:** Accepted — verdict settled 2026-07-08 as **SKIP**. This ADR records a pre-build impact analysis that decided *not* to implement the change; no code was written and no sweep was run.
**Date:** 2026-07-08
**Deciders:** Ethan (chose "confirm impact first, then decide"); analysis by the main agent, gated by the same reproducible-corpus discipline as ADR-004.
**Depends on:** none formally. Measured against the same reproducible baseline as [ADR-004](ADR-004-trigram-fuzzy-fts-no-ship.md) (hybrid 4.19 / 8-of-16 @ `6fa6013`).
**Required by:** none yet.

## Context

Dogfooding finding **List-1 #7** observed "magnet docs" — generated aggregate files whose
chunks surface across many unrelated queries (the original example was `GRAPH_REPORT.md`). Phase 5.2
proposed a bounded per-file penalty / prior to demote such files. Candidate signals named in the
plan: generated-artifact path patterns, extreme chunk count, or a docs-vs-impl type prior.

Per the lesson from ADR-004 (a ranking change can be corpus-specific and evaporate on a different
corpus), this experiment was **impact-checked before any code was written**, against the Phase-4
reproducible corpus (public `codebase-indexer` + 5 ADR-019 distractor repos, 11,339 chunks).

## Analysis (evidence)

**1. Magnets are real on this corpus.** A 112-query diagnostic (hybrid, top-10) counting how many
distinct query-targets each file surfaces under:

| file | across targets | total hits | chunks | is a target? |
|---|---|---|---|---|
| `graph.html` | 9/16 | 43 | 789 | no (generated viz) |
| `GRAPH_REPORT.md` | 9/16 | 27 | 44 | no (the dogfood culprit) |
| `adr-backlog.md` | 8/16 | 61 | 72 | no (aggregate backlog) |
| `references-code-intelligence.md` | 5/16 | 19 | 33 | no |
| `ADR-003-…-adapter-architecture.md` | 6/16 | 68 | 50 | **yes (q14)** |

**2. Chunk-count is NOT a usable separator.** Corpus chunk-count median = 5, p90 = 31, max = 789.
Legitimate impl targets carry *more* chunks than most magnets: `incremental_indexer.py` = 113,
`hybrid_retriever.py` = 88, `ast_chunker.py` = 86, versus `GRAPH_REPORT.md` = 44, `adr-backlog.md`
= 72. Only `graph.html` (789) is a clean outlier. A naive high-chunk-count penalty would regress
q1/q2/q3 (the impl targets).

**3. Magnets do not displace *recoverable* targets.** Of 21 top-3 misses in the suite, 14 are pure
semantic gaps (no magnet ranked above the target). The 7 with a magnet above the target were
checked at depth (top-30):

| query | target | true rank | magnets above | rank if all removed | recoverable? |
|---|---|---|---|---|---|
| q5 synonym | `stable_id.py` | 23 | 6 | 17 | no (still >10) |
| q7 base/reorder/truncate/recase | `backend.py` | not in top-30 | — | — | no (deep semantic miss) |
| q8 synonym | `import_resolver.py` | 12 | 4 | 8 | maybe (12→8, top-10 only) |

The entire achievable upside of magnet suppression on this corpus is **one cell** — q8 synonym
top-10 — and only if the penalty fully demotes all four magnets above the target (≈ +0.06 mean
robustness), against real regression risk to the high-chunk impl targets and to `ADR-003` (a
legitimate rank-1 doc target with magnet-like breadth).

## Decision

**Do not build magnet-doc suppression on this corpus.** The reproducible public-repo corpus does
not exhibit the metric-limiting magnet pathology the original personal-data corpus did: magnets
clutter results but are not the binding constraint on the top-3/top-10 hit metric the harness gates
on. Building a suppression would risk regression for at most a single-cell, contingent gain — it
fails the ship bar (beat baseline without regressing a rank-1 query) before it is even written.

## Consequences

- **Better:** No regression risk introduced; the reproducible baseline (hybrid 4.19 / 8-of-16)
  stands unchanged. The "confirm impact first" pre-check cost ~200 searches and avoided a full
  build+sweep cycle to rediscover a no-op (contrast ADR-004, which was measured the long way).
- **Worse / honest limits:** The top-3/top-10-hit-on-one-target metric does not capture the
  *subjective* clutter cost of magnets (a user still sees `graph.html` across many searches). If a
  future eval adds a precision/clutter metric, magnet suppression may become worth revisiting.
- **Future work:** If pursued, the signal must be smarter than chunk-count — e.g. low
  `coherence_score` (an aggregate's chunks are topically diffuse) combined with a generated-path /
  `.html` prior — and should target top-10 recovery on q8-like cases without touching the impl and
  ADR targets. Not scheduled.

## Implementation Log

- 2026-07-08 — Impact analysis run (magnet diagnostic + top-3 displacement check + top-30 depth
  check). Verdict SKIP. No source changed. Recorded in agent memory
  (`magnet-doc-suppression-skip`). Concludes Phase 5: both candidate experiments (trigram → ADR-004,
  magnet-doc → this ADR) evaluated and not shipped on the reproducible corpus.
