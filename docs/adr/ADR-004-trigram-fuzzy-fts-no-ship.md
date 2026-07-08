# ADR-004: OOV-Scoped Trigram Fuzzy FTS5 Channel — No-Ship

**Status:** Accepted — verdict settled 2026-07-08 as **NO-SHIP**; the implementation this ADR records is the revert of `1a370f2`, not a merge.
**Date:** 2026-07-08
**Deciders:** Ethan (gated by the `search-eval-scientist`'s paired sweep verdict)
**Depends on:** none formally. The RRF weights this channel fuses alongside — `dense=1.0` (implicit), `RRF_SPARSE_WEIGHT=0.4`, `PATH_BOOST_WEIGHT=1.5` — are R1's and R2's shipped decisions, whose ADR slots (`ADR-002`, `ADR-003`) are reserved per `docs/adr/README.md` § "Formalizing prior decisions" but not yet authored as files. No formal cross-reference link is made to a non-existent file.
**Required by:** none yet.

## Context

Two mutation-suite findings motivated this experiment, both recorded in
`docs/dogfooding/2026-06-22-dogfood-run.md`:

- **M1 — typo fragility is the dominant weakness.** A single-char typo in the key term is the
  worst perturbation the suite exercises; it knocks queries out of top-10 and (in the original
  corpus) collapsed dense top-3 from 11→6. Example failures: `C++ tree-sitter` typo → miss, `C#
  csharp` typo → miss, `recursive file search` typo → miss.
- **M5 — `C#`/`C++` queries stay perturbation-fragile even post-fix.** Single-char tokens (`c`)
  plus "magnet doc" collisions make these queries brittle under typo/reorder/recase mutations.

The hypothesis: a **trigram-based fuzzy FTS5 channel**, scoped to fire only when a query token is
genuinely out-of-vocabulary for the existing porter-stemmed sparse index, could rescue exactly
these typo'd/misspelled tokens without touching queries the porter index already resolves.

This was built and integrated as an **unmerged, ungated experiment** — flagged explicitly in the
`search-eval-scientist`'s memory as "exactly the kind of change the charter's 'no unmeasured
ranking change ships' rule exists to catch." Per that charter (`docs/adr/README.md` §
"Ranking-experiment ADRs" / `CONTRIBUTING.md` §4), it required a full paired before/after sweep
against a reproduced baseline before any ship decision — this ADR is that gate's record.

The baseline it was measured against is the **reproducible-corpus baseline** established at Phase
4 (commit `6fa6013`, re-stamped for determinism at `1179016`): **hybrid mean robustness 4.1875
(≈4.19) / 5, 8 of 16 targets fully robust**, on the public `codebase-indexer` target repo (pinned
SHA) + 5 `ADR-019`-pinned distractor repos. This baseline explicitly supersedes the older,
non-reproducible **4.69 / 12-of-16** figure cited by R1/R2 (`docs/adr/README.md` § "Formalizing
prior decisions") — see `.claude/agent-memory/search-eval-scientist/harness-built-phase4.md` for
the corpus-swap rationale. Any ranking ADR written after this one should cite **4.19 / 8-of-16**,
not 4.69/12-of-16, until a future ADR moves it forward again.

## Decision

Trialed: an **OOV-scoped trigram fuzzy FTS5 channel** for typo recovery, built on branch
`dogfooding/trigram-fuzzy-fts` (commit `4f3f28b`, "Add OOV-scoped trigram fuzzy FTS5 channel for
typo recovery") and integrated onto the hardening branch
(`hardening/loud-modes-config-docs-harness`) as commit `1a370f2`, "feat: integrate OOV-scoped
trigram fuzzy FTS5 channel (Phase 5.1 candidate)" — a 3-way-merge cherry-pick, not a fast-forward.

Precisely what changed:

- **`db.rs`**: a new `chunks_trigram` FTS5 table + sync triggers, plus `run_trigram_match` /
  `trigram_search_oov`. Backfilled from existing chunks via a `user_version` migration on first
  open — **no re-embedding required**.
- **`search.rs`**: `hybrid_search` fuses a third RRF channel (trigram) alongside dense/sparse/path.
  The channel is **OOV-scoped** — it only fires on query tokens the porter/sparse index cannot
  match — and **best-effort** (never aborts a query on failure).
- **`config.rs` / `file_indexer.toml`**: `trigram_weight` added as a first-class `[fusion]` key,
  resolved through the same config-file > `env(RRF_TRIGRAM_WEIGHT)` > default precedence as the
  other fusion weights. Shipped-in-branch default: **`RRF_TRIGRAM_WEIGHT=0.3`**.

95/95 tests passed pre-gate. The commit message is explicit that the channel was "NOT yet gated —
the sweep verdict + ADR decide ship/no-ship." This ADR is that verdict: **NO-SHIP**. The change
will be **reverted from the branch** (`git revert` of `1a370f2`), not merged forward or shipped.

## Evidence

Reproducible corpus: public `codebase-indexer` @ pinned SHA (16 target files) + 5
`ADR-019`-pinned distractor repos (`click`, `zustand`, `p-queue`, `serilog`, `spdlog`) —
`union-v1`, 11,339 chunks / 11,214 embedded. Harness: `tools/eval_search.py`, 16 targets × 7
mutation columns, dense-only + hybrid, `k=112` searches per run.

Baseline row: `git_sha=6fa6013` (re-stamped `1179016`), `label=baseline-hardened-master`.
Trigram-after row: `git_sha=1a370f2`, `label=trigram-after`. Both rows sourced from
`tools/baselines/search_eval_baseline.jsonl` (baseline row recovered from commit `1179016`'s copy
of that file — see "Harness follow-up" under Consequences for why the current working copy no
longer carries it).

**Paired sweep table — hybrid mode, RRF defaults (top-3):**

| variant (top-3) | baseline (pre-trigram) | trigram-after |
|---|---|---|
| base | 14/16 | 14/16 |
| synonym | 6/16 | 5/16  ← REGRESSED (q11) |
| typo | 13/16 | 13/16  (target column — zero lift) |
| reorder | 14/16 | 14/16 |
| truncate | 15/16 | 15/16 |
| recase | 14/16 | 14/16 |
| filename_handle | 16/16 | 16/16 |
| **mean robustness (/5)** | **4.1875** | **4.1875** (flat) |
| **fully robust** | **8/16** | **8/16** (flat) |

**Paired sweep table — hybrid mode (top-10), from the same runs:**

| variant (top-10) | baseline (pre-trigram) | trigram-after |
|---|---|---|
| base | 15/16 | 15/16 |
| synonym | 9/16 | 9/16 |
| typo | 14/16 | 14/16 |
| reorder | 14/16 | 14/16 |
| truncate | 15/16 | 15/16 |
| recase | 15/16 | 15/16 |
| filename_handle | 16/16 | 16/16 |

**Dense-only mode:** unchanged across all columns (scope-leak check passed — dense doesn't touch
the FTS/trigram channel). Top-3: base 12/16, synonym 4/16, typo 9/16, reorder 13/16, truncate
11/16, recase 12/16, filename_handle 11/16 — identical baseline vs. trigram-after. Mean robustness
3.6875/5, fully robust 5/16 — identical both runs.

**Latency:** essentially unchanged (~1.06s p50 on the trigram-after run; baseline row shows
~1.24–1.29s p50 in the same JSONL, attributed to run-to-run machine variance rather than the
channel itself — the channel is best-effort and only fires on OOV tokens, so it does not add a
fixed per-query cost).

**Index size:** unchanged — 48,356,214 bytes (≈48.4 MB) both runs; the trigram table backfilled
into the existing index without re-embedding.

**Net effect:** only **ONE of 224 search cells** changed between the two runs: **q11 synonym
top-3 (Y → miss)** — a regression, not an improvement. The typo column — the column this channel
was built to fix — shows **zero lift** at both top-3 and top-10.

## Consequences

**Better:** none. No measurable improvement on any variant, including the target (typo) column.

**Worse:** one regressed rank-1 query (q11, synonym mutation, top-3) — a query that was retrievable
before the channel and is not after. New maintenance surface (`chunks_trigram` table + triggers +
migration + `trigram_weight` config key) was added and must now be removed on revert.

**Neutral:** dense-only mode, latency, and index size are unaffected — the OOV-scoping worked as
designed in the sense that it did not leak into paths it shouldn't touch; it simply didn't fire
usefully where it was meant to.

**Verdict: NO-SHIP.** The metric that decided it: **zero lift on the typo column** (the column
this experiment exists to fix) combined with a **measured regression** on synonym top-3, against
the current baseline (hybrid 4.19/5 mean robustness, 8-of-16 fully robust, `6fa6013`).

**Root cause (why it didn't fire).** The channel is OOV-scoped — it only activates on tokens the
porter-stemmed sparse index cannot match at all. On this large multi-repo distractor corpus, the
typo'd tokens it was meant to rescue (e.g. `merkel`→merkle in q3, `flie`→file in q11) **accidentally
match distractor content** (German-language text stems, "flies"-family stems in the distractor
repos) closely enough that the porter index does *not* treat them as OOV — so the rescue path never
triggers. Where the channel does fire elsewhere, the added noise is what caused the q11 regression.
The branch's originally-claimed 4.75→4.81 lift was measured on a different, corpus-specific setup
and **did not reproduce** on this reproducible corpus.

**What would change the verdict:** either (a) a corpus/query set where the target typo'd tokens are
genuinely OOV against the sparse index (not accidentally shadowed by distractor-corpus stems), or
(b) a fix to OOV detection itself so cross-lingual/stem-collision matches aren't treated as
in-vocabulary. **Weight retuning is not a remedy** — `RRF_TRIGRAM_WEIGHT` controls how much the
channel's results count once it fires; the failure mode here is that it isn't firing on the target
tokens at all, so no weight value rescues q3/q11.

**Regressions watched going forward:** q11 (synonym mutation, top-3) is the one query that moved
from hit to miss. If this or a related fuzzy-matching idea is retried, re-check q11 specifically,
not just the typo column.

**Consequence for the baseline:** none. The reproducible baseline (hybrid **4.19 / 8-of-16** at
`6fa6013`) stands unchanged as the Phase-5 gate — this experiment neither moved it forward nor
required correcting it downward.

**Harness follow-up (flagged, not fixed by this ADR):** `tools/eval_common.py`'s
`append_baseline` dedups on `(mode, weights, corpus)` but **not** the run `--label`. Because the
baseline run and the trigram-after run shared the same `(mode, weights, corpus)` = `(hybrid, {},
union-v1)` key (and the same for dense), the trigram-after row **silently overwrote** the baseline
row in `tools/baselines/search_eval_baseline.jsonl` — the working copy of that file today contains
only two rows, both `label=trigram-after`; the baseline row had to be recovered from git history
at commit `1179016`. This contradicts `tools/README.md`'s guidance to use `--label` to distinguish
before/after pairs on the same corpus. Flagged for the `search-eval-scientist` to fix (dedupe key
should include `label`, or before/after pairs need a corpus/weights discriminator) so this doesn't
silently destroy evidence on the next paired sweep.

## Alternatives Considered

| Option | Why rejected |
|--------|-------------|
| Ship with `RRF_TRIGRAM_WEIGHT` retuned away from 0.3 | Root cause is the OOV-detection gate not firing on the target tokens on this corpus, not a weight-magnitude problem. No weight value rescues a channel that never activates on q3/q11's tokens — not attempted, since the sweep evidence already rules it out. |
| Ship as an opt-in flag (default off) for corpora where OOV-firing works better | Adds a permanent maintenance surface (`chunks_trigram` table, triggers, migration, config key) for a channel with **no measured benefit on any corpus tested so far**. Deferred until a concrete corpus/config demonstrates real lift — an untested "might help elsewhere" flag is not a ship. |
| Leave `1a370f2` in place on the hardening branch, unmerged, without an explicit revert | Passive abandonment risks the commit flowing to `master` un-gated by accident later (it already passed 95/95 tests and looks "integrated"). An explicit `git revert` is the safer, auditable action and is what this ADR records as the implementation. |

## Implementation Log

- [x] Trigram branch built (`dogfooding/trigram-fuzzy-fts`, `4f3f28b`) — OOV-scoped
      `chunks_trigram` FTS5 channel, `RRF_TRIGRAM_WEIGHT` default `0.3`.
- [x] Integrated onto the hardening branch via 3-way-merge cherry-pick (`1a370f2`); 95/95 tests
      passing pre-gate.
- [x] `search-eval-scientist` paired sweep run against the reproducible baseline (`6fa6013`) — see
      Evidence above.
- [ ] `git revert` of `1a370f2` on `hardening/loud-modes-config-docs-harness`.
- [ ] `search-eval-scientist`: fix `eval_common.append_baseline`'s dedupe key to include `--label`
      (or otherwise distinguish before/after pairs sharing `(mode, weights, corpus)`) so a paired
      sweep on the same corpus stops silently overwriting its own baseline row.
- [ ] Resolve every downstream obligation listed in **Required by** before setting status to
      **Accepted** — none exist for this ADR, so this is vacuously satisfied.

**Notes:**
<!-- 2026-07-08: Verdict recorded as NO-SHIP. Baseline row for this comparison had to be recovered
     from git history (commit 1179016) because the working copy of
     tools/baselines/search_eval_baseline.jsonl only retained the trigram-after row — see the
     "Harness follow-up" consequence above. -->
