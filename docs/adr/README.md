# Architecture Decision Records

This directory holds `file_indexer`'s Architecture Decision Records — the durable, numbered
record of significant design and ranking decisions. It formalizes what
`docs/dogfooding/2026-06-22-dogfood-run.md` was doing by hand for the R1 and R2 ranking
experiments before this process existed (see "Formalizing prior decisions" below). See
`CONTRIBUTING.md` §4 for how an ADR fits into the overall contribution workflow.

## When an ADR is required

Per `CONTRIBUTING.md` §1/§4: any **Major** change — anything touching `src/` (ranking/fusion
logic in `search.rs`, the indexing pipeline in `indexer.rs`, chunking in `chunker.rs` /
`chunker_map.rs`, chunk scoring in `scorer.rs`, the TUI, storage, or the CLI surface) — needs an
ADR written and committed (`Status: Proposed`) before implementation begins. Docs-only or
comment-only changes (Minor) do not require one.

## Numbering

Sequential, zero-padded to three digits: `ADR-XXX-short-slug.md`. Check `docs/adr/` for the
highest existing number and increment — never reuse or backfill a number, even for a rejected or
withdrawn ADR. `ADR-000-template.md` is reserved for the template itself and is never a real
decision.

## Status vocabulary

Recorded in the `**Status:**` header field, capitalized:

- **Proposed** — written and committed before implementation begins.
- **Accepted** — implementation landed; every obligation listed under that ADR's `Required by`
  has been resolved.
- **Superseded by ADR-XXX** — a later ADR replaced this decision. The superseding ADR should
  record this one under its own `Depends on`.
- **Deprecated** — no longer in effect, with no direct successor.

## Lifecycle

1. Write the ADR before implementation begins. Commit it with `Status: Proposed`.
2. Work the change on its branch. Update the Implementation Log as you go.
3. Flip to `Status: Accepted` and commit the final ADR as part of the merging PR.

## Cross-reference protocol (`Depends on` / `Required by`)

Bidirectional, and required whenever a change touches more than one ADR's territory. A
one-directional link is a defect: if A's `Depends on` names B, B's `Required by` must name A back,
kept in sync in the same PR.

- **Downstream ADR → `Depends on:`** name the upstream ADR and the *exact* artifact, decision, or
  confirmation needed from it, so an implementor knows to wait, and knows what for.
- **Upstream ADR → `Required by:`** name each consumer ADR and what it needs. Resolve every listed
  obligation before flipping the upstream ADR to `Accepted` — don't leave the downstream
  implementor to rediscover it later.

This repo renames the sibling `codebase-indexer` project's `Depended on by:` field to
`Required by:` — same semantics. The shape was established by
[`ADR-001-push-streamed-ingestion.md`](ADR-001-push-streamed-ingestion.md), which already uses
`Deciders` / `Depends on` / `Required by` (not the sibling's `Branch` / `Reviewer` /
`Depended on by`).

## Ranking-experiment ADRs

Every ranking, fusion, tokenization, or scoring change is sweep-gated: the `search-eval-scientist`
agent owns the mutation/sweep harness and returns a paired before/after table before any such
change ships. Once that evidence exists, record it as an ADR using the template's `## Evidence`
section, and:

- **Take the sweep table verbatim** from the search-eval-scientist — dense-only and hybrid,
  top-3/top-10 hit rates, mean robustness, git-SHA-stamped. Never paraphrase or recompute the
  numbers.
- **State the context as the specific finding** that motivated the trial — a mutation-suite
  finding number (e.g. M1, M7) or a dogfood-doc List-1 entry (e.g. "List-1 #7 magnet docs").
- **State the decision as ship / no-ship**, and name the metric that decided it.
- **Record no-ship decisions too.** A rejected experiment gets an ADR so it isn't blindly
  re-tried later; state what evidence would change the verdict.
- **Note regressions watched** — e.g. rank-1 doc/ADR queries regressing when a code-favoring
  change ships.
- **Cite the current baseline**, not a stale one. Baselines move forward as experiments ship; cite
  whatever the search-eval-scientist states is current at the time the sweep was run.

## Formalizing prior decisions

Two ranking experiments were already run and recorded informally in
`docs/dogfooding/2026-06-22-dogfood-run.md` before this process existed. Their ADR numbers are
reserved but the ADRs themselves are not yet written as of this file's creation:

- **R1 — weighted RRF fusion** (fixes finding M7, hybrid trailing pure dense): reserved
  **ADR-002**. Moved the corpus from 3.69 to 3.94 mean robustness (9/16 → 10/16 fully robust).
- **R2 — path-coverage boost** (fixes M8 code-vs-docs burial and M9 filename-handle collisions):
  reserved **ADR-003**, depends on R1's RRF weights being held fixed. Moved the corpus from 3.94
  to 4.69 mean robustness (10/16 → 12/16 fully robust) — this 4.69 / 12-of-16 figure is the
  baseline every ranking ADR written after R2 should cite, not R1's 3.94 / 10-of-16.

Writing these up is separate work, to be done once each experiment's exact sweep table is
re-confirmed against the eval harness — this README only reserves the slots so ADR numbering
stays consistent when they are authored.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](ADR-001-push-streamed-ingestion.md) | Push / Streamed Ingestion — a producer-driven ingress alongside the pull walker | Proposed |

Update this table whenever an ADR's file or status changes.
