# ADR-XXX: [Short Title]

**Status:** Proposed | Accepted | Superseded by ADR-XXX | Deprecated
**Date:** YYYY-MM-DD
**Deciders:** name(s)
**Depends on:** none | ADR-XXX — *what this ADR needs from it (the specific artifact, decision, or confirmation)*
**Required by:** none yet | ADR-XXX — *what that ADR needs from this one (resolve these before Accepted — see CONTRIBUTING.md §4)*

## Context

What situation or problem prompted this decision? Include any constraints, prior art, or relevant system state.

> Ranking / retrieval / fusion / scoring changes: state the specific finding that motivated the experiment (e.g. a numbered mutation-suite finding, a dogfood-doc List-1 entry) as precisely as possible — see `docs/adr/README.md` § Ranking-experiment ADRs for the full evidentiary requirements before writing this section.

## Decision

What did we decide to do? State it directly.

> Ranking-experiment ADRs: name precisely what was trialed — the exact parameter, weight, tokenizer, or threshold changed — not just the intent.

## Evidence

*(Required for ranking / fusion / tokenization / scoring changes. Omit this section entirely for changes with no eval harness to run against — e.g. a pure ingestion, storage-layer, or tooling ADR.)*

Paste the paired before/after sweep table **verbatim** from the `search-eval-scientist`: dense-only and hybrid, top-3/top-10 hit rates, mean robustness, git-SHA-stamped, measured against the current baseline. Do not paraphrase or recompute the numbers — an ADR that fudges its evidence is worse than none.

## Consequences

**Better:** what this enables or improves.
**Worse:** what gets harder, slower, or more complex as a result.
**Neutral:** notable side effects that are neither good nor bad.

> Ranking-experiment ADRs: state the ship / no-ship verdict explicitly and name the metric it turned on. If shipped, state the new baseline this ADR establishes. If rejected, state what evidence would change the verdict, and flag any regressions watched (e.g. rank-1 doc/ADR queries regressing when a code-favoring change ships).

## Alternatives Considered

| Option | Why rejected |
|--------|-------------|
| Option A | brief reason |
| Option B | brief reason |

## Implementation Log

> Updated during development. Record deviations from the design, surprises, and decisions made in the moment.

- [ ] Task or milestone one
- [ ] Task or milestone two
- [ ] Resolve every downstream obligation listed in **Required by** (answer/confirm what each consumer ADR needs) before setting status to **Accepted**

**Notes:**
<!-- Add dated comments as you go -->
<!-- 2026-06-11: Discovered X assumption was wrong; pivoted to Y approach instead -->
