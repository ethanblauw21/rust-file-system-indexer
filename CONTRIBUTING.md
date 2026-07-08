# Contributing to file_indexer

This process is ported from the sibling `codebase-indexer` repo's governance and adapted to this
project's shape (a Rust CLI/library, not a Python MCP server). See §7 for what was deliberately
**not** ported and why.

## 1. Change Classification

Before starting any work, classify your change. This determines how much process applies.

| Classification | Definition | Process |
|---------------|-----------|---------|
| **Minor** | Docs, comments, README, CLAUDE.md, ADR prose — no changes to `src/` | Commit directly to `master`. No ADR, no issue, no PR required. |
| **Major** | Any change to `src/` — ranking/fusion logic (`search.rs`), the indexing pipeline (`indexer.rs`), chunking (`chunker.rs`, `chunker_map.rs`), chunk scoring (`scorer.rs`), the TUI (`tui.rs`), storage (`storage.rs`, `db.rs`), the CLI surface (`main.rs`), or project config | Branch required. ADR required. PR required. |

When in doubt, treat it as Major.

## 2. Branch Naming

| Pattern | When to use |
|--------|-------------|
| `feature/adr-XXX-short-name` | New feature tied to an ADR |
| `fix/issue-XXX-short-name` | Bug fix tied to a GitHub issue |
| `fix/short-name` | Minor bug fix with no issue |
| `chore/short-name` | Tooling, deps, config, governance |
| `ranking/short-name` | A ranking/fusion/scoring experiment gated by the search-eval-scientist's sweep harness (see `docs/adr/README.md` § Ranking-experiment ADRs) |

## 3. Bug Reporting

**Non-trivial bugs** (root cause unknown, or the fix touches shared indexing/retrieval/ranking code — `indexer.rs`, `search.rs`, `chunker.rs`, `db.rs`) must be filed as a GitHub issue before a fix is started. Include a root-cause hypothesis and proposed solution if known.

**Minor bugs** (obvious one-liner fix, isolated to a single function, no shared code touched) can be fixed directly with a descriptive commit message.

## 4. Architecture Decision Records (ADRs)

ADRs are required for all Major changes.

- **Location:** `docs/adr/`
- **Template:** `docs/adr/ADR-000-template.md`
- **Process doc:** `docs/adr/README.md` — numbering, statuses, and the cross-reference protocol in full.
- **Numbering:** Sequential. Check existing ADRs and increment.
- **Lifecycle:**
  1. Write the ADR before implementation begins. Commit it to `master` with status `Proposed`.
  2. Work on the feature branch. Update the Implementation Log as you go — record deviations, surprises, and in-the-moment decisions.
  3. Update status to `Accepted` and commit the final ADR as part of the PR.
- **Cross-references (REQUIRED for multi-ADR changes):** When one ADR depends on another, the link must be recorded in *both* ADRs via the header fields, and kept in sync in the same PR. A one-directional link is a defect.
  - **Downstream ADR → `Depends on:`** name the upstream ADR and the *exact* artifact/decision/confirmation needed from it, so an implementor knows to **wait** and knows *what for* before starting.
  - **Upstream ADR → `Required by:`** name each consumer ADR and what it needs. On completing the upstream implementation, **resolve those obligations** (answer the open questions, confirm the contracts) before setting status to `Accepted` — rather than leaving the downstream implementor to rediscover them later.
  - This is the sibling repo's `Depends on:` / `Depended on by:` pair, renamed to `Depends on:` / `Required by:` — see `docs/adr/README.md` and `ADR-001-push-streamed-ingestion.md`, which already establishes this field shape.
- **Ranking-experiment ADRs:** every ranking/fusion/tokenization/scoring change is gated by the `search-eval-scientist`'s mutation/sweep harness — no such change ships without a paired before/after sweep table. See `docs/adr/README.md` § Ranking-experiment ADRs for the exact evidence shape required.
- **AI agents:** Use `/grill-plan` to draft an ADR before implementing non-trivial features.

## 5. Pull Requests

- Manually verify `cargo build` succeeds and `cargo test` passes, plus a smoke round-trip (`cargo run -- index <dir>` then `cargo run -- search <query>`) before requesting review.
- Update the ADR Implementation Log before merging (Major changes only).
- **Merge strategy:** Squash merge onto `master`. Each PR becomes one commit. See §7 — this is a deliberate divergence from the sibling repo, not an oversight.

## 6. Commit Messages

Imperative mood, present tense. Describe what the commit does, not what you did.

```
Add weighted RRF fusion to fix hybrid trailing pure dense
Fix FTS5 crash on punctuation-only query terms
Update chunker_map versioning docs for AI consumption clarity
```

## 7. Deliberate Divergences from the Sibling Repo's Governance

This project ported the sibling `codebase-indexer` repo's classification table and ADR
cross-reference protocol, but **not** everything — two things were deliberately left out rather
than silently dropped:

- **Merge strategy.** The sibling repo uses merge commits only (no squash, no rebase) to preserve
  branch history. This repo squash-merges (§5) — one commit per PR. If you are used to the
  sibling's convention, note that `git log --grep` here searches squashed PR summaries, not
  per-commit branch history.
- **The `commit-msg` git hook.** The sibling repo has a Python-specific `.githooks/commit-msg`
  hook that auto-appends `[ADR-XXX src/file.py]` tags to commits on `feature/adr-XXX-*` branches,
  sourcing the file list from the Python source tree. That mechanism is source-language-specific
  and was not ported. In this repo, ADR traceability instead relies on the branch-naming
  convention (§2) and the PR description referencing the ADR number directly — there is no
  automated commit tagging. If this becomes a pain point, a Rust-source-aware equivalent could be
  written, but none exists today.
