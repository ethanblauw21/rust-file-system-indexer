# ADR-006: CUDA Execution Provider for the Embedding Phase (opt-in, throughput-gated)

**Status:** Proposed
**Date:** 2026-07-08
**Deciders:** Ethan
**Depends on:** none
**Required by:** none yet

## Context

Indexing is slow. The suspected bottleneck is CPU ONNX inference in the embedding
phase (`Embedder::embed`, a single `Mutex<Session>`, `EMBED_BATCH_SIZE = 32` tuned
explicitly for CPU to keep each call under ~5 s — see `src/indexer.rs:27-30`). The
target machine has a discrete **NVIDIA RTX PRO 1000 Blackwell Laptop GPU (8 GB
VRAM)** and 64 GB RAM (confirmed via `nvidia-smi`, driver 596.08), so the ONNX
Runtime CUDA execution provider is available and is expected to give a large
embedding speedup (10–30× range for the embedding phase specifically — **to be
measured, not assumed**).

This is a **pure performance change**: identical embedding model
(`nomic-embed-text-v1.5`, Matryoshka 256-dim slice), identical vectors, identical
fusion math. It therefore does **NOT** go through the ranking/mutation sweep
harness — that harness gates *ranking* changes, and this changes none. It is
instead gated on a **before/after throughput measurement** on a fixed corpus. No
claim of speedup ships without both numbers.

**Settled constraints (encoded, not to be relitigated):**
1. CUDA is opt-in via a Cargo feature (`cuda`), never the default. The CPU ONNX
   path stays the portable default so the tool still builds/runs (and CI passes)
   on machines with no NVIDIA GPU / CUDA toolkit.
2. Measurement gates the change: CPU baseline profiled FIRST, CUDA measured on the
   SAME corpus, both numbers recorded.
3. The corpus must be large enough to reach thermal/power steady state (laptop
   Blackwell throttles under sustained load; a 30 s microbenchmark flatters
   itself). Plugged into wall power.
4. The crash-safe write ordering (serial SQLite→LanceDB two-phase commit,
   `lance_id IS NULL` recovery signal) is load-bearing and stays exactly as-is.
   This speeds up embedding, not persistence.
5. The embedding model is not swapped — a different/smaller model is a *quality*
   change that goes through the ranking harness and is out of scope here.

**Phase-0 gate (this ADR is conditional on it):** a per-phase profile of the CPU
baseline must first confirm that embedding actually dominates. If the profile
shows the wall is elsewhere (LanceDB writes, IVF rebuild, chunking), the CUDA work
buys little and this ADR is redirected/withdrawn rather than implemented. The
profile decides.

## Decision

*(Proposed — conditional on the Phase-0 profile confirming embedding dominates.)*

1. Add a `cuda` Cargo feature that registers the `ort` CUDA execution provider on
   the `SessionBuilder` in `Embedder::load`, falling back to the current CPU
   provider when the feature is off. Document the CUDA toolkit + cuDNN versions
   matched to `ort 2.0.0-rc.12` next to the existing ORT DLL setup instructions.
2. Keep the existing `Embedder::load` hard timeout: a CUDA EP that fails to
   initialize must surface a named `IndexerError::Embedding` (missing CUDA/cuDNN,
   version mismatch) within the timeout, never wedge.
3. Re-tune the embedding batch size for the GPU (32 is CPU-tuned and starves the
   GPU). Sweep {32,128,256,512}, pick the throughput-optimal value within 8 GB
   VRAM, and wire it through the **existing** config/CLI precedence
   (`[indexer].embed_batch_size` / `--embed-batch-size`) — no new ad-hoc knob.

## Evidence

*(Throughput measurement, not a ranking sweep. Filled during implementation.)*

**Corpus:** the reproducible eval union under `tools/corpus/.cache` — public
`codebase-indexer` @ pinned SHA (target) + 5 pinned distractor repos (`click`,
`zustand`, `p-queue`, `serilog`, `spdlog`), ~1,126 files (`tools/corpus/manifest.toml`).

**Phase-0 CPU baseline (git SHA `f399acc`, CPU EP, `EMBED_BATCH_SIZE=32`, `index --profile --reindex`, plugged into wall power):**

905 indexable files → 9,911 vectors. Wall-clock **4613 s (1h 16m)**, **2.1 chunks/sec**.
Cumulative time-in-stage (producer/consumer overlap, so stages need not sum to wall-clock):

| stage | seconds | % of summed stage time |
|---|---:|---:|
| **inference** | **4506.0** | **97.0%** |
| lance_write | 64.6 | 1.4% |
| chunking | 37.1 | 0.8% |
| tokenize | 27.5 | 0.6% |
| ivf_build | 6.2 | 0.1% |
| sqlite_write | 4.7 | 0.1% |
| summed | 4646.2 | — |

**Gate verdict: embedding dominates (inference = 97.0% of stage time, 97.7% of wall-clock).**
The single `Mutex<Session>` consumer is serialized on inference; summed stage time (4646 s)
≈ wall-clock (4613 s), so there is no producer/consumer overlap left to reclaim — the only
phase worth attacking is inference. Phase 1 (CUDA EP) is justified. Proceeding.

**CUDA path (git SHA `<pending>`, batch size `<pending>`, thermal steady state):**
_to be recorded — same per-phase breakdown + chunks/sec._

**Measured speedup:** _to be stated as a number, not an estimate._

## Consequences

**Better:** faster embedding phase on machines with an NVIDIA GPU; indexing large
corpora becomes materially cheaper in wall-clock. A `--profile` breakdown becomes
a permanent, reusable diagnostic for future perf work.

**Worse:** the `cuda` build acquires a CUDA toolkit + cuDNN dependency and a
version-matrix maintenance burden (mirroring the ORT DLL / Defect-1 experience).
Two build configurations to keep green.

**Neutral:** CPU path, vectors, fusion math, and the two-phase-commit write
ordering are unchanged. This change **bypasses the ranking harness by design** (no
ranking impact) and is gated on throughput instead — recorded here explicitly so
the divergence from the ranking-ADR discipline is auditable, not accidental.

## Alternatives Considered

| Option | Why rejected |
|--------|-------------|
| Make CUDA the default provider | Breaks the portable build and non-GPU CI; a CUDA toolkit is not universally present. Opt-in feature keeps CPU first-class. |
| Swap to a smaller/faster embedding model | That is a *quality* change (different vectors) and must go through the ranking sweep harness — out of scope for a pure-perf change. |
| Skip the Phase-0 profile and go straight to CUDA | Violates the measurement gate; risks optimizing a phase that isn't the bottleneck. The profile must confirm embedding dominates first. |
| Parallelize/overlap tokenization with inference now | Only justified if Phase-3 re-profiling shows tokenization became the new wall after the GPU speedup — deferred to a fresh measurement, not scoped in here. |

## Implementation Log

> Updated during development. Record deviations, surprises, and in-the-moment decisions.

- [x] Phase 0 — added per-phase timing (`--profile`, commit `f399acc`); measured CPU baseline on the 974/905-file corpus. **Embedding confirmed dominant: inference = 97.0% of stage time.** Gate cleared.
- [ ] Phase 1 — `cuda` Cargo feature + CUDA EP registration in `Embedder::load`; timeout still surfaces a named error; CPU/CUDA parity test (cosine ≈ 1.0).
- [ ] Phase 2 — batch-size sweep on CUDA; set throughput-optimal value via existing config/CLI knob.
- [ ] Phase 3 — re-profile CUDA path at steady state; report relocated bottleneck if any.
- [ ] Task 5 — record before/after throughput table; state measured speedup as a number.
- [ ] Confirm CPU-default build + `cargo test` still pass (non-GPU CI green).
- [ ] Resolve every downstream obligation in **Required by** before setting status to **Accepted** — none exist, so vacuously satisfied.

**Notes:**
<!-- 2026-07-08: ADR opened Proposed. Conditional on Phase-0 profile. -->
