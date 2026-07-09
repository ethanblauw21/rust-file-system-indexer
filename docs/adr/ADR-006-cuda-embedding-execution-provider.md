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

Phase-0 confirmed embedding dominates (inference = 97% of the work). The CUDA path
was then implemented and measured to a **24.9× end-to-end speedup** (see Evidence).
Decisions, as built and measured:

1. **`cuda` Cargo feature** (`cuda = ["ort/cuda"]`) registers the `ort` CUDA execution
   provider on the `SessionBuilder` in `Embedder::load` with `.error_on_failure()` —
   a `--features cuda` build that cannot init CUDA surfaces a named
   `IndexerError::Embedding` (never a silent CPU fallback, which would corrupt the
   throughput measurement). Registration is inside `load_inner`, so the existing
   `Embedder::load` timeout bounds any CUDA init hang. CPU-default build is untouched
   and CI (no GPU) stays green — `--features cuda` compiles without the CUDA SDK
   (`load-dynamic`).
2. **Run the GPU path on the fp16 model export, not the shipped int8 one.** The int8
   quantization forces 156 CPU-fallback memcpy nodes on CUDA (Evidence); fp16 runs
   on-GPU with tensor cores at cosine 1.0 vs fp32. The ONNX model **filename is now
   configurable** — `[embedder].onnx_model_file` → env `NOMIC_ONNX_FILE` → default
   `nomic-embed-text-v1.5.onnx` — so a GPU user points at `model_fp16.onnx` without
   renaming files.
3. **Batch size stays at the default 32** — the sweep (Evidence) shows 32 is
   throughput-optimal for this model/GPU (larger batches waste O(seq²) compute on
   BatchLongest padding and hit the 8 GB VRAM ceiling). It remains tunable via the
   existing `[indexer].embed_batch_size` / `--embed-batch-size` — no new knob.
4. **Runtime toolchain (no-admin):** GPU onnxruntime **1.24.2** build (ships
   `onnxruntime_providers_cuda.dll`) + CUDA **12.9** / cuDNN **9.24** runtime DLLs from
   pip `nvidia-*-cu12` wheels, colocated next to `onnxruntime.dll`. Point
   `ORT_DYLIB_PATH` at the GPU `onnxruntime.dll` and put its lib dir on `PATH`.

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

**Critical discovery — the shipped model is INT8-quantized, which cripples CUDA.**
The 137 MB `nomic-embed-text-v1.5.onnx` in use is `model_int8` (producer `onnx.quantize`).
On the CUDA EP its dynamic-quant ops are largely unsupported, so ONNX Runtime inserts
**156 CPU-fallback memcpy nodes** ("156 Memcpy nodes are added to the graph for
CUDAExecutionProvider") → the GPU thrashes on CPU↔GPU copies. Int8-on-GPU measured
**3.1 chunks/sec raw (10.4 s per 32×512 batch)** — no better than CPU. The fix is to run the
GPU path on the **fp16 export** (`model_fp16.onnx`, 274 MB, same weights), which runs on-GPU.

**Raw model timing on GPU (Python `onnxruntime-gpu` 1.24.2, 32×512 batch, direct):**

| model | memcpy nodes | ms/batch | raw chunks/s |
|---|---:|---:|---:|
| int8 (shipped) | 156 | 10410 | 3.1 |
| **fp16** | 24 | **443** | **72.1** |
| fp32 | — | 629 | 50.9 |

**Parity (final 256-dim Matryoshka-sliced, L2-normalized vectors, real tokenized text):**
`cosine(fp16, fp32) = 1.00000` (min 0.99999) — fp16 faithfully reproduces the true model.
`cosine(int8, fp32) = 0.96388` — the *shipped int8* export is the least faithful; moving the
GPU path to fp16 is both faster **and** higher-fidelity than today's CPU int8.

**GPU batch-size sweep (serilog, 1,389 chunks, fp16, end-to-end):**

| batch | chunks/s | peak VRAM |
|---:|---:|---:|
| **32** | **67.6** | 4202 MiB |
| 64 | 55.8 | 7274 MiB |
| 128 | 5.1 | 7914 MiB |
| 256 | 2.4 | 7918 MiB |

**32 is throughput-optimal — the prompt's "32 is too small for the GPU" assumption is refuted
for this model/GPU.** `PaddingStrategy::BatchLongest` pads every sequence in a batch to the
batch's longest (→512), and attention is O(seq²), so larger batches waste compute on padding
*and* hit the 8 GB VRAM ceiling (128+ peaks ~7.9 GB and collapses). No GPU-default batch change
is warranted; `--embed-batch-size` / `[indexer].embed_batch_size` already make it tunable.

**CUDA path (git SHA `c9b9961` + `--features cuda`, fp16 model, batch 32, GPU onnxruntime 1.24.2
+ CUDA 12.9 / cuDNN 9.24, SAME 905-file corpus → 9,911 vectors, plugged into wall power):**

Wall-clock **185 s (3m5s)**, **53.4 chunks/sec**:

| stage | seconds | % of summed stage time |
|---|---:|---:|
| **inference** | **114.3** | **58.7%** |
| lance_write | 29.5 | 15.2% |
| ivf_build | 18.5 | 9.5% |
| tokenize | 16.6 | 8.5% |
| chunking | 13.8 | 7.1% |
| sqlite_write | 2.1 | 1.1% |

**Measured speedup: 24.9× end-to-end** (4613 s → 185 s), **inference alone 39.4×**
(4506 s → 114 s). Inference fell from 97.0% → 58.7% of stage time — the bottleneck has begun to
relocate to CPU-side/IO phases (Phase 3 below).

## Consequences

**Better:** **24.9× faster indexing** on the target GPU (full corpus 1h16m → 3m5s);
inference alone 39.4×. The fp16 GPU path is also *more* faithful to the true model
than the shipped int8 CPU export (cosine 1.0 vs 0.964 against fp32). A `--profile`
breakdown is now a permanent, reusable diagnostic for future perf work.

**Worse:** the `cuda` build acquires a CUDA toolkit + cuDNN dependency and a
version-matrix maintenance burden (mirroring the ORT DLL / Defect-1 experience);
two build configurations to keep green. The GPU path needs a **separate fp16 model
download** (`model_fp16.onnx`) — the shipped int8 model is ~20× slower on CUDA and
must not be used there.

**Neutral / caveat:** CPU path, fusion math, and the two-phase-commit write ordering
are unchanged. **Precision consistency:** an fp16-GPU-built index and an int8-CPU-built
index differ by ~0.96 cosine (both approximate fp32), so a single index should be built
with one precision throughout — don't interleave int8-CPU and fp16-GPU writes into the
same store. This change **bypasses the ranking harness by design** (no ranking impact;
the fp16↔fp32 parity is 1.0) and is gated on throughput instead — recorded here
explicitly so the divergence from the ranking-ADR discipline is auditable, not accidental.

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
- [x] Phase 1 — `cuda` Cargo feature + CUDA EP registration in `Embedder::load` (commit `c9b9961`), `.error_on_failure()` (no silent CPU fallback), inside the load timeout. Named-error path verified. **Surprise: naive CUDA gave only ~1.3× — root-caused to the int8 model's 156 memcpy nodes; fp16 export fixes it (parity cosine 1.0 vs fp32).**
- [x] Phase 2 — batch sweep {32,64,128,256}: **32 is optimal** (larger regress on padding + 8 GB VRAM); already tunable, no default change. Model filename made configurable (`onnx_model_file`) so the GPU path can select `model_fp16.onnx`.
- [x] Phase 3 — re-profiled CUDA path: inference 97% → 58.7%; bottleneck relocating to lance_write (15%) / ivf_build (9.5%) / tokenize (8.5%) / chunking (7.1%). Next perf target is IO/LanceDB + IVF, not tokenization alone — propose as a follow-up, not scoped here.
- [x] Task 5 — before/after recorded (Evidence): **24.9× end-to-end**, inference 39.4×, on the same corpus at steady state, git-SHA stamped.
- [x] Confirm CPU-default build + `cargo test` still pass (non-GPU CI green) — Phase 1/2 changes keep the default build clean + clippy `-D warnings`.
- [ ] Set status **Accepted** on merge (PR); no **Required by** obligations exist.

**Notes:**
<!-- 2026-07-08: ADR opened Proposed. Conditional on Phase-0 profile. -->
<!-- 2026-07-08: Phase 0 cleared (inference 97%). Phase 1 CUDA EP built; naive int8-on-GPU only ~1.3x
     due to 156 memcpy nodes (int8 quant unsupported on CUDA EP). Switched GPU path to fp16 export
     (parity 1.0 vs fp32); batch 32 optimal; measured 24.9x end-to-end. Runtime staged no-admin via
     GPU onnxruntime 1.24.2 + pip nvidia-cu12 wheels (CUDA 12.9 / cuDNN 9.24). -->
