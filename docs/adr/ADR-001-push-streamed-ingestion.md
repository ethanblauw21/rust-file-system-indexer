# ADR-001: Push / Streamed Ingestion — a producer-driven ingress alongside the pull walker

**Status:** Proposed
**Date:** 2026-06-22
**Deciders:** Ethan
**Depends on:** homelab#ADR-022 (audit-db projection and semantic-history feed — the first concrete consumer that needs to stream in-memory, metadata-tagged records this indexer cannot reach by walking a directory)
**Required by:** — none yet —

## Context

Every ingestion path in this crate is **pull-based**: the indexer is given a *root directory* and walks it.

- `IncrementalIndexer::index_root(root_uri, …)` (`indexer.rs:634`) lists files via the `StorageClient` (`LocalStorageClient`'s stack-based DFS, `storage.rs`) and feeds each URI through `process_file_sync` (`indexer.rs:533`).
- `index_uris` (`indexer.rs:818`) and the `recheck` command take an explicit URI list, but still **read content from storage** — they call `process_file_sync`, which calls `storage.get_metadata` + `storage.get_file_bytes`.

So the contract today is: *the bytes must already be a file the walker can stat and read.* There is no way for an external producer to hand the indexer content it **already holds in memory**, with **metadata the filesystem cannot express**.

A consumer now needs exactly that. The `homelabMCPServer` wants to stream **change-events** — a redacted unified diff plus a structured record `{tool, pre_hash, post_hash, vmid, path, ts}` — into this index for hybrid (BM25 + dense) recall, and to get that record **back on every search hit** so a semantic match pivots to the exact audit row. That content:

1. **Is not a file on any disk this indexer can walk.** It is generated in-process by another program. Forcing it through the filesystem means writing throwaway temp files purely to be re-read — wasteful and racy.
2. **Carries producer metadata that must survive to query time.** The filesystem has nowhere to put `{tool, pre_hash, …}`; the walker would discard it.

Crucially, **the backend already supports everything except the ingress.** The expensive, proven machinery is content-source-agnostic:

- `FileChunker::chunk(data, meta, map)` (`chunker.rs:1184`) is **pure** — bytes + a `FileMetadata` in, `ChunkResult` out, no I/O. MIME dispatch (`resolve_parser`, `chunker.rs:1158`) already routes any `text/*` type to `parse_plaintext`, so a diff pushed as `text/x-diff` or `text/plain` chunks correctly with zero new parser.
- `write_file` (`indexer.rs:754`) → `flush_embeddings` (`indexer.rs:860`) does the SQLite upsert, chunk insert, embed batch, LanceDB add, and the `lance_id IS NULL` two-phase-commit crash signal — none of it cares where the bytes came from.
- The `chunks.meta TEXT` column (`db.rs:150`) **already round-trips arbitrary JSON** to query results: `ChunkRow.meta` (`db.rs:30`) and `FtsResult.meta` (`db.rs:54`) are returned on both search backends. The storage slot for `{tool, pre_hash, …}` exists and is already plumbed to the consumer — it is simply never *populated by a caller* today.

The gap is narrow and worth closing once, cleanly: **a programmatic ingress that accepts pre-fetched content + caller metadata and reuses the existing pipeline.**

## Decision

Add a **push ingestion path** as a second ingress alongside the pull walker. It does **not** replace or modify `index_root`; it shares the pipeline from chunking onward.

### Shape

A record type and a batch entrypoint:

```rust
pub struct IngestRecord {
    pub uri:         String,            // caller-chosen identity, e.g. "change://105/etc/nginx.conf@2026-06-22T18:03:01Z"
    pub content:     Vec<u8>,           // bytes the caller already holds (a redacted diff, a rendered config, …)
    pub mime_type:   String,            // explicit — there is no extension to dispatch on; use a text/* type
    pub modified_at: Option<f64>,       // caller's logical timestamp; defaults to now if absent
    pub meta:        serde_json::Value, // record-level metadata, merged onto every chunk's `meta`
}

impl IncrementalIndexer {
    pub async fn index_records(&self, records: &[IngestRecord]) -> Result<Stats, IndexerError>;
}
```

`index_records` builds a synthetic `FileMetadata` (`storage.rs:62` — `file_uri`/`name`/`mime_type`/`size_bytes`/`modified_at`) from each record **without touching `StorageClient`**, computes the content hash inline (the same `md5_hex` the pull path uses), then routes through the **existing** `FileChunker::chunk` and a **meta-aware `write_file`** into `flush_embeddings`. It reuses the same `EMBED_BATCH_SIZE` buffering, the LanceDB add, and the two-phase commit verbatim.

Two pull-path behaviors it must replicate, not just inherit: (1) **stale-vector eviction on re-push** — a re-pushed URI's old LanceDB vectors are removed (`get_lance_ids_for_file` + `vectors.remove_ids`, `indexer.rs:692`) *before* `write_file`, exactly as `index_root`/`index_uris` do; `write_file` alone clears only SQLite chunks, so skipping this would orphan the prior vectors (the ghost-vector class of bug). (2) **IVF-PQ rebuild** — after the final flush, `index_records` rebuilds the ANN index with the dynamic `compute_nlist` partition count (`indexer.rs:730`), so a growing feed keeps correct recall. (`index_uris` skips this because `recheck` touches few files; a push feed accumulates, so the rebuild is required, not optional.)

### The one real backend change: thread caller metadata onto chunks

Today `write_file` sets `ChunkInput.meta = c.meta.clone()` (`indexer.rs:788`) — chunker-derived metadata only. The push path must **merge** the record-level `meta` into each chunk's `meta` so `{tool, pre_hash, …}` lands in `chunks.meta` and returns via `ChunkRow.meta`/`FtsResult.meta`. Concretely, factor the chunk-write half of `write_file` to take an optional `record_meta: &serde_json::Value` and shallow-merge it under a reserved key (e.g. `meta["record"] = record_meta`) so it never collides with chunker-emitted keys. **This is additive:** the pull path passes `Value::Null`/empty and behaves byte-for-byte as before — the existing `index_root` tests (`indexer.rs:1037`) stay green.

### Recommended transport: an `ingest` CLI subcommand reading NDJSON from stdin

`index_records` is the library primitive. Expose it as a `clap` subcommand (`main.rs`) that reads **one `IngestRecord` per line as NDJSON from stdin** and batches into the indexer:

```
file_indexer ingest [--index-dir <dir>] [--batch <n>]   # records on stdin, one JSON object per line
```

On the wire, each line's `content` is a **UTF-8 string**, not a JSON byte array — these payloads are `text/*` (redacted diffs, rendered configs, command text), and a string is what a Node/shell producer emits naturally. The subcommand decodes it to the `Vec<u8>` that `IngestRecord.content` holds in memory, so the library type is unchanged; only the wire DTO fixes the encoding. (See *Potential extensions* for binary payloads.)

Rationale — this is the lowest-surface option that fits both repos:

- **Language-agnostic, zero new infrastructure.** The Node `homelabMCPServer` already shells out to other tools; it spawns `file_indexer ingest` and writes NDJSON to the child's stdin, exactly like its existing process integrations. No FFI, no Rust on the producer side.
- **No standing network surface.** A long-running socket/HTTP server was considered and **rejected**: it contradicts this project's "zero infrastructure" design and the consumer's own doctrine (its ADR-010 forbids standing network-reachable surfaces fronting the homelab). stdin streaming gives backpressure for free and dies with the process.
- **Composable.** `... | file_indexer ingest` works from any producer, not just the homelab — a generic producer-driven ingress.

The in-process `index_records` method remains available for any future in-tree caller.

## Scope boundaries

**In scope.** A programmatic, in-process ingress (`index_records`) and its NDJSON-over-stdin CLI front (`ingest`) that accept caller-held bytes plus caller metadata and reuse the pipeline from chunking onward, plus the one additive refactor that threads record-level `meta` onto each chunk.

**Out of scope — explicitly unchanged.** The `db.rs` schema, the embedder, LanceDB, the RRF search path, and the `StorageClient` trait are untouched. `index_root` and the pull walker keep their exact behavior; the existing pull tests (`indexer.rs:1037`) staying green is the proof the change is additive.

**Orthogonal, not addressed here.** The `GoogleDriveStorageClient` roadmap item is another *pull* backend — a new content source the walker can reach. This ADR opens the *push* axis instead; the two are independent and compose without interaction.

**Deliberately left to the feeder.** URI uniqueness and idempotency, history accumulation via the `@<ts>` URI suffix, a `source` discriminator in `meta`, and cross-source dedup are producer conventions this indexer does not enforce (see Honest limits).

**Potential extensions (deferred — built when a consumer needs them).** Recorded here so the chosen v1 shape does not silently foreclose them:

- **Binary push content.** The wire DTO fixes `content` as a UTF-8 string, sufficient for every consumer today (diffs, configs, command text). A binary producer (images, PDFs, archives) would add a base64-encoded `content` variant — or a length-framed transport — that decodes to the same `Vec<u8>` the in-memory `IngestRecord` already holds. The library primitive needs no change; this is a wire-format-only extension, and the MIME-dispatch path (`resolve_parser`) already routes non-`text/*` types to the binary/format parsers.
- **Incremental ANN rebuild.** `index_records` rebuilds the IVF-PQ index per invocation (correctness over cost). For a high-frequency feed batched into many small `index_records` calls this is rebuild-per-batch; the future optimization is to rebuild only when `ntotal` crosses a `compute_nlist` partition-count boundary (or on an explicit end-of-stream flush). Deferred — premature until feed volume measurably justifies it.
- **A long-running ingress.** A persistent socket/HTTP feeder was rejected for v1 (standing network surface; conflicts with homelab ADR-010). If an in-process, non-network transport is ever wanted (e.g. a Unix-domain socket or an FFI binding for an in-tree caller), it would sit on top of the same `index_records` primitive — the library boundary is drawn so the transport can change without touching the pipeline.

## Consequences

**Positive.**
- A second, producer-driven ingress that **reuses the entire proven pipeline** (chunk → embed → LanceDB → two-phase commit) with one small additive refactor. No new parser, no new store, no embedding-path change.
- Unlocks the homelab change-event feed and, more generally, ingestion of any content that is **not a walkable file** — rendered artifacts, API responses, event streams.
- The metadata passthrough turns the index into a **queryable event store**, not just a content index: a hit hands back the producer's structured record, enabling the pivot the consumer needs.
- The embedder is already optional (`flush_embeddings` no-ops when `NOMIC_ONNX_PATH` is unset, `indexer.rs:867`), so pushed records degrade gracefully to **FTS5/BM25-only** recall when no model is loaded.

**Negative / cost.**
- A new public surface (`index_records` + the `ingest` subcommand) plus the `write_file` meta-merge refactor (small, but it touches the hot write path — covered by the existing pull tests staying green).
- **Two identity regimes now coexist.** Walked files are deduped by `(file_uri, mtime → MD5)` change detection; pushed records bypass that (see limits). The feeder owns URI uniqueness and idempotency.
- NDJSON parsing + stdin backpressure on the `ingest` path; malformed lines need a defined policy (skip-and-count vs abort).

**Honest limits.**
- **Incremental change detection does not apply to pushed records.** `process_file_sync`'s mtime/MD5 skip (`indexer.rs:546`, `:559`) is a *pull* optimization keyed on re-walking the same path. A pushed record is **authoritative**: it is always chunked and embedded. Because `upsert_file` is `ON CONFLICT(file_uri) DO UPDATE` (`db.rs:331`), **re-pushing the same `uri` replaces** that record's chunks (and `delete_chunks_for_file` clears the old ones). So the URI scheme is load-bearing: change-events must use **unique** URIs (the `@<ts>` suffix) to *accumulate* history; reusing a URI is an intentional *overwrite*. The feeder owns this choice — the indexer cannot infer intent.
- **`stable_id` uniqueness rides on URI uniqueness.** `stable_id(file_uri, tier, chunk_index)` (`indexer.rs:43`) is a 60-bit hash of those three fields; distinct URIs give distinct LanceDB IDs. Colliding URIs across logically-distinct events would collide vectors — another reason the `@<ts>` scheme matters.
- **MIME must be a `text/*` type for diffs/configs.** `resolve_parser` sends anything not matching a known type and not starting with `text/` to `parse_binary_fallback` (`chunker.rs:1171`). The producer sets `mime_type` explicitly; `text/x-diff` or `text/plain` is correct. (Note: only Tier-1/2 chunks are FTS-indexed — the `chunks_ai` trigger excludes Tier-3, `db.rs:166` — but diffs are small and chunk into T1/T2, so they remain BM25-searchable.)
- **The index inherits the producer's trust status.** This crate is content-agnostic: it stores whatever bytes it is handed. If the producer pushes redacted diffs but the consumer *also* pull-indexes an unredacted mirror, the resulting store mixes both — a `source` discriminator in `meta` is a **feeder** convention, not something this indexer enforces. Dedup across sources likewise stays the feeder's responsibility (consistent with the existing "no cross-instance dedup" stance).

## Implementation notes

1. **`IngestRecord` + `index_records`** in `indexer.rs`: synthesize `FileMetadata`, `md5_hex(&record.content)`, reuse the `text_buffer`/`lance_id_buf`/`chunk_id_buf`/`tier_buf` accumulation loop and `flush_embeddings` as `index_uris` does (`indexer.rs:818`) — **plus** the re-push stale-vector eviction (`get_lance_ids_for_file` + `remove_ids`, `indexer.rs:692`) and a final `compute_nlist` IVF-PQ rebuild (`indexer.rs:730`), neither of which `index_uris` performs.
2. **Refactor `write_file`** to accept `record_meta: &serde_json::Value` and merge it under the reserved `"record"` key of each `ChunkInput.meta` (`indexer.rs:787`); the pull callers (`index_root`, `index_uris`) pass `&Value::Null`.
3. **`ingest` subcommand** in `main.rs`: deserialize NDJSON from stdin with `content` as a **UTF-8 string** (a thin wire DTO that decodes into `IngestRecord { content: Vec<u8>, .. }`), batch by `--batch`, call `index_records`, print a `Stats` line mirroring `run_index`'s summary. Define the malformed-line policy explicitly (skip-and-count vs abort).
4. **Tests:** an `index_records` unit test (no `NOMIC_ONNX_PATH`) asserting a pushed record produces FTS-searchable chunks whose `ChunkRow.meta` carries the record metadata; an NDJSON-parse test for the subcommand including the malformed-line policy. The existing `index_root_skips_unchanged_files` test must remain unchanged — proof the refactor is additive.

The unchanged-surface guarantee (`db.rs` schema, embedder, LanceDB, search, `StorageClient` trait) and the orthogonal pull/push axis distinction are stated once under **Scope boundaries** above.
