# Vector store: LanceDB vs FAISS

Moved from CLAUDE.md, verbatim except the line-number fixes noted in the PR. Read when working on this area. Line-number citations drift; re-check them against the code.

### Vector Store: LanceDB vs. FAISS

LanceDB is the correct choice for this use case: it stores vectors on disk (no RAM footprint at rest), integrates directly with Arrow/Lance columnar format, and requires zero infrastructure. FAISS is in-process and memory-mapped — it would require `IndexIVFPQ` with `OnDiskInvertedLists` to avoid loading the full index into RAM, and still needs manual serialization.

If migrating to FAISS for finer control: use `IndexIVFPQ` (Product Quantization cuts memory 8–16×), back it with `OnDiskInvertedLists` for memory-mapping, and set `nprobe = max(1, √nlist)` at query time. The `stable_id` scheme and SQLite metadata layer are FAISS-agnostic and would carry over unchanged.
