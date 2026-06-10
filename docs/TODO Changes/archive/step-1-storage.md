# COMPLETED at 5/13/26

# Step 1 — Storage & Traits (`storage.rs`) ✅

**Status:** Complete

## Goal
Define the `StorageClient` trait and implement `LocalStorageClient` for the local filesystem.

## Files Produced
- `src/storage.rs`
- `src/error.rs`
- `Cargo.toml` (full dependency graph for all steps)

## What Was Built

### `StorageClient` trait
Object-safe trait replacing Python's `BaseStorageClient(ABC)`. Methods:
- `get_file_bytes(&self, file_uri: &str) -> Result<Bytes, IndexerError>`
- `get_metadata(&self, file_uri: &str) -> Result<FileMetadata, IndexerError>`
- `list_files<'a>(&'a self, root_uri: &'a str) -> Box<dyn Iterator<Item = Result<String, IndexerError>> + 'a>`

### `FileMetadata` struct
Typed replacement for Python's untyped `dict`. Fields: `file_uri`, `name`, `mime_type`, `size_bytes`, `modified_at`.

### `LocalStorageClient`
Stack-based directory walker (`LocalWalker`) replicating `os.walk(topdown=True)` pruning semantics. Enforces:
- `MAX_FILE_SIZE` (50 MiB) check **before** allocation
- `IGNORED_DIRS`, `IGNORED_EXTS`, `IGNORED_NAMES`, `IGNORED_NAME_PREFIXES`
- Conditional ignore (`functions/` only allowed under `firebase/`)

### `IndexerError` enum
`thiserror`-derived error type with variants: `Io`, `FileTooLarge`, `NotFound`, `UnsupportedMime`, `Parse`, `Database`, `Embedding`, `VectorStore`, `Other`.

## Python → Rust Paradigm Shifts
| Python | Rust |
|---|---|
| `BaseStorageClient(ABC)` | `StorageClient` trait |
| `io.BytesIO` | `bytes::Bytes` (ref-counted, zero-copy) |
| `dict` return | `FileMetadata` struct |
| `try/except Exception: pass` | `Result<T, IndexerError>` + `?` |
| Size check after allocation | Size check before `fs::read()` |

## Tests
5 unit tests — all passing:
- `list_files_yields_only_allowed`
- `get_file_bytes_reads_content`
- `get_file_bytes_rejects_oversized`
- `get_metadata_detects_mime_type`
- `mime_for_extension_covers_all_code_types`
