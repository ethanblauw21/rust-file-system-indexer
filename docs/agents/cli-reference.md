# CLI subcommands

Moved from CLAUDE.md, verbatim except the line-number fixes noted in the PR. Read when working on this area. Line-number citations drift; re-check them against the code.

### CLI subcommands (`src/main.rs`)

```bash
file_indexer index <root> [--reindex] [--exclude NAME]... [--no-embed] [--embed-batch-size N]
                                                              # incremental index; hard-errors if no
                                                              # embedder is configured unless --no-embed
file_indexer search <query> [--mode hybrid|dense|sparse] [--like PATH] [--tier N] [--type EXT] [--output table|json|md]
file_indexer tui                                            # interactive ratatui UI
file_indexer explain <path> [--all-chunks]                 # per-chunk breakdown for one file
file_indexer score [--rescore]                             # compute chunk quality scores
file_indexer scores [--flagged-only] [--tier N]            # display flagged chunks
file_indexer recheck [--dry-run]                           # reindex files whose chunker method drifted
file_indexer ingest [--batch N]                             # push pre-fetched NDJSON records via stdin
                                                              # (one JSON object per line: {uri, content,
                                                              # mime_type, modified_at?, meta?})
```
