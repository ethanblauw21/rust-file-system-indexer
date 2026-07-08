# `tools/baselines/`

`search_eval_baseline.jsonl` is the git-SHA-stamped, append-only (dedup-by-key)
output of `tools/eval_search.py run`. It intentionally starts **absent** in this
commit rather than seeded with a fabricated row — see `tools/README.md`
"Corpus provenance" for why the historical 4.69/12-of-16 number cannot be
mechanically reproduced here, and why the first real run of this harness
establishes a new baseline rather than replaying the old one.

The file is created automatically by the first `python tools/eval_search.py run`
invocation. Do not hand-edit it — use `eval_common.append_baseline` (via
`eval_search.py`) so the dedupe-by-key and sort-for-stable-diffs invariants hold.
