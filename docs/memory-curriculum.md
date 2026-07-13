# Official-safe memory curriculum

The official DABStep runs retrieve MemoryLake memories read-only. This document
describes where those memories come from and how the set is frozen and audited,
so the provenance chain is verifiable end to end:

> public docs → verified semantic rules → sanitizer → frozen MemoryLake project → read-only retrieval

Benchmark tasks and answers are never part of that chain.

## 1. Distill (curriculum pass)

The curriculum agent studies only the public context documentation
(`manual.md`, `payments-readme.md`), verifies non-obvious semantics with Python
experiments against the dataset, and proposes schema-level memories
(metric definitions, wildcard matching semantics, output contracts, data-quality
caveats). It never sees a benchmark task.

```bash
PYTHONPATH=src python -m dabstep_agent_pydantic.curriculum \
  --data-dir <context-dir> \
  --output results/curriculum_memories.jsonl \
  --dry-run
```

`--dry-run` produces the candidate file without writing anything, so the set can
be reviewed first. Every candidate passes a sanitizer that rejects:

- references to real business entities (merchant names loaded from the dataset),
- references to benchmark tasks (`task <id>` patterns),
- answer-like computed values (high-precision decimals),
- fragments, duplicates, and unknown categories.

Accepted and rejected candidates are both recorded in the output JSONL with
their status and rejection reason.

## 2. Write and freeze

Re-run without `--dry-run` to write the accepted memories into the dedicated
official project (verbatim, `infer=false`, metadata
`asset_type=curriculum, source=public_docs, official_safe=true, contains_answer=false`):

```bash
PYTHONPATH=src python -m dabstep_agent_pydantic.curriculum \
  --data-dir <context-dir> \
  --memorylake-project-id "$MEMORYLAKE_PROJECT_ID" \
  --memorylake-user-id "$MEMORYLAKE_USER_ID" \
  --output results/curriculum_memories.jsonl
```

Then snapshot the project and record the content hash:

```bash
PYTHONPATH=src python -m dabstep_agent_pydantic.memory_export \
  --memorylake-project-id "$MEMORYLAKE_PROJECT_ID" \
  --memorylake-user-id "$MEMORYLAKE_USER_ID" \
  --output results/memory_export.jsonl
```

The reported `content_sha256` pins exactly what the agent can retrieve. Export
again after the official run and compare hashes to prove the memory set did not
change mid-run (official runs use `--disable-memory-writes`, so it cannot).

## 3. Audit

`tests/test_official_safety.py` scans both artifacts
(`results/curriculum_memories.jsonl`, `results/memory_export.jsonl`) for task
references and answer-like values, and — with `DABSTEP_CONTEXT_DIR` set — for
real merchant names:

```bash
DABSTEP_CONTEXT_DIR=<context-dir> PYTHONPATH=src python -m pytest tests/test_official_safety.py -q
```

## Project separation

- `dabstep-official`: curriculum memories only; frozen before official runs;
  runtime access is read-only (`--disable-memory-writes`).
- demo projects (learning-curve experiments, writes enabled) use a different
  project id and are never referenced by official runs.
