# CLAUDE.md

This file provides repository guidance to coding agents.

## Project

DABStep MemoryLake Agent combines a proven Pydantic AI benchmark runtime with
an AI-native semantic compiler and a reproducible skill-learning pipeline.
Models choose semantic intent, tools, and reusable-skill proposals;
deterministic components compile, execute, verify, and trace those choices.
MemoryLake stores versioned public knowledge and learned convention digests.

## Setup and Tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

pytest -q
pytest tests/test_workflow.py -q
pytest tests/test_workflow.py::test_name -q
```

Python 3.11 through 3.13 is supported.

## Four-Stage CLI

`dabstep-agent` supports the public `learn`, `freeze`, `run`, and
`report` subcommands. The older standalone entry points remain compatibility
aliases.

```bash
dabstep-download-data

dabstep-agent learn --tasks tasks.json --data-dir context \
  --official-dev tasks_dev.json --output artifacts/skills

dabstep-agent freeze \
  --docs data/context/manual.md data/context/payments-readme.md

dabstep-agent run --input tasks.json --data-dir context \
  --output results/run.jsonl --run-mode memory-assisted \
  --disable-memory-writes

dabstep-agent report --run results/run.jsonl \
  --submission results/submission.jsonl
```

Bare `tasks.json`, `tasks_dev.json`, and `context` paths resolve against
the downloader's `data/` directory. A successful freeze writes the ignored
`artifacts/freeze_state.json`; later runs recover the MemoryLake project id
with precedence CLI > environment > freeze state.

## Runtime Architecture

`runner.py` orchestrates retrieval and task concurrency.
`workflow.py` contains the established Pydantic AI graph.
`semantic_workflow.py` wraps it with five promotion modes:

- `legacy`: established runtime, the default;
- `shadow`: return legacy while retaining semantic traces;
- `candidate` and `primary`: verified semantic answer with legacy fallback;
- `strict`: certified semantic answer only.

The semantic path is:

1. deterministic family routing;
2. family-scoped compact intent planning;
3. deterministic intent-to-`AnalysisSpec` compilation;
4. bounded rival construction and execution;
5. candidate-level verification;
6. traced semantic answer or mode-appropriate fallback.

Keep fallback behavior explicit. Planner, compiler, executor, verifier, and
MemoryLake failures must remain visible in traces and must not silently return
an unverified semantic answer.

## Learn Pipeline

`src/dabstep_agent_pydantic/distill/learn.py` orchestrates:

1. normalized template induction;
2. safe signature compilation;
3. grid-first declarative candidate generation;
4. cached candidate-matrix execution;
5. full, budget, or file-based reference acquisition;
6. relative evidence gates and targeted escalation;
7. invariants and provenance hardening;
8. overlap calibration audit;
9. executable skill emission and optional LLM-digest rendering.

Use `--resume` for any nontrivial run. Keep reference concurrency in the
6-12 range unless the endpoint is known to sustain more. Use
`--max-templates` only for smoke tests; release validation should run the
complete default-eligible set.

Solver-filed proposals are accepted through `--from-proposals`. A proposal
may prioritize a public template but must never bypass discrimination or audit.

## Generated Assets

The following are local-only and ignored:

- `data/`;
- `artifacts/`, including generated skills and freeze state;
- `results/`;
- `workspace/`;
- references, candidate matrices, model output, submissions, and caches.

Do not add generated artifacts to Git, even for demonstrations. Run
`python scripts/release_audit.py` before public commits.

## Evaluation Boundary

Accepted answers, golden answers, pseudo-gold mappings, task scores, and
submission-derived answer maps are allowed only in isolated post-inference
scoring. They must not enter source, tests, assets, prompts, skills,
MemoryLake, planning, execution, or verification.

Keep clean, memory-assisted, shadow, and post-hoc scoring modes visibly
separated. Official-style runs use MemoryLake read-only with
`--disable-memory-writes`.

## Verification

Before release:

```bash
python scripts/release_audit.py
pytest tests/test_release_audit.py tests/test_official_safety.py -q
pytest -q
uv build --wheel
```

The public branch must have one root commit and must preserve the repository's
existing Apache-2.0 license verbatim.
