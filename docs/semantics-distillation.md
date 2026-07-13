# Semantics Distillation

The learn pipeline converts recurring interpretation choices into reusable,
auditable skills. It is intentionally selective: unresolved templates stay on
the model path, while recurring conventions that clear the evidence gates can
be executed deterministically and supplied back to the model as concise
guidance.

## Inputs and Outputs

Inputs:

- public task questions and output guidelines;
- public DABStep context files and documentation;
- optional benchmark-published dev answers;
- model-generated reference labels created during the local learn run;
- optional solver-filed template proposals.

Outputs under the ignored `artifacts/skills/` directory:

- `skill_*.json`: executable declarative skill artifacts;
- `note_*.json`: advisory conventions that led their discrimination but did
  not clear the deterministic adoption gate;
- optional `learned_conventions.md`: an LLM-readable schema-level digest
  rendered from the adopted artifacts;
- `_learn_summary.json`: template funnel, evidence, audit, and cost summary;
- resumable references, proposal queues, and candidate-matrix caches.

No generated artifact is committed to the public repository.

## Template Funnel

For the public 450-task suite, exact normalized induction currently produces
104 question templates. The default `--min-instances 5` gate admits 28
recurring templates into the main learn pass. This concentration is the main
cost advantage: the system learns recurring semantics instead of treating all
450 instances as unrelated model calls.

Thin templates are not discarded from runtime capability. They remain on the
LLM path and can later receive priority through solver-filed proposals or
family-level evidence.

## Pipeline

### 1. Induce Templates

`distill/templates.py` normalizes instance-specific literals into
placeholders and groups equivalent question shapes. This stage requires no
model calls.

### 2. Compile Signatures

`distill/signatures.py` creates a matcher and parameter extractor for each
eligible template. Templates that cannot be represented safely are skipped
instead of receiving a permissive matcher.

### 3. Enumerate Interpretations

`distill/hypotheses.py`, `spec.py`, and `combinators.py` build declarative
interpretation candidates over a restricted DSL. A zero-token grid covers
standard semantic axes first. Teacher proposals can extend the grid, but they
do not receive special trust.

### 4. Execute the Candidate Matrix

Every candidate runs on sampled real instances through the shared mechanism
layer. The resulting matrix reveals which instances actually distinguish
rivals. Matrix results are cached by template, candidate set, data
fingerprints, and code version.

If all candidates are extensionally equal on the available instances, the
pipeline records that fact rather than spending model tokens on labels that
cannot separate them.

### 5. Acquire Evidence

The pipeline supports three evidence strategies:

| Strategy | Behavior |
| --- | --- |
| Full | Solve each planned instance once, aggregate across the template, and escalate only unresolved top-rival disagreements. |
| Budget | Label the most informative disagreement points, cluster equivalent numeric answers, and stop sequentially when the decision is stable. |
| File | Reuse an external model-reference file for deterministic reruns. |

References persist immediately. An interrupted run can continue with
`--resume` without repurchasing completed evidence.

### 6. Discriminate and Adopt

`distill/discriminate.py` ranks candidates by agreement on participating
instances. Full mode applies:

- a minimum agreement floor;
- a required lead over the best distinguishable rival;
- an exact binomial evidence test;
- minimum participation;
- targeted escalation when the initial evidence is ambiguous.

Budget mode uses its configured consensus and early-stopping gates. A leading
candidate that lacks sufficient evidence can become an advisory note instead
of an executable skill.

### 7. Harden

`distill/harden.py` adds invariants, distinguishability evidence, regression
anchors, and provenance. Skill JSON stores declarative semantics rather than
unrestricted generated Python.

### 8. Audit

Every `dabstep-agent learn` run invokes `scripts/skill_audit.py` when the
repository audit tooling is present. For families overlapping the independent
calibration oracle, the learned skill is recompiled and compared on real
instances; contradictory artifacts are removed.

Skills outside oracle coverage still need the normal discrimination,
participation, invariant, and provenance gates. Oracle overlap is an additional
veto, not a claim that every possible template is hand-labeled.

### 9. Render for Model Consumption

`distill/emit.py` can render `learned_conventions.md` from the same
artifacts; the compatibility uploader exposes this through
`dabstep-upload-manual --skills-digest artifacts/skills`. Each entry contains
the canonical question shape, adopted convention, and evidence summary. It
contains no task ids, entities, or answer values and can be frozen into
MemoryLake as compact reusable guidance.

## Agent-Directed Proposals

During runtime the model can propose that a recurring uncertainty deserves a
skill. Proposal records identify the normalized public template and a
schema-level reason. Running:

```bash
dabstep-agent learn --tasks tasks.json --data-dir context \
  --official-dev tasks_dev.json --output artifacts/skills \
  --from-proposals artifacts/skills/_proposals.jsonl --resume
```

prioritizes those templates while preserving exactly the same evidence and
audit gates. The agent chooses where learning effort is valuable; the
certification pipeline decides whether the result is trustworthy.

## Cost Controls

- zero-token template induction, signature compilation, and candidate execution;
- disagreement-driven reference acquisition;
- candidate-matrix caching;
- immediate reference persistence;
- concurrent template and reference workers;
- bounded escalation;
- proposal-scoped learning;
- family-level evidence pooling for compatible thin templates;
- resumable completed skills and notes.

The default release command uses full mode because the validator needs a
complete local learn run. For development, budget mode or
`--max-templates <n>` provides a faster end-to-end smoke test.

## Provenance Boundary

Accepted or golden answers are isolated to scoring after inference. They do not
enter the learn pipeline, candidate matrix, model prompts, MemoryLake,
artifacts, or runtime. `scripts/release_audit.py` and the official-safety test
suite enforce the tracked-tree boundary.
