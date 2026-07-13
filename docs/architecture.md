# Architecture

The agent has two cooperating runtimes and one shared knowledge plane:

- the established Pydantic AI runtime, which remains the default benchmark path;
- the semantic compiler runtime, where the model chooses a compact intent and
  deterministic components execute and verify it;
- MemoryLake, which serves versioned public documentation and reusable learned
  conventions to both paths.

This design gives the model agency over interpretation and tool use while
keeping data execution, output contracts, and provenance deterministic.

## End-to-End Phases

1. **Learn** induces recurring templates, executes rival interpretations, buys
   model evidence where candidates disagree, and emits certified skill
   artifacts that can also be rendered as an LLM-readable convention digest.
2. **Freeze** checks document fingerprints, uploads knowledge documents to
   MemoryLake, and persists the project id in
   `artifacts/freeze_state.json`.
3. **Run** retrieves frozen knowledge read-only, solves tasks through the
   selected semantic mode, and records complete traces.
4. **Report** exports the leaderboard schema and optionally compares isolated
   post-inference runs.

Downloaded data, learned skills, freeze state, model outputs, and submissions
are local generated assets and are excluded from the public repository.

## Runtime Selection

`SemanticMode` in `semantic_workflow.py` controls promotion:

| Mode | Semantics |
| --- | --- |
| `legacy` | Return the established runtime answer. |
| `shadow` | Return the established answer and retain the semantic candidate trace. |
| `candidate` | Return an accepted semantic answer; otherwise fall back. |
| `primary` | Semantic-first path with the same guarded fallback contract. |
| `strict` | Require a certified semantic answer and surface unresolved cases. |

The default is `legacy`, which preserves the proven benchmark runtime while
the semantic path is evaluated family by family.

## Established Pydantic AI Runtime

The established graph is not a flat prompt:

1. `PlanNode` creates a typed `PlanDecision` from the question, guidelines,
   route cards, file summary, and retrieved knowledge.
2. `SolveNode` first accepts an exact audited generated-skill match when one
   applies. Otherwise it exposes planner-selected toolsets so the model can
   inspect data, call helpers, invoke learned-skill tools, or, in development
   mode, file a reusable skill proposal.
3. `VerifyNode` checks the parsed output contract and family-specific
   consistency. Structured feedback can trigger one bounded retry.
4. `FinalizeNode` attaches route, tool, skill, retrieval, and reasoning traces.

Generated skills support two guarded paths: exact signature matches can execute
as the primary deterministic route, while the uncovered solver path exposes
skills as structured tools for model-judged applicability. Both paths retain
provenance, document-freshness, and output-shape guards.

## Semantic Compiler Runtime

The semantic path consists of five bounded stages:

1. **Family routing.** Existing deterministic planning selects a compact intent
   family: general, customer/fraud, fee, or unsupported.
2. **Compact intent planning.** A model chooses only semantic parameters in a
   family-specific Pydantic schema. It does not compute the final answer or
   emit arbitrary code.
3. **Compilation.** `intent_compiler.py` converts the intent into a typed
   `AnalysisSpec` over a closed operation vocabulary.
4. **Candidate execution.** `candidate_builder.py` expands uncertain axes into
   a small rival set; `analysis_executor.py` executes every candidate on the
   real data.
5. **Verification.** `semantic_verifier.py` compares candidate outputs,
   certified policies, structural checks, and optional judge evidence before
   selecting an answer.

Every failure in planner validation, compilation, candidate construction,
execution, or verification produces a traceable fallback reason. Candidate and
primary modes preserve answer availability through the established runtime;
strict mode intentionally does not.

## Model-Call Budget

The semantic runtime uses a `UsageLedger` with an explicit call budget.
Deterministic fast intents can consume zero model calls. A normal compact
planner uses one call, permits one schema-repair call, and reserves additional
budget for verifier judges only when executable rivals cannot be resolved
structurally.

The trace records prompt size, schema size, attempts, per-stage usage, latency,
candidate outputs, verifier decisions, selected path, and fallback reason.

## MemoryLake

MemoryLake retrieval runs before task solving:

- verbatim and family queries provide complementary recall;
- family filters keep unrelated curriculum out of context;
- whole chunks are admitted under a context budget;
- repeated family retrieval is cached with single-flight behavior;
- document fingerprints identify the exact knowledge snapshot used;
- official-style runs disable writes.

Retrieval errors are captured in the trace and degrade to the non-memory
runtime. They do not invalidate the task.

## Skill Lifecycle

The runtime and learn pipeline form a controlled feedback loop:

1. a development-mode solver encounters a reusable ambiguity;
2. it files a schema-level proposal without an answer literal;
3. `learn --from-proposals` matches that proposal to an induced public
   template;
4. the normal candidate, evidence, hardening, and audit gates run unchanged;
5. adopted artifacts become deterministic capabilities and compact LLM
   guidance for later runs.

Official-style read-only runs disable proposal writes. This is agent-directed
learning, not runtime self-modification: a development proposal grants
priority, never trust.

## Submission Boundary

Runtime JSONL contains operational traces for reproducibility. The report
exporter reduces it to `task_id` and `agent_answer`.

Accepted or golden answers are permitted only in an isolated post-inference
scoring step. They are never inputs to prompts, skill generation, MemoryLake,
planning, execution, or verification.
