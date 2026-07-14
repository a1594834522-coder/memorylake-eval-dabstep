from __future__ import annotations

import time
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from pydantic_graph import BaseNode
from pydantic_graph import End
from pydantic_graph import GraphBuilder
from pydantic_graph import GraphRunContext
from pydantic_graph import StepContext
from pydantic_ai.usage import UsageLimits

from dabstep_agent_pydantic.agent import DABStepAnswer
from dabstep_agent_pydantic.agent import DABStepDeps
from dabstep_agent_pydantic.agent import create_agent
from dabstep_agent_pydantic.dataset import Task
from dabstep_agent_pydantic.distill.shadow import calibration_note_for
from dabstep_agent_pydantic.distill.shadow import generated_skills_mode
from dabstep_agent_pydantic.distill.shadow import try_solve_generated
from dabstep_agent_pydantic.evaluation_policy import EvaluationPolicy
from dabstep_agent_pydantic.runtime_util import cached_load_dabstep_data as _cached_load_dabstep_data
from dabstep_agent_pydantic.output_contract import CENTS_SCALE_PRIMITIVES
from dabstep_agent_pydantic.output_contract import scorer_aligned_precision
from dabstep_agent_pydantic.planning import PlanDecision
from dabstep_agent_pydantic.planning import plan_task
from dabstep_agent_pydantic.python_tool import PythonWorkspace
from dabstep_agent_pydantic.runtime_assets import load_runtime_assets
from dabstep_agent_pydantic.semantic_workflow import SemanticMode
from dabstep_agent_pydantic.semantic_workflow import SemanticRuntimeHooks
from dabstep_agent_pydantic.semantic_workflow import run_semantic_workflow
from dabstep_agent_pydantic.toolsets import tool_selection_mode
from dabstep_agent_pydantic.toolsets import toolset_names_for_plan
from dabstep_agent_pydantic.toolsets import toolsets_for_plan
from dabstep_agent_pydantic.usage_telemetry import UsageLedger
from dabstep_agent_pydantic.usage_telemetry import call_usage_from_result
from dabstep_agent_pydantic.verification import verify_record


SolveHook = Callable[..., Awaitable[dict[str, object]]]
VerifyHook = Callable[[dict[str, object], "WorkflowState"], str | None]


@dataclass
class WorkflowInput:
    task: Task
    data_dir: Path
    workspace_dir: Path
    file_summary: str
    assets_dir: Path | None = None
    memory_context: str | None = None
    evaluation_policy: EvaluationPolicy = field(default_factory=EvaluationPolicy.development)
    solve_hook: SolveHook | None = None
    verify_hook: VerifyHook | None = None


@dataclass
class WorkflowState:
    inputs: WorkflowInput
    plan: PlanDecision | None = None
    record: dict[str, object] | None = None
    stages: list[str] = field(default_factory=list)
    solver_attempts: int = 0
    verifier_feedback: str | None = None
    selected_route_ids: list[str] = field(default_factory=list)
    selected_toolsets: list[str] = field(default_factory=list)
    usage_ledger: UsageLedger = field(default_factory=UsageLedger)


@dataclass
class PlanNode(BaseNode[WorkflowState, None, dict[str, object]]):
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> "SolveNode":
        ctx.state.stages.append("plan")
        runtime_assets = load_runtime_assets(ctx.state.inputs.assets_dir)
        plan = plan_task(
            question=ctx.state.inputs.task.question,
            guidelines=ctx.state.inputs.task.guidelines,
            route_cards=runtime_assets.route_cards,
        )
        ctx.state.plan = plan
        ctx.state.selected_route_ids = plan.selected_route_ids
        ctx.state.selected_toolsets = toolset_names_for_plan(plan)
        return SolveNode()


@dataclass
class SolveNode(BaseNode[WorkflowState, None, dict[str, object]]):
    feedback: str | None = None

    async def run(self, ctx: GraphRunContext[WorkflowState]) -> "VerifyNode":
        ctx.state.stages.append("solve")
        ctx.state.solver_attempts += 1
        solve_kwargs = {
            "task": ctx.state.inputs.task,
            "data_dir": ctx.state.inputs.data_dir,
            "workspace_dir": ctx.state.inputs.workspace_dir,
            "file_summary": ctx.state.inputs.file_summary,
            "assets_dir": ctx.state.inputs.assets_dir,
            "memory_context": ctx.state.inputs.memory_context,
            "evaluation_policy": ctx.state.inputs.evaluation_policy,
            "plan": ctx.state.plan,
            "feedback": self.feedback,
        }
        if ctx.state.inputs.solve_hook is not None:
            ctx.state.record = await ctx.state.inputs.solve_hook(**solve_kwargs)
        else:
            ctx.state.record = await _default_solve(
                **solve_kwargs,
                usage_ledger=ctx.state.usage_ledger,
                retries=ctx.state.solver_attempts - 1,
            )
        return VerifyNode()


@dataclass
class VerifyNode(BaseNode[WorkflowState, None, dict[str, object]]):
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> SolveNode | FinalizeNode:
        ctx.state.stages.append("verify")
        assert ctx.state.record is not None
        verify_hook = ctx.state.inputs.verify_hook or _default_verify
        feedback = verify_hook(ctx.state.record, ctx.state)
        ctx.state.verifier_feedback = feedback
        max_retries = ctx.state.plan.max_solver_retries if ctx.state.plan else 0
        if feedback and ctx.state.solver_attempts <= max_retries:
            return SolveNode(feedback=feedback)
        return FinalizeNode()


@dataclass
class FinalizeNode(BaseNode[WorkflowState, None, dict[str, object]]):
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> End[dict[str, object]]:
        ctx.state.stages.append("finalize")
        assert ctx.state.record is not None
        record = dict(ctx.state.record)
        answer = record.get("agent_answer")
        if isinstance(answer, str):
            record["agent_answer"] = scorer_aligned_precision(
                answer,
                force_cents=record.get("answer_primitive") in CENTS_SCALE_PRIMITIVES,
            )
        record["workflow_trace"] = {
            "stages": list(ctx.state.stages),
            "solver_attempts": ctx.state.solver_attempts,
            "selected_route_ids": list(ctx.state.selected_route_ids),
            "selected_toolsets": list(ctx.state.selected_toolsets),
            "verifier_feedback": ctx.state.verifier_feedback,
            "plan": ctx.state.plan.model_dump(mode="json") if ctx.state.plan else None,
        }
        record["usage_trace"] = ctx.state.usage_ledger.summary()
        return End(record)


graph_builder = GraphBuilder(input_type=WorkflowInput, state_type=WorkflowState, output_type=dict[str, object])


@graph_builder.step
async def start(ctx: StepContext[WorkflowState, None, WorkflowInput]) -> PlanNode:
    return PlanNode()


graph_builder.add(
    graph_builder.node(PlanNode),
    graph_builder.node(SolveNode),
    graph_builder.node(VerifyNode),
    graph_builder.node(FinalizeNode),
    graph_builder.edge_from(graph_builder.start_node).to(start),
)
dabstep_graph = graph_builder.build()


async def run_task_workflow(
    task: Task,
    *,
    data_dir: Path,
    workspace_dir: Path,
    file_summary: str,
    assets_dir: Path | None = None,
    memory_context: str | None = None,
    evaluation_policy: EvaluationPolicy | None = None,
    solve_hook: SolveHook | None = None,
    verify_hook: VerifyHook | None = None,
    semantic_mode: SemanticMode | str = SemanticMode.LEGACY,
    semantic_hooks: SemanticRuntimeHooks | None = None,
) -> dict[str, object]:
    async def legacy_runner() -> dict[str, object]:
        return await _run_legacy_task_workflow(
            task,
            data_dir=data_dir,
            workspace_dir=workspace_dir,
            file_summary=file_summary,
            assets_dir=assets_dir,
            memory_context=memory_context,
            evaluation_policy=evaluation_policy,
            solve_hook=solve_hook,
            verify_hook=verify_hook,
        )

    return await run_semantic_workflow(
        task=task,
        mode=SemanticMode(semantic_mode),
        data_dir=data_dir,
        workspace_dir=workspace_dir,
        file_summary=file_summary,
        memory_context=memory_context,
        legacy_runner=legacy_runner,
        hooks=semantic_hooks,
    )


async def _run_legacy_task_workflow(
    task: Task,
    *,
    data_dir: Path,
    workspace_dir: Path,
    file_summary: str,
    assets_dir: Path | None = None,
    memory_context: str | None = None,
    evaluation_policy: EvaluationPolicy | None = None,
    solve_hook: SolveHook | None = None,
    verify_hook: VerifyHook | None = None,
) -> dict[str, object]:
    workflow_input = WorkflowInput(
        task=task,
        data_dir=data_dir,
        workspace_dir=workspace_dir,
        file_summary=file_summary,
        assets_dir=assets_dir,
        memory_context=memory_context,
        evaluation_policy=evaluation_policy or EvaluationPolicy.development(),
        solve_hook=solve_hook,
        verify_hook=verify_hook,
    )
    state = WorkflowState(inputs=workflow_input)
    return await dabstep_graph.run(inputs=workflow_input, state=state)


def build_task_prompt(task: Task, *, feedback: str | None = None, plan: PlanDecision | None = None) -> str:
    feedback_text = f"\nVERIFIER FEEDBACK: {feedback}\n" if feedback else ""
    ambiguity_text = _ambiguity_prompt(plan)
    hint_text = _toolset_hint_prompt(plan)
    note = calibration_note_for(task.question)
    note_text = (
        f"\nINTERPRETATION NOTE (schema-level convention distilled for this metric family; "
        f"apply it unless the question wording overrides it):\n{note}\n" if note else ""
    )
    return f"""\
QUESTION: {task.question}

GUIDELINES: {task.guidelines or "N/A"}
{ambiguity_text}{hint_text}{note_text}
{feedback_text}
"""


def _toolset_hint_prompt(plan: PlanDecision | None) -> str:
    """In open tool-selection mode the plan no longer restricts the library;
    surface its recommendation as advice so the route-card knowledge still
    reaches the model."""
    if tool_selection_mode() != "open" or plan is None or not plan.toolset_ids:
        return ""
    recommended = ", ".join(plan.toolset_ids)
    return f"""
TOOLSET HINT: tasks of this family are usually solved with: {recommended}.
The full tool library is available; treat the hint as advice, not a restriction.
"""


def _ambiguity_prompt(plan: PlanDecision | None) -> str:
    if plan is None or not plan.ambiguity_axes:
        return ""
    axes = ", ".join(plan.ambiguity_axes)
    return f"""
AMBIGUITY AXES: {axes}
For each ambiguity axis, compute the plausible candidate interpretations side by side, compare the resulting values, and choose using only the question wording: nouns, modifiers, scope words, and explicit guideline text. Explain the chosen interpretation in reasoning.
"""


async def _default_solve(
    *,
    task: Task,
    data_dir: Path,
    workspace_dir: Path,
    file_summary: str,
    assets_dir: Path | None,
    memory_context: str | None,
    evaluation_policy: EvaluationPolicy,
    plan: PlanDecision | None,
    feedback: str | None,
    usage_ledger: UsageLedger,
    retries: int,
) -> dict[str, object]:
    start_time = time.time()
    runtime_assets = load_runtime_assets(assets_dir)
    selected_cards = [
        card
        for card in runtime_assets.route_cards
        if plan is None or card.route_id in set(plan.selected_route_ids)
    ]
    deps = DABStepDeps(
        data_dir=data_dir,
        workspace=PythonWorkspace(workspace_dir),
        file_summary=file_summary,
        runtime_patterns=runtime_assets.patterns or None,
        memory_context=memory_context,
        route_cards=selected_cards,
        analysis_plan=plan.analysis_plan if plan else None,
        evaluation_policy=evaluation_policy,
    )

    if generated_skills_mode() == "primary":
        skill_data = _cached_load_dabstep_data(data_dir)
        generated = try_solve_generated(task.question, task.guidelines or "", skill_data,
                                        data_dir=data_dir)
        if generated is not None:
            code_path = deps.workspace.save_generated_code(task.task_id)
            return {
                "task_id": task.task_id,
                "agent_answer": generated.agent_answer,
                "reasoning": "Learned skill (spec-as-code) compiled from the learn pipeline.",
                "used_code": True,
                "elapsed_seconds": round(time.time() - start_time, 3),
                "code_path": str(code_path),
                "deterministic_route": f"generated:{generated.skill_id}",
                "answer_primitive": generated.primitive,
            }

    agent = create_agent()
    usage_ledger.ensure_can_call("solver")
    model_start = time.perf_counter()
    result = await agent.run(
        build_task_prompt(task, feedback=feedback, plan=plan),
        deps=deps,
        toolsets=toolsets_for_plan(plan) if plan else None,
        usage_limits=UsageLimits(request_limit=None),
    )
    usage_ledger.record(
        call_usage_from_result(
            result,
            stage="solver",
            latency_ms=(time.perf_counter() - model_start) * 1000,
            retries=retries,
        )
    )
    code_path = deps.workspace.save_generated_code(task.task_id)
    output: DABStepAnswer = result.output
    return {
        "task_id": task.task_id,
        "agent_answer": output.agent_answer,
        "reasoning": output.reasoning,
        "used_code": output.used_code,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "code_path": str(code_path),
    }


def _default_verify(record: dict[str, object], state: WorkflowState) -> str | None:
    return verify_record(
        record,
        task=state.inputs.task,
        plan=state.plan,
        data_dir=state.inputs.data_dir,
    )
