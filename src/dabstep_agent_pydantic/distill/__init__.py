"""Semantics distillation: template -> interpretation specs -> compiled skills.

The learn pipeline never emits free-form generated code. Interpretation
candidates are declarative `InterpretationSpec` objects in a restricted DSL;
`compile_spec` assembles them from the manual-mandated mechanism layer in
`dabstep_core`. Correctness of an adopted spec is established by discrimination
against model-generated reference answers plus spec-derived invariants.
"""

from dabstep_agent_pydantic.distill.spec import InterpretationSpec
from dabstep_agent_pydantic.distill.combinators import compile_spec

__all__ = ["InterpretationSpec", "compile_spec"]
