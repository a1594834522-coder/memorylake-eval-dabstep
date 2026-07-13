from __future__ import annotations

from typing import Any

from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisOutputContract
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.analysis_spec_v2 import InterpretationMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import SourceSpec
from dabstep_agent_pydantic.distill.spec import InterpretationSpec
from dabstep_agent_pydantic.runtime_util import decimal_places


def adapt_interpretation_spec(
    interpretation: InterpretationSpec,
    *,
    params: dict[str, Any],
    guidelines: str,
) -> AnalysisSpec:
    output = _adapt_output(interpretation=interpretation, guidelines=guidelines)
    source = "fees" if interpretation.population == "fee_rules" else "payments"
    return AnalysisSpec(
        spec_id=f"interpretation:{interpretation.name}",
        source=SourceSpec(table=source),
        measure=InterpretationMeasure(
            interpretation=interpretation,
            params=params,
        ),
        output=output,
    )


def _adapt_output(
    *,
    interpretation: InterpretationSpec,
    guidelines: str,
) -> AnalysisOutputContract:
    output = interpretation.output
    if output.kind == "decimal":
        default_places = 6 if output.decimals_default is None else output.decimals_default
        places = decimal_places(guidelines, default=default_places)
        return AnalysisOutputContract(kind="decimal", decimals=places)
    if output.kind == "integer":
        return AnalysisOutputContract(kind="integer")
    if output.kind in {"id_list", "string_list"}:
        return AnalysisOutputContract(kind="comma_list", empty_string_allowed=True)
    if output.kind == "single_string" and output.tie_policy == "list_all_sorted":
        return AnalysisOutputContract(kind="comma_list", empty_string_allowed=True)
    if output.kind == "single_string":
        return AnalysisOutputContract(kind="single_string", empty_string_allowed=True)
    raise ValueError(f"unsupported interpretation output kind: {output.kind}")
