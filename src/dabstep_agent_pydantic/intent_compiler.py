"""Deterministic compilers from compact SemanticIntent to AnalysisSpec.

Compilers inspect only intent fields and optional guidelines-derived output
settings. They never read task IDs, expected answers, or data values.
"""

from __future__ import annotations

from dabstep_agent_pydantic.analysis_spec_v2 import AggregateMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import AllTimeScope
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisOutputContract
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.analysis_spec_v2 import DayRangeScope
from dabstep_agent_pydantic.analysis_spec_v2 import EqFilter
from dabstep_agent_pydantic.analysis_spec_v2 import FilterSpec
from dabstep_agent_pydantic.analysis_spec_v2 import MonthScope
from dabstep_agent_pydantic.analysis_spec_v2 import NullFilter
from dabstep_agent_pydantic.analysis_spec_v2 import OrderSpec
from dabstep_agent_pydantic.analysis_spec_v2 import RatioMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import SourceSpec
from dabstep_agent_pydantic.analysis_spec_v2 import YearScope
from dabstep_agent_pydantic.distill.spec import FeeRulesSpec
from dabstep_agent_pydantic.distill.spec import InterpretationSpec
from dabstep_agent_pydantic.distill.spec import OutputSpec
from dabstep_agent_pydantic.distill.spec import PaymentsSpec
from dabstep_agent_pydantic.dabstep_core import get_month_day_range
from dabstep_agent_pydantic.fee_spec_adapter import adapt_interpretation_spec
from dabstep_agent_pydantic.semantic_intent import CustomerFraudIntent
from dabstep_agent_pydantic.semantic_intent import FeeIntent
from dabstep_agent_pydantic.semantic_intent import GeneralIntent
from dabstep_agent_pydantic.semantic_intent import IntentFilter
from dabstep_agent_pydantic.semantic_intent import IntentTimeScope
from dabstep_agent_pydantic.semantic_intent import SemanticIntent
from dabstep_agent_pydantic.semantic_intent import UnsupportedIntent

_FEE_CONTEXT_DIMS = (
    "account_type",
    "aci",
    "card_scheme",
    "is_credit",
    "merchant_category_code",
)


class IntentCompilationError(ValueError):
    """Raised when a compact intent cannot be compiled into AnalysisSpec."""


def compile_semantic_intent(
    intent: SemanticIntent,
    *,
    guidelines: str = "",
    policy_ids: list[str] | None = None,
) -> AnalysisSpec:
    if isinstance(intent, UnsupportedIntent):
        raise IntentCompilationError(intent.reason)
    if isinstance(intent, GeneralIntent):
        return _compile_general(intent, policy_ids=policy_ids or [])
    if isinstance(intent, CustomerFraudIntent):
        return _compile_customer_fraud(
            intent,
            policy_ids=policy_ids or [],
            guidelines=guidelines,
        )
    if isinstance(intent, FeeIntent):
        return _compile_fee(intent, guidelines=guidelines, policy_ids=policy_ids or [])
    raise IntentCompilationError(f"unsupported intent type: {type(intent).__name__}")


def _compile_fee(
    intent: FeeIntent,
    *,
    guidelines: str,
    policy_ids: list[str],
) -> AnalysisSpec:
    interpretation = _fee_interpretation(intent)
    params = dict(intent.params)
    # Drop decimals from interpretation params; output contract owns precision.
    params.pop("decimals", None)
    try:
        spec = adapt_interpretation_spec(
            interpretation,
            params=params,
            guidelines=guidelines,
        )
    except ValueError as exc:
        raise IntentCompilationError(str(exc)) from exc
    unresolved = list(intent.uncertainty_axes)
    policies = list(dict.fromkeys([*policy_ids, *spec.policy_ids]))
    if intent.wildcard_policy != "manual":
        unresolved.append(f"wildcard_policy:{intent.wildcard_policy}")
    if intent.fee_reducer != "sum_all_matching" and intent.operation in {
        "period_total_fees",
        "applicable_fee_ids",
        "affected_merchants",
        "fee_rate_delta",
        "merchant_mcc_delta",
    }:
        unresolved.append(f"fee_reducer:{intent.fee_reducer}")
    return spec.model_copy(update={
        "policy_ids": policies,
        "unresolved_axes": list(dict.fromkeys([*spec.unresolved_axes, *unresolved])),
    })


def _fee_interpretation(intent: FeeIntent) -> InterpretationSpec:
    cite = "manual fee-rule semantics (null/[] wildcard, fee = fixed + rate*value/10000)"
    op = intent.operation
    if op == "period_total_fees":
        return InterpretationSpec(
            name="intent_period_total_fees",
            population="payments",
            payments=PaymentsSpec(
                primitive="period_total_fees",
                reducer=intent.fee_reducer,
            ),
            output=OutputSpec(kind="decimal", decimals_default=_decimals_or_default(intent.decimals, 2)),
            manual_citation=cite,
        )
    if op == "applicable_fee_ids":
        return InterpretationSpec(
            name="intent_applicable_fee_ids",
            population="payments",
            payments=PaymentsSpec(primitive="applicable_fee_ids_period"),
            output=OutputSpec(kind="id_list"),
            manual_citation=cite,
        )
    if op == "fee_at_amount":
        return InterpretationSpec(
            name="intent_fee_at_amount",
            population="fee_rules",
            fee_rules=FeeRulesSpec(
                context_dims=_context_dims_from_params(intent.params),
                value="fee_at_amount",
                reducer="mean",
                wildcard_policy=intent.wildcard_policy,
            ),
            output=OutputSpec(kind="decimal", decimals_default=_decimals_or_default(intent.decimals, 6)),
            manual_citation=cite,
        )
    if op == "card_scheme_extreme":
        if "merchant" in intent.params or "year" in intent.params:
            raise IntentCompilationError(
                "merchant card-scheme steering requires a payment-population primitive"
            )
        dims = _context_dims_from_params(intent.params)
        if "card_scheme" in intent.params:
            return InterpretationSpec(
                name="intent_named_card_scheme_fee",
                population="fee_rules",
                fee_rules=FeeRulesSpec(
                    context_dims=dims,
                    value="fee_at_amount",
                    reducer="mean",
                    wildcard_policy=intent.wildcard_policy,
                ),
                output=OutputSpec(
                    kind="decimal",
                    decimals_default=_decimals_or_default(intent.decimals, 6),
                ),
                manual_citation=cite,
            )
        extreme = "argmax" if (intent.objective or "max") == "max" else "argmin"
        return InterpretationSpec(
            name="intent_card_scheme_extreme",
            population="fee_rules",
            fee_rules=FeeRulesSpec(
                context_dims=dims,
                value="fee_at_amount",
                reducer="mean",
                wildcard_policy=intent.wildcard_policy,
                group_by="card_scheme",
                group_extreme=extreme,
            ),
            output=OutputSpec(kind="single_string", tie_policy="list_all_sorted"),
            manual_citation=cite,
        )
    if op == "aci_extreme":
        if "merchant" in intent.params or "year" in intent.params:
            raise IntentCompilationError(
                "merchant ACI steering is not the same as a static fee-rule extreme"
            )
        extreme = "argmax" if (intent.objective or "max") == "max" else "argmin"
        dims = [d for d in _context_dims_from_params(intent.params) if d != "aci"]
        return InterpretationSpec(
            name="intent_aci_extreme",
            population="fee_rules",
            fee_rules=FeeRulesSpec(
                context_dims=dims,
                value="fee_at_amount",
                reducer="sum",
                wildcard_policy=intent.wildcard_policy,
                group_by="aci",
                group_extreme=extreme,
            ),
            output=OutputSpec(kind="single_string", tie_policy="list_all_sorted"),
            manual_citation=cite,
        )
    if op == "affected_merchants":
        mode = intent.affected_mode or "baseline_members"
        return InterpretationSpec(
            name="intent_affected_merchants",
            population="payments",
            payments=PaymentsSpec(
                primitive="affected_merchants",
                affected_mode=mode,
            ),
            output=OutputSpec(kind="string_list"),
            manual_citation=cite,
        )
    if op == "fee_rate_delta":
        basis = intent.delta_basis or "rate"
        return InterpretationSpec(
            name="intent_fee_rate_delta",
            population="payments",
            payments=PaymentsSpec(
                primitive="period_fee_rate_delta",
                reducer=intent.fee_reducer,
                delta_basis=basis,
            ),
            output=OutputSpec(kind="decimal", decimals_default=_decimals_or_default(intent.decimals, 14)),
            manual_citation=cite,
        )
    if op == "merchant_mcc_delta":
        return InterpretationSpec(
            name="intent_merchant_mcc_delta",
            population="payments",
            payments=PaymentsSpec(primitive="mcc_change_fee_delta"),
            output=OutputSpec(kind="decimal", decimals_default=_decimals_or_default(intent.decimals, 6)),
            manual_citation=cite,
        )
    raise IntentCompilationError(f"unsupported fee operation: {op}")


def _context_dims_from_params(params: dict) -> list[str]:
    return [dim for dim in _FEE_CONTEXT_DIMS if dim in params]


def _decimals_or_default(decimals: int | None, default: int) -> int:
    return default if decimals is None else decimals


def _compile_general(intent: GeneralIntent, *, policy_ids: list[str]) -> AnalysisSpec:
    if intent.operation in {"correlation", "outlier"}:
        raise IntentCompilationError(
            f"no deterministic primitive for operation={intent.operation}"
        )
    if intent.operation == "duplicate":
        raise IntentCompilationError("no deterministic primitive for operation=duplicate")

    time_scope = _compile_time_scope(intent.time_scope)
    filters = _compile_filters(intent.filters)
    unresolved = list(intent.uncertainty_axes)
    output = _output_contract(intent.output_kind, intent.decimals)

    if intent.operation == "missingness":
        if intent.column is None:
            raise IntentCompilationError("missingness requires column")
        measure = AggregateMeasure(
            kind="count",
            column=intent.column,
            filters=[NullFilter(op="is_null", column=intent.column)],
            missing="include",
        )
        return AnalysisSpec(
            source=SourceSpec(table=intent.source),
            time_scope=time_scope,
            filters=filters,
            measure=measure,
            output=AnalysisOutputContract(kind="integer"),
            policy_ids=policy_ids,
            unresolved_axes=unresolved,
        )

    if intent.operation == "domain":
        if intent.column is None:
            raise IntentCompilationError("domain requires column")
        return AnalysisSpec(
            source=SourceSpec(table=intent.source),
            time_scope=time_scope,
            filters=filters,
            measure=AggregateMeasure(kind="nunique", column=intent.column, missing="exclude"),
            group_by=[intent.column],
            ordering=[OrderSpec(by=intent.column, direction="asc")],
            output=AnalysisOutputContract(kind="comma_list", empty_string_allowed=True),
            policy_ids=policy_ids,
            unresolved_axes=unresolved,
        )

    if intent.operation == "ratio":
        measure = RatioMeasure(
            numerator=AggregateMeasure(
                kind=intent.numerator_aggregation or "sum",
                column=intent.numerator_column,
                filters=_compile_filters(intent.numerator_filters),
            ),
            denominator=AggregateMeasure(
                kind=intent.denominator_aggregation or "sum",
                column=intent.denominator_column,
                filters=_compile_filters(intent.denominator_filters),
            ),
            scale=intent.ratio_scale,
        )
        return AnalysisSpec(
            source=SourceSpec(table=intent.source),
            time_scope=time_scope,
            filters=filters,
            measure=measure,
            group_by=list(intent.group_by),
            ordering=_ordering(intent),
            limit=intent.limit,
            output=output,
            policy_ids=policy_ids,
            unresolved_axes=unresolved,
        )

    # aggregate / grouped_aggregate
    if intent.aggregation is None:
        raise IntentCompilationError("aggregation is required")
    measure = AggregateMeasure(
        kind=intent.aggregation,
        column=intent.column,
        missing="exclude",
    )
    group_by = list(intent.group_by)
    ordering = _ordering(intent)
    limit = intent.limit
    out = output
    if intent.operation == "grouped_aggregate" or group_by:
        if not group_by:
            raise IntentCompilationError("grouped_aggregate requires group_by")
        label_only = intent.limit == 1 and intent.output_kind in {
            "single_string", "integer", "comma_list"
        }
        if label_only:
            out = AnalysisOutputContract(kind="comma_list", empty_string_allowed=True)
        elif intent.extreme is not None:
            ordering = [
                OrderSpec(
                    by="value",
                    direction="asc" if intent.extreme == "min" else "desc",
                )
            ]
            limit = limit or 1
            if out.kind not in {"group_value_list", "comma_list"}:
                out = AnalysisOutputContract(kind="comma_list", empty_string_allowed=True)
        elif out.kind not in {"group_value_list", "comma_list"}:
            out = AnalysisOutputContract(
                kind="group_value_list",
                decimals=intent.decimals if intent.decimals is not None else 2,
            )
    return AnalysisSpec(
        source=SourceSpec(table=intent.source),
        time_scope=time_scope,
        filters=filters,
        measure=measure,
        group_by=group_by,
        ordering=ordering,
        limit=limit,
        output=out,
        policy_ids=policy_ids,
        unresolved_axes=unresolved,
    )


def _compile_customer_fraud(
    intent: CustomerFraudIntent,
    *,
    policy_ids: list[str],
    guidelines: str,
) -> AnalysisSpec:
    if "yes or no" in guidelines.lower():
        raise IntentCompilationError(
            "yes/no fraud comparisons require a deterministic comparison primitive"
        )
    if intent.operation in {"outlier_fraud", "correlation"}:
        raise IntentCompilationError(
            f"no deterministic primitive for operation={intent.operation}"
        )

    time_scope = _compile_time_scope(intent.time_scope)
    filters = _compile_filters(intent.filters)
    unresolved = list(intent.uncertainty_axes)
    policies = list(policy_ids)

    if intent.operation == "repeat_customer_percentage":
        raise IntentCompilationError(
            "repeat_customer_percentage requires a dedicated repeat-population primitive"
        )

    if intent.operation == "missing_identity":
        missing_measure = AggregateMeasure(
            kind="count",
            column=intent.identity_field,
            filters=[NullFilter(op="is_null", column=intent.identity_field)],
            missing="include",
        )
        if intent.output_kind == "decimal":
            measure = RatioMeasure(
                numerator=missing_measure,
                denominator=AggregateMeasure(kind="count"),
                scale=100,
            )
            output = _output_contract(intent.output_kind, intent.decimals)
        else:
            measure = missing_measure
            output = AnalysisOutputContract(kind="integer")
        return AnalysisSpec(
            source=SourceSpec(table="payments"),
            time_scope=time_scope,
            filters=filters,
            measure=measure,
            output=output,
            policy_ids=policies,
            unresolved_axes=unresolved,
        )

    # fraud_rate / fraud_rate_extreme
    if intent.fraud_rate_basis is None:
        raise IntentCompilationError("fraud_rate_basis is required")
    if intent.fraud_rate_basis == "eur_volume":
        numerator = AggregateMeasure(
            kind="sum",
            column="eur_amount",
            filters=[EqFilter(column="has_fraudulent_dispute", value=True)],
        )
        denominator = AggregateMeasure(kind="sum", column="eur_amount")
    else:
        numerator = AggregateMeasure(
            kind="count",
            filters=[EqFilter(column="has_fraudulent_dispute", value=True)],
        )
        denominator = AggregateMeasure(kind="count")
    measure = RatioMeasure(numerator=numerator, denominator=denominator, scale=100)
    unresolved = list(dict.fromkeys([
        *unresolved,
        f"fraud_rate_basis:{intent.fraud_rate_basis}",
    ]))

    group_by: list[str] = []
    ordering: list[OrderSpec] = []
    limit = intent.limit
    output = _output_contract(intent.output_kind, intent.decimals)

    if intent.operation == "fraud_rate_extreme":
        if intent.group_by is None:
            raise IntentCompilationError("fraud_rate_extreme requires group_by")
        group_by = [intent.group_by]
        direction = "asc" if (intent.extreme or "max") == "min" else "desc"
        ordering = [OrderSpec(by="value", direction=direction)]
        limit = limit or 1
        # Grouped AnalysisSpec cannot emit single_string; canonicalize to list.
        if output.kind not in {"group_value_list", "comma_list"}:
            output = AnalysisOutputContract(kind="comma_list", empty_string_allowed=True)

    return AnalysisSpec(
        source=SourceSpec(table="payments"),
        time_scope=time_scope,
        filters=filters,
        measure=measure,
        group_by=group_by,
        ordering=ordering,
        limit=limit,
        output=output,
        policy_ids=policies,
        unresolved_axes=unresolved,
    )


def _compile_time_scope(scope: IntentTimeScope):
    if scope.kind == "all":
        return AllTimeScope()
    if scope.kind == "year":
        assert scope.year is not None
        return YearScope(year=scope.year)
    if scope.kind == "month":
        assert scope.year is not None and scope.month is not None
        return MonthScope(year=scope.year, month=scope.month)
    if scope.kind == "month_range":
        assert scope.year is not None
        assert scope.start_month is not None and scope.end_month is not None
        start_day, _ = get_month_day_range(scope.year, scope.start_month)
        _, end_day = get_month_day_range(scope.year, scope.end_month)
        return DayRangeScope(
            year=scope.year,
            start_day=start_day,
            end_day=end_day,
        )
    if scope.kind == "day":
        assert scope.year is not None and scope.day_of_year is not None
        return DayRangeScope(
            year=scope.year,
            start_day=scope.day_of_year,
            end_day=scope.day_of_year,
        )
    raise IntentCompilationError(f"unsupported time scope: {scope.kind}")


def _compile_filters(filters: list[IntentFilter]) -> list[FilterSpec]:
    compiled = []
    seen: dict[str, object] = {}
    for item in filters:
        column = item.column
        value = item.value
        if column == "payment_method":
            column = "card_scheme"
        if isinstance(value, str):
            normalized = value.strip().lower().replace("_", "-")
            if column in {"device_type", "shopper_interaction"}:
                if normalized in {"in-person", "in person", "in-store", "in store", "pos"}:
                    column, value = "shopper_interaction", "POS"
                elif normalized in {"ecommerce", "e-commerce", "online"}:
                    column, value = "shopper_interaction", "Ecommerce"
        if column in seen:
            if seen[column] != value:
                raise IntentCompilationError(
                    f"multiple equality values for {column} require an in-filter primitive"
                )
            continue
        seen[column] = value
        compiled.append(EqFilter(column=column, value=value))
    return compiled


def _output_contract(kind: str, decimals: int | None) -> AnalysisOutputContract:
    if kind == "decimal":
        if decimals is None:
            return AnalysisOutputContract(kind="single_string")
        return AnalysisOutputContract(kind="decimal", decimals=decimals)
    if kind == "group_value_list":
        return AnalysisOutputContract(kind="group_value_list", decimals=decimals or 2)
    if kind == "comma_list":
        return AnalysisOutputContract(kind="comma_list", empty_string_allowed=True)
    if kind == "single_string":
        return AnalysisOutputContract(kind="single_string", empty_string_allowed=True)
    return AnalysisOutputContract(kind=kind)  # type: ignore[arg-type]


def _ordering(intent: GeneralIntent) -> list[OrderSpec]:
    if intent.order_by is None:
        return []
    return [OrderSpec(by=intent.order_by, direction=intent.order_direction)]
