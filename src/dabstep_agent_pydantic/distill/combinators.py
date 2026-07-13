"""Compile an InterpretationSpec into an executable candidate.

`compile_spec(spec) -> fn(data, params, guidelines) -> str`

The compiler only assembles mechanism-layer functions from `dabstep_core`
(wildcard predicates, fee formula, monthly context) — it never re-implements
matching semantics. Rejected-interpretation candidates (min/first match,
symmetric difference, sampled-day tuples, ...) are implemented here on top of
the same mechanism layer so discrimination compares real computations.

`params` is produced by the signature layer: question-derived values keyed by
canonical names (amount, account_type, aci, card_scheme, is_credit, merchant,
year, month, fee_id, new_value, new_mcc, mcc_code, percentile).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from dabstep_agent_pydantic.dabstep_core import (
    DABStepData,
    applicable_fee_ids_for_merchant_period,
    calculate_fee,
    fee_affected_merchants_for_year,
    fee_fixed_component_delta_for_month,
    fee_rate_delta_for_period,
    field_domain_values,
    format_decimal_places,
    matches_fee_rule,
    merchant_mcc_fee_delta_for_year,
    optimize_aci_for_fraudulent_transactions,
    repeat_customer_percentage,
    total_fees_for_merchant_period,
    unique_merchant_count,
)
from dabstep_agent_pydantic.dabstep_core import (
    _matches_fee_for_payment,
    _matching_fee_group_rows,
    _merchant_by_name,
    _merchant_month_context,
    _merchant_month_payments,
    _merchants_matching_fee_context,
    _payment_fee_context,
    _yearly_payment_context,
    _fee_by_id,
)
from dabstep_agent_pydantic.runtime_util import decimal_places as _decimal_places
from dabstep_agent_pydantic.distill.memo import memo_get_or_compute
from dabstep_agent_pydantic.distill.spec import InterpretationSpec, OutputSpec

CandidateFn = Callable[[DABStepData, dict[str, Any], str], str]


class SpecNotExecutable(Exception):
    """The spec is valid DSL but the compiler has no execution strategy for it."""


def compile_spec(spec: InterpretationSpec) -> CandidateFn:
    output = spec.output

    def run(data: DABStepData, params: dict[str, Any], guidelines: str) -> str:
        return _format_output(execute_spec_value(spec, data, params), output, guidelines)

    return run


def execute_spec_value(
    spec: InterpretationSpec,
    data: DABStepData,
    params: dict[str, Any],
) -> Any:
    """Execute a closed InterpretationSpec without applying its output format."""
    if spec.population == "fee_rules":
        core = _compile_fee_rules(spec)
    else:
        core = _compile_payments(spec)
    return core(data, params)


# --------------------------------------------------------------------------
# fee-rules family: match-context -> per-rule value -> reduce (optional group)
# --------------------------------------------------------------------------


def _compile_fee_rules(spec: InterpretationSpec):
    fr = spec.fee_rules
    assert fr is not None

    def matched_rules(data: DABStepData, params: dict[str, Any]) -> list[dict[str, Any]]:
        context: dict[str, Any] = dict(fr.extra_context)
        for dim in fr.context_dims:
            if params.get(dim) is not None:
                context[dim] = params[dim]
        rows = data.fees.to_dict(orient="records")
        if fr.wildcard_policy == "strict":
            return [row for row in rows if _matches_strict(row, context)]
        if "card_scheme" in context:
            return [row for row in rows if _matches_fee_for_payment(row, context)]
        return [row for row in rows if matches_fee_rule(row, context)]

    def rule_value(row: dict[str, Any], params: dict[str, Any]) -> float:
        if fr.value == "rule_id":
            return int(row["ID"])
        return calculate_fee(row["fixed_amount"], row["rate"], float(params["amount"]))

    if fr.group_by is None:

        def core(data: DABStepData, params: dict[str, Any]):
            rows = matched_rules(data, params)
            if fr.reducer == "collect_ids":
                return sorted(int(row["ID"]) for row in rows)
            values = [rule_value(row, params) for row in rows]
            return _reduce_values(values, fr.reducer)

        return core

    group_dim = fr.group_by
    extreme = fr.group_extreme

    def grouped_core(data: DABStepData, params: dict[str, Any]):
        rows = matched_rules(data, params)
        universe = sorted(
            {value for row in rows for value in (row.get(group_dim) or [])}
        )
        per_group: dict[Any, list[float]] = defaultdict(list)
        for row in rows:
            members = row.get(group_dim) or universe  # empty list = wildcard
            value = rule_value(row, params)
            for member in members:
                per_group[member].append(value)
        if not per_group:
            raise SpecNotExecutable(f"no groups for {group_dim}")
        reduced = {member: _reduce_values(vals, fr.reducer) for member, vals in per_group.items()}
        target = max(reduced.values()) if extreme == "argmax" else min(reduced.values())
        winners = sorted(member for member, val in reduced.items() if abs(val - target) < 1e-12)
        return winners

    return grouped_core


def _matches_strict(row: dict[str, Any], context: dict[str, Any]) -> bool:
    """Rejected reading: rule fields must contain/equal the value explicitly."""
    for key, value in context.items():
        field = row.get(key)
        if isinstance(field, list):
            if value not in field:
                return False
        elif field is None or field != value:
            return False
    return True


def _reduce_values(values: list[float], reducer: str) -> float:
    if not values:
        raise SpecNotExecutable("no matching fee rules")
    if reducer == "mean":
        return sum(values) / len(values)
    if reducer == "sum":
        return sum(values)
    if reducer == "min":
        return min(values)
    if reducer == "max":
        return max(values)
    raise SpecNotExecutable(f"unsupported reducer {reducer}")


# --------------------------------------------------------------------------
# payments family: domain primitives with interpretation axes as parameters
# --------------------------------------------------------------------------


def _compile_payments(spec: InterpretationSpec):
    p = spec.payments
    assert p is not None
    primitive = p.primitive

    if primitive == "period_total_fees":
        if p.reducer == "sum_all_matching":
            return lambda data, params: total_fees_for_merchant_period(
                data, merchant=params["merchant"], year=_year_param(data, params),
                month=params.get("month"), day_of_year=params.get("day_of_year"),
            )
        return lambda data, params: _period_fees_per_txn_reduced(
            data, merchant=params["merchant"], year=int(params["year"]),
            month=params.get("month"), reducer=p.reducer,
        )

    if primitive == "period_fee_rate_delta":
        if p.delta_basis == "fixed_component":
            return lambda data, params: _fixed_component_delta_period(
                data, merchant=params["merchant"], year=int(params["year"]),
                month=params.get("month"), fee_id=int(params["fee_id"]),
            )
        return lambda data, params: fee_rate_delta_for_period(
            data, merchant=params["merchant"], year=int(params["year"]),
            month=params.get("month"), fee_id=int(params["fee_id"]),
            new_rate=float(params["new_value"]),
        )

    if primitive == "mcc_change_fee_delta":
        return lambda data, params: merchant_mcc_fee_delta_for_year(
            data, merchant=params["merchant"], year=int(params["year"]), new_mcc=int(params["new_mcc"])
        )

    if primitive == "affected_merchants":
        mode = p.affected_mode or "baseline_members"

        def affected(data: DABStepData, params: dict[str, Any]):
            year = int(params["year"])
            fee_id = int(params["fee_id"])
            restriction = params.get("account_type")
            if mode == "baseline_members" or restriction is None:
                return fee_affected_merchants_for_year(data, year=year, fee_id=fee_id)
            if mode == "losers_only":
                return fee_affected_merchants_for_year(
                    data, year=year, fee_id=fee_id, only_account_type=restriction
                )
            fee_row = _fee_by_id(data.fees, fee_id)
            modified = dict(fee_row)
            modified["account_type"] = [restriction]
            context = _yearly_payment_context(data, year=year)
            before = _merchants_matching_fee_context(context, fee_row)
            after = _merchants_matching_fee_context(context, modified)
            return sorted(before.symmetric_difference(after))

        return affected

    if primitive == "applicable_fee_ids_period":
        if p.tuple_scope == "sampled_first_day":
            return _applicable_ids_sampled_first_day
        return lambda data, params: applicable_fee_ids_for_merchant_period(
            data, merchant=params["merchant"], year=_year_param(data, params),
            month=int(params["month"]) if params.get("month") else None,
            day_of_year=params.get("day_of_year"),
        )

    if primitive == "steer_optimal_aci":
        policy = p.aci_candidate_policy or "exclude_current"
        reducer = p.reducer

        def steer(data: DABStepData, params: dict[str, Any]):
            year = _year_param(data, params)
            if policy == "exclude_current" and reducer == "sum_all_matching":
                result = optimize_aci_for_fraudulent_transactions(
                    data, merchant=params["merchant"], year=year, month=params.get("month")
                )
                return str(result["aci"])
            return _steer_aci_variant(
                data, merchant=params["merchant"], year=year,
                month=params.get("month"), exclude_current=(policy == "exclude_current"),
                reducer=reducer,
            )

        return steer

    if primitive == "field_domain_values":
        return lambda data, params: field_domain_values(
            str(params.get("field", "account_type")), frames=[data.fees, data.merchants]
        )

    if primitive == "unique_merchant_count":
        return lambda data, params: unique_merchant_count(data.payments, merchant_profiles=data.merchants)

    if primitive == "repeat_customer_percentage":
        return lambda data, params: repeat_customer_percentage(
            data.payments, identity_field="email_address",
            high_value_quantile=float(params["percentile"]) / 100,
        )

    raise SpecNotExecutable(f"unknown primitive {primitive}")


def _month_slice(data: DABStepData, merchant: str, year: int, month: int):
    """Memoized (payments, context_base) for a merchant-month."""
    def compute():
        payments = _merchant_month_payments(data, merchant=merchant, year=year, month=month)
        merchant_row = _merchant_by_name(data.merchants, merchant)
        context_base = _merchant_month_context(data, merchant_row=merchant_row, payments=payments)
        return payments, context_base
    return memo_get_or_compute(("slice", id(data), merchant, year, month), compute)


def _month_range(year: int, month: int | None) -> list[int]:
    return [month] if month else list(range(1, 13))


def _year_param(data: DABStepData, params: dict[str, Any]) -> int:
    """Year from the question, else the dataset's single observed year."""
    if params.get("year") is not None:
        return int(params["year"])
    years = sorted({int(y) for y in data.payments["year"].dropna().unique()})
    if len(years) != 1:
        raise SpecNotExecutable("year not in question and dataset spans multiple years")
    return years[0]


def _period_fees_per_txn_reduced(
    data: DABStepData, *, merchant: str, year: int, month: int | None, reducer: str
) -> float:
    """Rejected-candidate implementations: per-transaction min/first matching rule."""
    fee_rows = data.fees.to_dict(orient="records")
    total = 0.0
    for active_month in _month_range(year, month):
        try:
            payments, context_base = _month_slice(data, merchant, year, active_month)
        except ValueError:
            continue
        if payments.empty:
            continue
        for payment in payments.to_dict(orient="records"):
            context = _payment_fee_context(context_base, payment)
            fees = [
                calculate_fee(fee["fixed_amount"], fee["rate"], payment["eur_amount"])
                for fee in fee_rows
                if _matches_fee_for_payment(fee, context)
            ]
            if not fees:
                continue
            total += min(fees) if reducer == "min_match" else fees[0]
    return total


def _fixed_component_delta_period(
    data: DABStepData, *, merchant: str, year: int, month: int | None, fee_id: int
) -> float:
    total = 0.0
    for active_month in _month_range(year, month):
        try:
            total += fee_fixed_component_delta_for_month(
                data, merchant=merchant, year=year, month=active_month,
                fee_id=fee_id, new_fixed_amount=0,
            )
        except ValueError:
            continue
    return total


def _applicable_ids_sampled_first_day(data: DABStepData, params: dict[str, Any]) -> list[int]:
    """Rejected candidate: derive payment tuples from the month's first active day only."""
    merchant = params["merchant"]
    year = int(params["year"])
    month = int(params["month"])
    payments = _merchant_month_payments(data, merchant=merchant, year=year, month=month)
    if payments.empty:
        return []
    first_day = int(payments["day_of_year"].min())
    sampled = payments[payments["day_of_year"] == first_day].copy()
    merchant_row = _merchant_by_name(data.merchants, merchant)
    context_base = _merchant_month_context(data, merchant_row=merchant_row, payments=payments)
    fee_rows = data.fees.to_dict(orient="records")
    matched: set[int] = set()
    for payment in sampled.to_dict(orient="records"):
        context = _payment_fee_context(context_base, payment)
        matched.update(int(fee["ID"]) for fee in fee_rows if _matches_fee_for_payment(fee, context))
    return sorted(matched)


def _steer_aci_variant(
    data: DABStepData, *, merchant: str, year: int, month: int | None,
    exclude_current: bool, reducer: str,
) -> str:
    acis = sorted(str(v) for v in data.payments["aci"].dropna().unique())
    fee_rows = data.fees.to_dict(orient="records")
    totals: dict[str, float] = {aci: 0.0 for aci in acis}
    current: set[str] = set()
    for active_month in _month_range(year, month):
        try:
            payments, context_base = _month_slice(data, merchant, year, active_month)
        except ValueError:
            continue
        fraud = payments[payments["has_fraudulent_dispute"].astype(bool)].copy()
        if fraud.empty:
            continue
        current |= {str(v) for v in fraud["aci"].dropna().unique()}
        for aci in acis:
            simulated = fraud.copy()
            simulated["aci"] = aci
            if reducer == "sum_all_matching":
                for group in _matching_fee_group_rows(simulated, context_base=context_base, fee_rows=fee_rows):
                    for fee in group["matched_fees"]:
                        totals[aci] += float(fee["fixed_amount"] or 0) * group["transactions"]
                        totals[aci] += float(fee["rate"] or 0) * group["amount_sum"] / 10000
            else:  # min_match per transaction
                for payment in simulated.to_dict(orient="records"):
                    context = _payment_fee_context(context_base, payment)
                    fees = [
                        calculate_fee(fee["fixed_amount"], fee["rate"], payment["eur_amount"])
                        for fee in fee_rows
                        if _matches_fee_for_payment(fee, context)
                    ]
                    if fees:
                        totals[aci] += min(fees)
    covered = {aci: cost for aci, cost in totals.items() if cost > 0.0}
    if exclude_current:
        eligible = {aci: cost for aci, cost in covered.items() if aci not in current} or covered
    else:
        eligible = covered
    if not eligible:
        raise SpecNotExecutable("no candidate ACI produced a fee estimate")
    return min(sorted(eligible), key=lambda aci: (eligible[aci], aci))


# --------------------------------------------------------------------------
# output formatting
# --------------------------------------------------------------------------


def _format_output(value: Any, output: OutputSpec, guidelines: str) -> str:
    if output.kind == "decimal":
        places = _decimal_places(guidelines, default=output.decimals_default or 6)
        return format_decimal_places(float(value), places)
    if output.kind == "integer":
        return str(int(value))
    if output.kind == "id_list":
        return ", ".join(str(item) for item in value)
    if output.kind == "string_list":
        return ", ".join(str(item) for item in value)
    if output.kind == "single_string":
        if isinstance(value, (list, tuple, set)):
            items = sorted(str(item) for item in value)
            if output.tie_policy == "list_all_sorted":
                return ", ".join(items)
            return items[0] if items else ""
        return str(value)
    raise SpecNotExecutable(f"unknown output kind {output.kind}")
