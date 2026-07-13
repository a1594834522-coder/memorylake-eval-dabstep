from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from dabstep_agent_pydantic.analysis_spec_v2 import AggregateMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import AllTimeScope
from dabstep_agent_pydantic.analysis_spec_v2 import AnalysisSpec
from dabstep_agent_pydantic.analysis_spec_v2 import ComparisonFilter
from dabstep_agent_pydantic.analysis_spec_v2 import DayRangeScope
from dabstep_agent_pydantic.analysis_spec_v2 import EqFilter
from dabstep_agent_pydantic.analysis_spec_v2 import FilterSpec
from dabstep_agent_pydantic.analysis_spec_v2 import InFilter
from dabstep_agent_pydantic.analysis_spec_v2 import InterpretationMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import MonthScope
from dabstep_agent_pydantic.analysis_spec_v2 import NullFilter
from dabstep_agent_pydantic.analysis_spec_v2 import RatioMeasure
from dabstep_agent_pydantic.analysis_spec_v2 import YearScope
from dabstep_agent_pydantic.dabstep_core import DABStepData
from dabstep_agent_pydantic.dabstep_core import get_month_day_range
from dabstep_agent_pydantic.distill.combinators import execute_spec_value
from dabstep_agent_pydantic.output_contract import format_analysis_output


_DATA_FINGERPRINT_CACHE: dict[tuple[int, ...], str] = {}


@dataclass(frozen=True)
class ExecutionResult:
    raw_value: Any
    formatted_value: str
    row_counts: dict[str, int]
    intermediates: dict[str, int | float]
    policy_ids: list[str]
    plan_fingerprint: str
    execution_fingerprint: str


def execute_analysis(data: DABStepData, spec: AnalysisSpec) -> ExecutionResult:
    frame = data[spec.source.table].copy()
    row_counts = {"source": len(frame)}

    for join in spec.joins:
        right = data[join.right].copy()
        frame = frame.merge(
            right,
            how=join.how,
            left_on=[key.left_column for key in join.keys],
            right_on=[key.right_column for key in join.keys],
            suffixes=("", f"__{join.right}"),
        )
        row_counts[f"join:{join.right}"] = len(frame)

    if not isinstance(spec.time_scope, AllTimeScope):
        frame = _apply_time_scope(frame, spec.time_scope)
        row_counts["time_scope"] = len(frame)

    if spec.filters:
        frame = _apply_filters(frame, spec.filters)
        row_counts["filters"] = len(frame)

    if isinstance(spec.measure, InterpretationMeasure):
        source_fingerprint = _data_fingerprint(data)
        raw_value = execute_spec_value(
            spec.measure.interpretation,
            data,
            dict(spec.measure.params),
        )
        raw_value = _to_builtin_tree(raw_value)
        row_counts["measure"] = len(frame)
        intermediates = {}
    elif spec.group_by:
        source_fingerprint = _frame_fingerprint(frame)
        raw_value, intermediates = _execute_grouped(frame, spec, row_counts)
    else:
        source_fingerprint = _frame_fingerprint(frame)
        raw_value, intermediates = _execute_measure(
            frame,
            spec.measure,
            row_counts=row_counts,
            row_count_prefix="measure",
        )

    formatted = format_analysis_output(
        raw_value,
        kind=spec.output.kind,
        decimals=spec.output.decimals,
        empty_string_allowed=spec.output.empty_string_allowed,
    )
    plan_fingerprint = analysis_plan_fingerprint(spec)
    execution_fingerprint = _stable_fingerprint({
        "plan_fingerprint": plan_fingerprint,
        "source_fingerprint": source_fingerprint,
        "row_counts": row_counts,
        "intermediates": intermediates,
        "raw_value": raw_value,
        "formatted_value": formatted,
    })
    return ExecutionResult(
        raw_value=raw_value,
        formatted_value=formatted,
        row_counts=row_counts,
        intermediates=intermediates,
        policy_ids=list(spec.policy_ids),
        plan_fingerprint=plan_fingerprint,
        execution_fingerprint=execution_fingerprint,
    )


def analysis_plan_fingerprint(spec: AnalysisSpec) -> str:
    return _stable_fingerprint(spec.model_dump(mode="json"))


def _apply_time_scope(frame: pd.DataFrame, scope) -> pd.DataFrame:
    if isinstance(scope, YearScope):
        _require_columns(frame, [scope.year_column])
        return frame.loc[frame[scope.year_column] == scope.year].copy()
    if isinstance(scope, MonthScope):
        _require_columns(frame, [scope.year_column, scope.day_column])
        start_day, end_day = get_month_day_range(scope.year, scope.month)
        mask = (
            (frame[scope.year_column] == scope.year)
            & frame[scope.day_column].between(start_day, end_day, inclusive="both")
        )
        return frame.loc[mask].copy()
    if isinstance(scope, DayRangeScope):
        _require_columns(frame, [scope.year_column, scope.day_column])
        mask = (
            (frame[scope.year_column] == scope.year)
            & frame[scope.day_column].between(scope.start_day, scope.end_day, inclusive="both")
        )
        return frame.loc[mask].copy()
    raise TypeError(f"unsupported time scope: {type(scope).__name__}")


def _apply_filters(frame: pd.DataFrame, filters: list[FilterSpec]) -> pd.DataFrame:
    filtered = frame
    for condition in filters:
        _require_columns(filtered, [condition.column])
        series = filtered[condition.column]
        if isinstance(condition, EqFilter):
            mask = series == condition.value if condition.op == "eq" else series != condition.value
        elif isinstance(condition, InFilter):
            mask = series.isin(condition.values)
            if condition.op == "not_in":
                mask = ~mask
        elif isinstance(condition, ComparisonFilter):
            operations = {
                "gt": series > condition.value,
                "gte": series >= condition.value,
                "lt": series < condition.value,
                "lte": series <= condition.value,
            }
            mask = operations[condition.op]
        elif isinstance(condition, NullFilter):
            mask = series.isna() if condition.op == "is_null" else series.notna()
        else:
            raise TypeError(f"unsupported filter: {type(condition).__name__}")
        filtered = filtered.loc[mask]
    return filtered.copy()


def _execute_measure(
    frame: pd.DataFrame,
    measure: AggregateMeasure | RatioMeasure,
    *,
    row_counts: dict[str, int],
    row_count_prefix: str,
) -> tuple[int | float, dict[str, int | float]]:
    if isinstance(measure, AggregateMeasure):
        value, rows = _execute_aggregate(frame, measure)
        row_counts[row_count_prefix] = rows
        return value, {}

    numerator, numerator_rows = _execute_aggregate(frame, measure.numerator)
    denominator, denominator_rows = _execute_aggregate(frame, measure.denominator)
    row_counts[f"{row_count_prefix}:numerator"] = numerator_rows
    row_counts[f"{row_count_prefix}:denominator"] = denominator_rows
    if denominator == 0:
        if measure.zero_denominator == "error":
            raise ZeroDivisionError("analysis ratio denominator is zero")
        value = 0.0
    else:
        value = float(numerator) / float(denominator) * measure.scale
    return value, {
        "numerator": float(numerator),
        "denominator": float(denominator),
    }


def _execute_aggregate(frame: pd.DataFrame, measure: AggregateMeasure) -> tuple[int | float, int]:
    population = _apply_filters(frame, measure.filters) if measure.filters else frame.copy()
    if measure.column is None:
        return len(population), len(population)

    _require_columns(population, [measure.column])
    series = population[measure.column]
    if measure.missing == "error" and series.isna().any():
        raise ValueError(f"missing values found in {measure.column}")
    if measure.missing == "exclude":
        series = series.dropna()
    elif measure.missing == "include" and measure.kind in {"sum", "mean"}:
        series = series.fillna(0)

    if measure.kind == "sum":
        return float(series.sum()), len(series)
    if measure.kind == "mean":
        return (float(series.mean()) if len(series) else 0.0), len(series)
    if measure.kind == "count":
        return len(series), len(series)
    if measure.kind == "nunique":
        return int(series.nunique(dropna=measure.missing != "include")), len(series)
    raise TypeError(f"unsupported aggregate: {measure.kind}")


def _execute_grouped(
    frame: pd.DataFrame,
    spec: AnalysisSpec,
    row_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    _require_columns(frame, list(spec.group_by))
    group_key = spec.group_by[0] if len(spec.group_by) == 1 else list(spec.group_by)
    items: list[dict[str, Any]] = []
    for raw_group, group_frame in frame.groupby(group_key, dropna=True, sort=False):
        group_values = raw_group if isinstance(raw_group, tuple) else (raw_group,)
        group = {
            column: _to_builtin(value)
            for column, value in zip(spec.group_by, group_values, strict=True)
        }
        local_counts: dict[str, int] = {}
        value, _intermediates = _execute_measure(
            group_frame,
            spec.measure,
            row_counts=local_counts,
            row_count_prefix="measure",
        )
        items.append({"group": group, "value": _to_builtin(value)})

    row_counts["groups"] = len(items)
    items.sort(key=lambda item: tuple(str(item["group"][column]) for column in spec.group_by))
    for order in reversed(spec.ordering):
        if order.by == "value":
            key = lambda item: item["value"]
        else:
            key = lambda item, column=order.by: str(item["group"][column])
        items.sort(key=key, reverse=order.direction == "desc")
    if spec.limit is not None:
        items = items[:spec.limit]
        row_counts["output_groups"] = len(items)
    return items, {}


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"analysis columns not found: {missing}")


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.to_json(
        orient="split",
        date_format="iso",
        default_handler=str,
        double_precision=15,
    )
    return _stable_fingerprint(payload)


def _data_fingerprint(data: DABStepData) -> str:
    # Runtime-loaded DABStepData frames are treated as immutable. Cache their
    # content fingerprint because one generated skill batch reuses the same
    # five frames across many concrete parameter sets.
    frames = (
        data.fees,
        data.payments,
        data.merchants,
        data.acquirer_countries,
        data.merchant_category_codes,
    )
    cache_key = tuple(id(frame) for frame in frames)
    cached = _DATA_FINGERPRINT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    fingerprint = _stable_fingerprint({
        name: _frame_fingerprint(data[name])
        for name in (
            "fees",
            "payments",
            "merchants",
            "acquirer_countries",
            "merchant_category_codes",
        )
    })
    _DATA_FINGERPRINT_CACHE[cache_key] = fingerprint
    return fingerprint


def _stable_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_builtin(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _to_builtin_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin_tree(item) for item in value]
    return _to_builtin(value)
