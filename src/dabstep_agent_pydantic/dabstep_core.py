from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


__all__ = [
    "DABStepData",
    "MonthlyMetrics",
    "aci_fee_extreme_for_scheme_transaction",
    "add_intracountry_flag",
    "applicable_fee_ids_for_merchant_period",
    "average_transaction_amount_per_unique_email",
    "average_transactions_per_unique_email",
    "average_scheme_fee",
    "calculate_fee",
    "calculate_fee_monthly_metrics",
    "calculate_monthly_metrics",
    "fee_fixed_component_delta_for_month",
    "fee_rate_delta_for_month",
    "fee_rate_delta_for_period",
    "fee_factor_monotonicity",
    "field_domain_values",
    "fee_affected_merchants_for_year",
    "format_decimal_places",
    "fraud_rate_by_group",
    "fraud_rate_by_volume",
    "get_month_day_range",
    "load_dabstep_data",
    "match_count_summary",
    "matching_fee_ids",
    "matches_bool_field",
    "matches_capture_delay",
    "matches_fee_rule",
    "matches_fraud_level",
    "matches_list_field",
    "matches_monthly_volume",
    "mcc_code_for_description",
    "merchant_mcc_fee_delta_for_year",
    "most_expensive_mccs_for_amount",
    "optimize_aci_for_fraudulent_transactions",
    "parse_fraud_value",
    "parse_volume_value",
    "repeat_customer_percentage",
    "top_fraud_ip_country_by_rate",
    "total_fees_for_merchant_period",
    "unique_merchant_count",
]


@dataclass(frozen=True)
class DABStepData:
    fees: pd.DataFrame
    payments: pd.DataFrame
    merchants: pd.DataFrame
    acquirer_countries: pd.DataFrame
    merchant_category_codes: pd.DataFrame

    def __iter__(self):
        yield self.fees
        yield self.payments
        yield self.merchants
        yield self.merchant_category_codes
        yield self.acquirer_countries

    def __getitem__(self, key: str) -> pd.DataFrame:
        fields = {
            "fees": self.fees,
            "payments": self.payments,
            "merchants": self.merchants,
            "acquirer_countries": self.acquirer_countries,
            "merchant_category_codes": self.merchant_category_codes,
            "mcc_table": self.merchant_category_codes,
        }
        return fields[key]


@dataclass(frozen=True)
class MonthlyMetrics:
    volume: float
    fraud_volume: float
    fraud_rate_pct: float


def load_dabstep_data(data_dir: str | Path) -> DABStepData:
    data_dir = Path(data_dir)
    return DABStepData(
        fees=pd.DataFrame(_read_json(data_dir / "fees.json")),
        payments=_read_csv(data_dir / "payments.csv"),
        merchants=pd.DataFrame(_read_json(data_dir / "merchant_data.json")),
        acquirer_countries=_read_csv(data_dir / "acquirer_countries.csv"),
        merchant_category_codes=_read_csv(data_dir / "merchant_category_codes.csv"),
    )


def add_intracountry_flag(payments: pd.DataFrame, acquirer_countries: pd.DataFrame) -> pd.DataFrame:
    enriched = payments.copy()
    if "acquirer_country" not in enriched.columns:
        acquirer_map = acquirer_countries.set_index("acquirer")["country_code"]
        enriched["acquirer_country"] = enriched["acquirer"].map(acquirer_map)
    enriched["intracountry"] = enriched["issuing_country"] == enriched["acquirer_country"]
    return enriched


def calculate_fee(fixed_amount: float, rate: int | float, transaction_value: float) -> float:
    return float(fixed_amount or 0) + float(rate or 0) * float(transaction_value) / 10000


def format_decimal_places(value: int | float, places: int) -> str:
    return f"{float(value):.{places}f}"


def matches_list_field(field_value: object, target: object) -> bool:
    if _is_missing(field_value):
        return True
    if isinstance(field_value, list):
        return not field_value or target in field_value
    return field_value == target


def matches_bool_field(field_value: object, target: bool) -> bool:
    if _is_missing(field_value):
        return True
    return bool(field_value) is bool(target)


def matches_capture_delay(fee_delay: object, merchant_delay: object) -> bool:
    if _is_missing(fee_delay):
        return True
    text = str(fee_delay).strip()
    target_text = str(merchant_delay).strip()
    if not text:
        return True
    if text in {"immediate", "manual"}:
        return target_text == text
    try:
        target = float(target_text)
    except ValueError:
        return target_text == text
    if text.startswith("<"):
        return target < float(text[1:])
    if text.startswith(">"):
        return target > float(text[1:])
    range_match = re.fullmatch(r"(.+?)\s*-\s*(.+)", text)
    if range_match:
        return float(range_match.group(1)) <= target <= float(range_match.group(2))
    return target_text == text


def matches_fee_rule(rule: dict[str, object], context: dict[str, object]) -> bool:
    return (
        _matches_if_present(context, "account_type", lambda value: matches_list_field(rule.get("account_type"), value))
        and _matches_if_present(context, "aci", lambda value: matches_list_field(rule.get("aci"), value))
        and _matches_if_present(context, "is_credit", lambda value: matches_bool_field(rule.get("is_credit"), bool(value)))
        and _matches_if_present(
            context,
            "capture_delay",
            lambda value: matches_capture_delay(rule.get("capture_delay"), value),
        )
        and _matches_if_present(
            context,
            "merchant_category_code",
            lambda value: matches_list_field(rule.get("merchant_category_code"), value),
        )
        and _matches_if_present(
            context,
            "monthly_volume",
            lambda value: matches_monthly_volume(rule.get("monthly_volume"), float(value or 0)),
        )
        and _matches_if_present(
            context,
            "monthly_fraud_rate_pct",
            lambda value: matches_fraud_level(rule.get("monthly_fraud_level"), float(value or 0)),
        )
        and _matches_if_present(
            context,
            "intracountry",
            lambda value: matches_bool_field(rule.get("intracountry"), bool(value)),
        )
    )


def matching_fee_ids(fees: pd.DataFrame, context: dict[str, object]) -> list[int]:
    matches = [
        int(row["ID"])
        for row in fees.to_dict(orient="records")
        if matches_fee_rule(row, context)
    ]
    return sorted(matches)


def applicable_fee_ids_for_merchant_period(
    data: DABStepData | None = None,
    *,
    merchant: str,
    year: int,
    month: int | None = None,
    day_of_year: int | None = None,
    fees: pd.DataFrame | list[dict[str, object]] | None = None,
    payments: pd.DataFrame | None = None,
    merchants: pd.DataFrame | list[dict[str, object]] | None = None,
    acquirer_countries: pd.DataFrame | None = None,
    mcc_table: pd.DataFrame | None = None,
    merchant_category_codes: pd.DataFrame | None = None,
    period_type: str | None = None,
    fraud_metric: str = "volume",
) -> list[int]:
    del period_type
    data = data or _coerce_dabstep_data(
        fees=fees,
        payments=payments,
        merchants=merchants,
        acquirer_countries=acquirer_countries,
        merchant_category_codes=merchant_category_codes if merchant_category_codes is not None else mcc_table,
    )
    if month is None:
        if day_of_year is None:
            raise ValueError("month is required when day_of_year is not provided")
        month = _month_from_day_of_year(year, day_of_year)
    month_payments = _merchant_month_payments(data, merchant=merchant, year=year, month=month)
    target_payments = month_payments
    if day_of_year is not None:
        target_payments = month_payments[month_payments["day_of_year"] == day_of_year].copy()
    if target_payments.empty:
        return []

    merchant_row = _merchant_by_name(data.merchants, merchant)
    context_base = _merchant_month_context(
        data,
        merchant_row=merchant_row,
        payments=month_payments,
        fraud_metric=fraud_metric,
    )
    matches: set[int] = set()
    fee_rows = data.fees.to_dict(orient="records")
    for payment in target_payments.to_dict(orient="records"):
        context = _payment_fee_context(context_base, payment)
        matches.update(int(fee["ID"]) for fee in fee_rows if _matches_fee_for_payment(fee, context))
    return sorted(matches)


def fee_rate_delta_for_month(
    data: DABStepData,
    *,
    merchant: str,
    year: int,
    month: int,
    fee_id: int,
    new_rate: int | float,
) -> float:
    fee_row = _fee_by_id(data.fees, fee_id)
    payments = _merchant_month_payments(data, merchant=merchant, year=year, month=month)
    merchant_row = _merchant_by_name(data.merchants, merchant)
    context_base = _merchant_month_context(data, merchant_row=merchant_row, payments=payments)

    delta = 0.0
    for payment in payments.to_dict(orient="records"):
        context = _payment_fee_context(context_base, payment)
        if _matches_fee_for_payment(fee_row, context):
            delta += calculate_fee(fee_row["fixed_amount"], new_rate, payment["eur_amount"])
            delta -= calculate_fee(fee_row["fixed_amount"], fee_row["rate"], payment["eur_amount"])
    return delta


def fee_rate_delta_for_period(
    data: DABStepData,
    *,
    merchant: str,
    year: int,
    fee_id: int,
    new_rate: int | float,
    month: int | None = None,
) -> float:
    months = [month] if month is not None else list(range(1, 13))
    total = 0.0
    for active_month in months:
        if active_month is None:
            continue
        try:
            total += fee_rate_delta_for_month(
                data,
                merchant=merchant,
                year=year,
                month=active_month,
                fee_id=fee_id,
                new_rate=new_rate,
            )
        except ValueError:
            continue
    return total


def fee_affected_merchants_for_year(
    data: DABStepData,
    *,
    year: int,
    fee_id: int,
    only_account_type: str | None = None,
) -> list[str]:
    fee_row = _fee_by_id(data.fees, fee_id)
    yearly_context = _yearly_payment_context(data, year=year)
    original_merchants = _merchants_matching_fee_context(yearly_context, fee_row)
    if only_account_type is None:
        return sorted(original_merchants)

    modified_fee = dict(fee_row)
    modified_fee["account_type"] = [only_account_type]
    modified_merchants = _merchants_matching_fee_context(yearly_context, modified_fee)
    # "Affected by this change" counts merchants that lose the fee under the
    # restriction; merchants that would newly gain it are not affected today.
    return sorted(original_merchants - modified_merchants)


def _yearly_payment_context(data: DABStepData, *, year: int) -> pd.DataFrame:
    payments = data.payments[data.payments["year"] == year].copy()
    if payments.empty:
        return payments

    if "acquirer" not in payments.columns and "acquirer_country" not in payments.columns:
        acquirer_by_merchant = data.merchants.set_index("merchant")["acquirer"].map(_first_value)
        payments["acquirer"] = payments["merchant"].map(acquirer_by_merchant)
    payments = add_intracountry_flag(payments, data.acquirer_countries)
    payments["month"] = _month_series_from_day_of_year(year, payments["day_of_year"])

    merchant_columns = [
        column
        for column in ("merchant", "account_type", "capture_delay", "merchant_category_code")
        if column in data.merchants.columns
    ]
    payments = payments.merge(
        data.merchants[merchant_columns].drop_duplicates("merchant"),
        on="merchant",
        how="left",
    )

    fraud_mask = (
        payments["has_fraudulent_dispute"].astype(bool)
        if "has_fraudulent_dispute" in payments.columns
        else pd.Series(False, index=payments.index)
    )
    payments["_fraud_eur_amount"] = payments["eur_amount"].where(fraud_mask, 0.0)
    monthly = (
        payments.groupby(["merchant", "month"], dropna=True)
        .agg(monthly_volume=("eur_amount", "sum"), fraud_volume=("_fraud_eur_amount", "sum"))
        .reset_index()
    )
    monthly["monthly_fraud_rate_pct"] = (
        monthly["fraud_volume"].div(monthly["monthly_volume"]).fillna(0.0) * 100
    )
    payments = payments.merge(
        monthly[["merchant", "month", "monthly_volume", "monthly_fraud_rate_pct"]],
        on=["merchant", "month"],
        how="left",
    )
    return payments.drop(columns=["_fraud_eur_amount"], errors="ignore")


def _month_series_from_day_of_year(year: int, day_of_year: pd.Series) -> pd.Series:
    month_ends: list[int] = []
    running_end = 0
    for month in range(1, 13):
        running_end += calendar.monthrange(year, month)[1]
        month_ends.append(running_end)
    return pd.cut(
        day_of_year.astype(int),
        bins=[0, *month_ends],
        labels=list(range(1, 13)),
        include_lowest=True,
    ).astype(int)


def _merchants_matching_fee_context(payments: pd.DataFrame, fee: dict[str, object]) -> set[str]:
    if payments.empty or "merchant" not in payments.columns:
        return set()
    mask = payments["card_scheme"].eq(fee.get("card_scheme"))
    mask &= _list_rule_mask(payments, "account_type", fee.get("account_type"))
    mask &= _list_rule_mask(payments, "aci", fee.get("aci"))
    mask &= _bool_rule_mask(payments, "is_credit", fee.get("is_credit"))
    mask &= _capture_delay_rule_mask(payments, "capture_delay", fee.get("capture_delay"))
    mask &= _list_rule_mask(payments, "merchant_category_code", fee.get("merchant_category_code"))
    mask &= _range_rule_mask(payments, "monthly_volume", fee.get("monthly_volume"), parser=parse_volume_value)
    mask &= _range_rule_mask(
        payments,
        "monthly_fraud_rate_pct",
        fee.get("monthly_fraud_level"),
        parser=parse_fraud_value,
    )
    mask &= _bool_rule_mask(payments, "intracountry", fee.get("intracountry"))
    return {
        merchant
        for merchant in payments.loc[mask.fillna(False), "merchant"].dropna().astype(str)
        if merchant.strip()
    }


def _true_mask(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=frame.index)


def _false_mask(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=frame.index)


def _list_rule_mask(frame: pd.DataFrame, column: str, rule: object) -> pd.Series:
    if _is_missing(rule):
        return _true_mask(frame)
    if column not in frame.columns:
        return _false_mask(frame)
    if isinstance(rule, list):
        if not rule:
            return _true_mask(frame)
        return frame[column].isin(rule)
    return frame[column].eq(rule)


def _bool_rule_mask(frame: pd.DataFrame, column: str, rule: object) -> pd.Series:
    if _is_missing(rule):
        return _true_mask(frame)
    if column not in frame.columns:
        return _false_mask(frame)
    return frame[column].astype(bool).eq(bool(rule))


def _capture_delay_rule_mask(frame: pd.DataFrame, column: str, rule: object) -> pd.Series:
    if _is_missing(rule):
        return _true_mask(frame)
    if column not in frame.columns:
        return _false_mask(frame)
    return frame[column].map(lambda value: matches_capture_delay(rule, value))


def _range_rule_mask(frame: pd.DataFrame, column: str, rule: object, *, parser) -> pd.Series:
    if _is_missing(rule):
        return _true_mask(frame)
    if column not in frame.columns:
        return _false_mask(frame)
    text = str(rule).strip()
    if not text:
        return _true_mask(frame)
    values = pd.to_numeric(frame[column], errors="coerce")
    if text.startswith("<"):
        return values < parser(text[1:])
    if text.startswith(">"):
        return values > parser(text[1:])
    range_match = re.fullmatch(r"(.+?)\s*-\s*(.+)", text)
    if range_match:
        lower = parser(range_match.group(1))
        upper = parser(range_match.group(2))
        return values.between(lower, upper)
    return values.eq(parser(text))


def fee_fixed_component_delta_for_month(
    data: DABStepData,
    *,
    merchant: str,
    year: int,
    month: int,
    fee_id: int,
    new_fixed_amount: int | float,
) -> float:
    fee_row = _fee_by_id(data.fees, fee_id)
    payments = _merchant_month_payments(data, merchant=merchant, year=year, month=month)
    merchant_row = _merchant_by_name(data.merchants, merchant)
    context_base = _merchant_month_context(data, merchant_row=merchant_row, payments=payments)

    delta = 0.0
    for payment in payments.to_dict(orient="records"):
        context = _payment_fee_context(context_base, payment)
        if _matches_fee_for_payment(fee_row, context):
            delta += calculate_fee(new_fixed_amount, fee_row["rate"], payment["eur_amount"])
            delta -= calculate_fee(fee_row["fixed_amount"], fee_row["rate"], payment["eur_amount"])
    return delta



def _learn_memoized(name: str, key: tuple, compute):
    """Learn-run scoped memo for pure primitives (no-op outside learn)."""
    from dabstep_agent_pydantic.runtime_memo import memo_get_or_compute
    return memo_get_or_compute((name, *key), compute)

def total_fees_for_merchant_period(
    data: DABStepData,
    *,
    merchant: str,
    year: int,
    month: int | None = None,
    day_of_year: int | None = None,
    merchant_category_code: int | None = None,
    card_scheme: str | None = None,
    aci: str | None = None,
    only_fraudulent: bool = False,
    fraud_metric: str = "volume",
) -> float:
    return _learn_memoized(
        "total_fees",
        (id(data), merchant, year, month, day_of_year, merchant_category_code, card_scheme, aci, only_fraudulent, fraud_metric),
        lambda: _total_fees_for_merchant_period_impl(
            data, merchant=merchant, year=year, month=month, day_of_year=day_of_year,
            merchant_category_code=merchant_category_code, card_scheme=card_scheme, aci=aci,
            only_fraudulent=only_fraudulent, fraud_metric=fraud_metric,
        ),
    )


def _total_fees_for_merchant_period_impl(
    data: DABStepData,
    *,
    merchant: str,
    year: int,
    month: int | None = None,
    day_of_year: int | None = None,
    merchant_category_code: int | None = None,
    card_scheme: str | None = None,
    aci: str | None = None,
    only_fraudulent: bool = False,
    fraud_metric: str = "volume",
) -> float:
    months = [_month_from_day_of_year(year, day_of_year)] if day_of_year is not None else ([month] if month else list(range(1, 13)))
    merchant_row = _merchant_by_name(data.merchants, merchant)
    total = 0.0
    fee_rows = data.fees.to_dict(orient="records")
    for active_month in months:
        if active_month is None:
            continue
        try:
            month_payments = _merchant_month_payments(data, merchant=merchant, year=year, month=active_month)
        except ValueError:
            continue
        target_payments = month_payments
        if day_of_year is not None:
            target_payments = month_payments[month_payments["day_of_year"] == day_of_year].copy()
        if only_fraudulent and "has_fraudulent_dispute" in target_payments:
            target_payments = target_payments[target_payments["has_fraudulent_dispute"].astype(bool)].copy()
        if target_payments.empty:
            continue

        context_base = _merchant_month_context(
            data,
            merchant_row=merchant_row,
            payments=month_payments,
            fraud_metric=fraud_metric,
        )
        if merchant_category_code is not None:
            context_base["merchant_category_code"] = merchant_category_code
        simulated_payments = target_payments.copy()
        if card_scheme is not None:
            simulated_payments["card_scheme"] = card_scheme
        if aci is not None:
            simulated_payments["aci"] = aci
        total += _sum_matching_fees_for_payment_groups(
            simulated_payments,
            context_base=context_base,
            fee_rows=fee_rows,
        )
    return total


def merchant_mcc_fee_delta_for_year(
    data: DABStepData,
    *,
    merchant: str,
    year: int,
    new_mcc: int,
    fraud_metric: str = "volume",
) -> float:
    original_total = total_fees_for_merchant_period(
        data,
        merchant=merchant,
        year=year,
        fraud_metric=fraud_metric,
    )
    counterfactual_total = total_fees_for_merchant_period(
        data,
        merchant=merchant,
        year=year,
        merchant_category_code=new_mcc,
        fraud_metric=fraud_metric,
    )
    return counterfactual_total - original_total


def optimize_aci_for_fraudulent_transactions(
    data: DABStepData,
    *,
    merchant: str,
    year: int,
    month: int | None = None,
    candidate_acis: list[str] | None = None,
) -> dict[str, object]:
    merchant_row = _merchant_by_name(data.merchants, merchant)
    acis = candidate_acis or sorted(str(value) for value in data.payments["aci"].dropna().unique())
    totals = {aci: 0.0 for aci in acis}
    current_acis: set[str] = set()

    months = [month] if month is not None else list(range(1, 13))
    fee_rows = data.fees.to_dict(orient="records")
    for active_month in months:
        if active_month is None:
            continue
        try:
            payments = _merchant_month_payments(data, merchant=merchant, year=year, month=active_month)
        except ValueError:
            continue
        fraud_payments = payments[payments["has_fraudulent_dispute"].astype(bool)].copy()
        if fraud_payments.empty:
            continue
        current_acis |= {str(value) for value in fraud_payments["aci"].dropna().unique()}
        context_base = _merchant_month_context(data, merchant_row=merchant_row, payments=payments)
        for aci in acis:
            simulated_payments = fraud_payments.copy()
            simulated_payments["aci"] = aci
            totals[aci] += _sum_matching_fees_for_payment_groups(
                simulated_payments,
                context_base=context_base,
                fee_rows=fee_rows,
            )

    # Moving to an ACI whose transactions match no fee rule is not a real option,
    # and the question asks for a *different* ACI than the fraudulent traffic uses.
    covered = {aci: cost for aci, cost in totals.items() if cost > 0.0}
    eligible = {aci: cost for aci, cost in covered.items() if aci not in current_acis} or covered

    best: dict[str, object] | None = None
    for aci, cost in eligible.items():
        candidate = {"aci": aci, "cost": cost, "formatted": f"{aci}:{cost:.2f}"}
        if best is None or cost < float(best["cost"]):
            best = candidate

    if best is None:
        raise ValueError("no candidate ACI produced a fee estimate")
    return best


def top_fraud_ip_country_by_rate(payments: pd.DataFrame) -> str:
    best_country = None
    best_rate = -1.0
    for country, group in payments.groupby("ip_country"):
        metrics = calculate_monthly_metrics(group)
        if metrics.fraud_rate_pct > best_rate:
            best_country = country
            best_rate = metrics.fraud_rate_pct
    return str(best_country) if best_country is not None else ""


def average_scheme_fee(
    fees: pd.DataFrame,
    *,
    card_scheme: str,
    transaction_value: float,
    is_credit: bool | None = None,
    account_type: str | None = None,
    merchant_category_code: int | None = None,
) -> float:
    filtered = fees[fees["card_scheme"] == card_scheme]
    if is_credit is not None:
        filtered = filtered[filtered["is_credit"].isna() | (filtered["is_credit"] == is_credit)]
    if account_type is not None:
        filtered = filtered[filtered["account_type"].apply(lambda value: matches_list_field(value, account_type))]
    if merchant_category_code is not None:
        filtered = filtered[
            filtered["merchant_category_code"].apply(lambda value: matches_list_field(value, merchant_category_code))
        ]
    values = [
        calculate_fee(row["fixed_amount"], row["rate"], transaction_value)
        for row in filtered.to_dict(orient="records")
    ]
    if not values:
        raise ValueError("no matching fee rules")
    return sum(values) / len(values)


def aci_fee_extreme_for_scheme_transaction(
    fees: pd.DataFrame,
    *,
    card_scheme: str,
    transaction_value: float,
    is_credit: bool | None = None,
    objective: str = "max",
) -> dict[str, object]:
    filtered = fees[fees["card_scheme"].astype(str).str.lower() == card_scheme.lower()]
    if is_credit is not None:
        filtered = filtered[filtered["is_credit"].isna() | (filtered["is_credit"] == is_credit)]
    candidates = sorted(
        {
            str(aci)
            for values in filtered["aci"]
            if isinstance(values, list)
            for aci in values
            if not _is_missing(aci)
        }
    )
    if not candidates:
        raise ValueError("no ACI candidates found")

    totals: dict[str, float] = {}
    for aci in candidates:
        context = {"aci": aci}
        if is_credit is not None:
            context["is_credit"] = is_credit
        total = 0.0
        for fee in filtered.to_dict(orient="records"):
            if matches_fee_rule(fee, context):
                total += calculate_fee(fee["fixed_amount"], fee["rate"], transaction_value)
        totals[aci] = total

    if objective == "min":
        selected = sorted(totals.items(), key=lambda item: (item[1], item[0]))[0]
    elif objective == "max":
        selected = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[0]
    else:
        raise ValueError(f"unsupported objective: {objective}")
    return {"aci": selected[0], "cost": selected[1], "formatted": f"{selected[0]}:{selected[1]:.2f}"}


def unique_merchant_count(payments: pd.DataFrame, *, merchant_profiles: pd.DataFrame | None = None) -> int:
    del merchant_profiles
    if "merchant" not in payments.columns:
        raise ValueError("merchant column not found")
    return int(payments["merchant"].dropna().nunique())


def most_expensive_mccs_for_amount(data: DABStepData, *, amount: float) -> list[int]:
    """MCCs with the highest mean fee for a hypothetical transaction of `amount` EUR.

    An empty merchant_category_code list is a wildcard that applies to every MCC
    observed across the fee schedule; the per-MCC cost is the mean of
    fixed_amount + rate * amount / 10000 over all applicable rules, and ties are
    returned in ascending order.
    """
    fee_rows = data.fees.to_dict(orient="records")
    universe = sorted({
        int(mcc)
        for fee in fee_rows
        for mcc in (fee.get("merchant_category_code") or [])
    })
    per_mcc: dict[int, list[float]] = {}
    for fee in fee_rows:
        mccs = fee.get("merchant_category_code") or universe
        cost = calculate_fee(fee["fixed_amount"], fee["rate"], amount)
        for mcc in mccs:
            per_mcc.setdefault(int(mcc), []).append(cost)
    if not per_mcc:
        raise ValueError("no fee rule provides MCC coverage")
    means = {mcc: sum(values) / len(values) for mcc, values in per_mcc.items()}
    top = max(means.values())
    return sorted(mcc for mcc, value in means.items() if abs(value - top) < 1e-12)


def mcc_code_for_description(mcc_table: pd.DataFrame, description: str) -> int:
    description_text = description.strip().lower()
    matches = mcc_table[
        mcc_table["description"].astype(str).str.lower().str.contains(description_text, regex=False, na=False)
    ]
    if matches.empty:
        raise ValueError(f"MCC description not found: {description}")
    return int(matches.iloc[0]["mcc"])


def _fee_by_id(fees: pd.DataFrame, fee_id: int) -> dict[str, object]:
    rows = fees[fees["ID"] == fee_id]
    if rows.empty:
        raise ValueError(f"fee ID not found: {fee_id}")
    return rows.iloc[0].to_dict()


def _coerce_dabstep_data(
    *,
    fees: pd.DataFrame | list[dict[str, object]] | None,
    payments: pd.DataFrame | None,
    merchants: pd.DataFrame | list[dict[str, object]] | None,
    acquirer_countries: pd.DataFrame | None,
    merchant_category_codes: pd.DataFrame | None,
) -> DABStepData:
    if fees is None or payments is None or merchants is None or acquirer_countries is None:
        raise ValueError("data or fees/payments/merchants/acquirer_countries must be provided")
    return DABStepData(
        fees=fees if isinstance(fees, pd.DataFrame) else pd.DataFrame(fees),
        payments=payments,
        merchants=merchants if isinstance(merchants, pd.DataFrame) else pd.DataFrame(merchants),
        acquirer_countries=acquirer_countries,
        merchant_category_codes=merchant_category_codes if merchant_category_codes is not None else pd.DataFrame(),
    )


def _merchant_by_name(merchants: pd.DataFrame, merchant: str) -> dict[str, object]:
    rows = merchants[merchants["merchant"] == merchant]
    if rows.empty:
        raise ValueError(f"merchant not found: {merchant}")
    return rows.iloc[0].to_dict()


def _merchant_month_payments(data: DABStepData, *, merchant: str, year: int, month: int) -> pd.DataFrame:
    start_day, end_day = get_month_day_range(year, month)
    payments = data.payments[
        (data.payments["merchant"] == merchant)
        & (data.payments["year"] == year)
        & (data.payments["day_of_year"].between(start_day, end_day))
    ].copy()
    if payments.empty:
        raise ValueError(f"no payments for merchant={merchant} year={year} month={month}")
    if "acquirer" not in payments.columns and "acquirer_country" not in payments.columns:
        merchant_row = _merchant_by_name(data.merchants, merchant)
        payments["acquirer"] = _first_value(merchant_row["acquirer"])
    return add_intracountry_flag(payments, data.acquirer_countries)


def _merchant_month_context(
    data: DABStepData,
    *,
    merchant_row: dict[str, object],
    payments: pd.DataFrame,
    fraud_metric: str = "volume",
) -> dict[str, object]:
    if fraud_metric == "count":
        metrics = calculate_fee_monthly_metrics(payments)
    elif fraud_metric == "volume":
        metrics = calculate_monthly_metrics(payments)
    else:
        raise ValueError("fraud_metric must be 'count' or 'volume'")
    return {
        "account_type": merchant_row.get("account_type"),
        "capture_delay": merchant_row.get("capture_delay"),
        "merchant_category_code": merchant_row.get("merchant_category_code"),
        "monthly_volume": metrics.volume,
        "monthly_fraud_rate_pct": metrics.fraud_rate_pct,
    }


def _payment_fee_context(context_base: dict[str, object], payment: dict[str, object]) -> dict[str, object]:
    return {
        **context_base,
        "card_scheme": payment.get("card_scheme"),
        "is_credit": payment.get("is_credit"),
        "aci": payment.get("aci"),
        "intracountry": bool(payment.get("intracountry")),
    }


def _matches_fee_for_payment(fee: dict[str, object], context: dict[str, object]) -> bool:
    return (
        fee.get("card_scheme") == context.get("card_scheme")
        and matches_fee_rule(fee, context)
    )


def _sum_matching_fees_for_payment_groups(
    payments: pd.DataFrame,
    *,
    context_base: dict[str, object],
    fee_rows: list[dict[str, object]],
) -> float:
    total = 0.0
    for group in _matching_fee_group_rows(payments, context_base=context_base, fee_rows=fee_rows):
        count = int(group["transactions"])
        amount_sum = float(group["amount_sum"])
        for fee in group["matched_fees"]:
            total += float(fee["fixed_amount"] or 0) * count
            total += float(fee["rate"] or 0) * amount_sum / 10000
    return total


def match_count_summary(
    payments: pd.DataFrame,
    fees: pd.DataFrame | list[dict[str, object]],
    *,
    context_base: dict[str, object] | None = None,
) -> dict[str, object]:
    fee_rows = fees.to_dict(orient="records") if isinstance(fees, pd.DataFrame) else list(fees)
    rows = _matching_fee_group_rows(payments, context_base=context_base or {}, fee_rows=fee_rows)
    transactions = sum(int(row["transactions"]) for row in rows)
    zero_match = sum(int(row["transactions"]) for row in rows if int(row["match_count"]) == 0)
    single_match = sum(int(row["transactions"]) for row in rows if int(row["match_count"]) == 1)
    multi_match = sum(int(row["transactions"]) for row in rows if int(row["match_count"]) > 1)
    max_matches = max((int(row["match_count"]) for row in rows), default=0)
    warning = None
    if zero_match > 0:
        warning = f"{zero_match} transactions match no fee rule — check wildcard handling (null/[] means match-all)"
    elif multi_match == 0 and transactions > 0:
        warning = "no transaction matches more than one rule — filters may be too strict"
    return {
        "transactions": transactions,
        "zero_match": zero_match,
        "single_match": single_match,
        "multi_match": multi_match,
        "max_matches": max_matches,
        "warning": warning,
    }


def _matching_fee_group_rows(
    payments: pd.DataFrame,
    *,
    context_base: dict[str, object],
    fee_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if payments.empty:
        return []
    group_columns = ["card_scheme", "is_credit", "aci", "intracountry"]
    working = payments.copy()
    if "eur_amount" not in working.columns:
        working["eur_amount"] = 0.0
    grouped = (
        working.groupby(group_columns, dropna=False)["eur_amount"]
        .agg(["count", "sum"])
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for group in grouped.to_dict(orient="records"):
        context = {
            **context_base,
            "card_scheme": group.get("card_scheme"),
            "is_credit": group.get("is_credit"),
            "aci": group.get("aci"),
            "intracountry": bool(group.get("intracountry")),
        }
        matched_fees = [fee for fee in fee_rows if _matches_fee_for_payment(fee, context)]
        rows.append(
            {
                "transactions": int(group["count"]),
                "amount_sum": float(group["sum"]),
                "match_count": len(matched_fees),
                "matched_fees": matched_fees,
            }
        )
    return rows


def _first_value(value: object) -> object:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def get_month_day_range(year: int, month: int) -> tuple[int, int]:
    start = sum(calendar.monthrange(year, prior_month)[1] for prior_month in range(1, month)) + 1
    end = start + calendar.monthrange(year, month)[1] - 1
    return start, end


def _month_from_day_of_year(year: int, day_of_year: int) -> int:
    running_end = 0
    for month in range(1, 13):
        running_end += calendar.monthrange(year, month)[1]
        if day_of_year <= running_end:
            return month
    raise ValueError(f"day_of_year out of range for {year}: {day_of_year}")


def calculate_monthly_metrics(df: pd.DataFrame) -> MonthlyMetrics:
    volume = float(df["eur_amount"].sum()) if not df.empty else 0.0
    fraud_mask = df["has_fraudulent_dispute"].astype(bool) if "has_fraudulent_dispute" in df else False
    fraud_volume = float(df.loc[fraud_mask, "eur_amount"].sum()) if volume else 0.0
    fraud_rate_pct = fraud_volume / volume * 100 if volume else 0.0
    return MonthlyMetrics(volume=volume, fraud_volume=fraud_volume, fraud_rate_pct=fraud_rate_pct)


def calculate_fee_monthly_metrics(df: pd.DataFrame) -> MonthlyMetrics:
    volume = float(df["eur_amount"].sum()) if not df.empty else 0.0
    fraud_mask = df["has_fraudulent_dispute"].astype(bool) if "has_fraudulent_dispute" in df else False
    fraud_volume = float(df.loc[fraud_mask, "eur_amount"].sum()) if volume else 0.0
    fraud_count = int(fraud_mask.sum()) if hasattr(fraud_mask, "sum") else 0
    fraud_rate_pct = fraud_count / len(df) * 100 if len(df) else 0.0
    return MonthlyMetrics(volume=volume, fraud_volume=fraud_volume, fraud_rate_pct=fraud_rate_pct)


def fraud_rate_by_volume(payments: pd.DataFrame) -> float:
    return calculate_monthly_metrics(payments).fraud_rate_pct


def fraud_rate_by_group(payments: pd.DataFrame, group_by: str) -> dict[str, float]:
    return {
        str(group): fraud_rate_by_volume(group_frame)
        for group, group_frame in payments.groupby(group_by, dropna=True)
    }


def average_transactions_per_unique_email(payments: pd.DataFrame) -> float:
    email_payments = _non_null_email_payments(payments)
    unique_emails = email_payments["email_address"].nunique(dropna=True)
    return float(len(email_payments) / unique_emails) if unique_emails else 0.0


def average_transaction_amount_per_unique_email(payments: pd.DataFrame) -> float:
    email_payments = _non_null_email_payments(payments)
    if email_payments.empty:
        return 0.0
    return float(email_payments.groupby("email_address")["eur_amount"].mean().mean())


def repeat_customer_percentage(
    payments: pd.DataFrame,
    *,
    identity_field: str = "email_address",
    high_value_quantile: float | None = None,
) -> float:
    if identity_field not in payments.columns:
        raise ValueError(f"identity field not found: {identity_field}")
    population = payments
    if high_value_quantile is not None:
        threshold = payments["eur_amount"].quantile(high_value_quantile)
        population = payments[payments["eur_amount"] > threshold]
    counts = payments[identity_field].dropna().value_counts()
    repeat_values = set(counts[counts > 1].index)
    denominator = len(population)
    if denominator == 0:
        return 0.0
    return float(population[identity_field].isin(repeat_values).sum() / denominator * 100)


def field_domain_values(
    field_name: str,
    *,
    manual_values: list[object] | None = None,
    frames: list[pd.DataFrame] | None = None,
) -> list[str]:
    values: set[str] = {
        str(value)
        for value in [*_default_manual_values(field_name), *(manual_values or [])]
        if not _is_missing(value)
    }
    for frame in frames or []:
        if field_name not in frame.columns:
            continue
        for value in frame[field_name].dropna():
            if isinstance(value, list):
                values.update(str(item) for item in value if not _is_missing(item))
            else:
                values.add(str(value))
    return sorted(values)


def _default_manual_values(field_name: str) -> list[object]:
    if field_name == "account_type":
        return ["R", "D", "H", "F", "S", "O"]
    return []


def fee_factor_monotonicity() -> dict[str, list[str]]:
    return {
        "decrease_makes_cheaper": ["monthly_fraud_level"],
        "higher_tier_makes_more_expensive": ["monthly_fraud_level"],
        "formula_operands": ["fixed_amount", "rate", "eur_amount"],
    }


def matches_monthly_volume(rule_vol: object, monthly_vol: float) -> bool:
    return _matches_range_rule(rule_vol, monthly_vol, parser=parse_volume_value)


def matches_fraud_level(rule_fraud: object, fraud_pct: float) -> bool:
    return _matches_range_rule(rule_fraud, fraud_pct, parser=parse_fraud_value)


def parse_volume_value(value: object) -> float:
    text = str(value).strip().lower().replace(",", "")
    multiplier = 1.0
    if text.endswith("k"):
        multiplier = 1_000.0
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000.0
        text = text[:-1]
    return float(text) * multiplier


def parse_fraud_value(value: object) -> float:
    text = str(value).strip().replace("%", "")
    return float(text)


def _matches_range_rule(rule: object, value: float, *, parser) -> bool:
    if _is_missing(rule):
        return True
    text = str(rule).strip()
    if not text:
        return True
    if text.startswith("<"):
        return value < parser(text[1:])
    if text.startswith(">"):
        return value > parser(text[1:])

    range_match = re.fullmatch(r"(.+?)\s*-\s*(.+)", text)
    if range_match:
        lower = parser(range_match.group(1))
        upper = parser(range_match.group(2))
        return lower <= value <= upper

    return value == parser(text)


def _matches_if_present(context: dict[str, object], key: str, matcher) -> bool:
    if key not in context or context[key] is None:
        return True
    return matcher(context[key])


def _non_null_email_payments(payments: pd.DataFrame) -> pd.DataFrame:
    if "email_address" not in payments.columns:
        raise ValueError("email_address column not found")
    return payments[payments["email_address"].notna()].copy()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    drop_columns = [column for column in frame.columns if not str(column) or str(column).startswith("Unnamed")]
    return frame.drop(columns=drop_columns)
