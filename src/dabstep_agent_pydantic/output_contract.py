from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputContract:
    raw_guidelines: str
    decimals: int | None = None
    list_shape: str | None = None
    case_sensitive_choices: tuple[str, ...] = ()
    empty_string_allowed: bool = False
    not_applicable_allowed: bool = False
    bare_number: bool = False


def parse_guidelines(text: str | None) -> OutputContract:
    guidelines = (text or "").strip()
    lowered = guidelines.lower()
    decimals = _decimal_places(guidelines)
    list_shape = _list_shape(guidelines)
    return OutputContract(
        raw_guidelines=guidelines,
        decimals=decimals,
        list_shape=list_shape,
        case_sensitive_choices=_case_sensitive_choices(guidelines),
        empty_string_allowed="empty string" in lowered,
        not_applicable_allowed="not applicable" in lowered,
        bare_number=_expects_bare_number(guidelines, decimals=decimals, list_shape=list_shape),
    )


def validate_output_contract(answer: str, contract: OutputContract) -> str | None:
    answer = answer.strip()
    if not answer:
        if contract.empty_string_allowed:
            return None
        return "Return a non-empty agent_answer unless the task explicitly asks for an empty string."

    if contract.not_applicable_allowed and answer == "Not Applicable":
        return None

    if contract.case_sensitive_choices and answer.lower() in {choice.lower() for choice in contract.case_sensitive_choices}:
        if answer not in contract.case_sensitive_choices:
            return (
                "The guideline gives case-sensitive literal choices; return one exactly as written: "
                + ", ".join(contract.case_sensitive_choices)
                + "."
            )

    if contract.decimals is not None and contract.bare_number:
        if not re.fullmatch(rf"-?\d+\.\d{{{contract.decimals}}}", answer):
            return (
                f"The guideline asks for a number rounded to {contract.decimals} decimal places; "
                f"return only that formatted number (e.g. produced by format_decimal_places(value, {contract.decimals}))."
            )

    if contract.list_shape == "bracketed_group_value":
        if not _is_bracketed_group_value_list(answer):
            return "The guideline asks for a comma-separated [group: value] list; format each item as [group: value]."
    elif contract.list_shape == "comma":
        if " and " in answer or ";" in answer:
            return "The guideline asks for a comma-separated list; separate items with commas only."

    return None


def format_analysis_output(
    value,
    *,
    kind: str,
    decimals: int | None = None,
    empty_string_allowed: bool = False,
) -> str:
    if kind == "decimal":
        if decimals is None:
            raise ValueError("decimal output requires decimals")
        return f"{float(value):.{decimals}f}"
    if kind == "integer":
        numeric = float(value)
        if not numeric.is_integer():
            raise ValueError(f"integer output received non-integral value: {value}")
        return str(int(numeric))
    if kind == "single_string":
        if isinstance(value, (list, tuple, set)):
            items = sorted(str(item) for item in value)
            return items[0] if items else ""
        return str(value)
    if kind == "comma_list":
        if not value:
            return "" if empty_string_allowed else ""
        if all(isinstance(item, dict) and isinstance(item.get("group"), dict) for item in value):
            return ", ".join(_format_group_label(item["group"]) for item in value)
        return ", ".join(str(item) for item in value)
    if kind == "group_value_list":
        if decimals is None:
            raise ValueError("group_value_list output requires decimals")
        if not value:
            return "" if empty_string_allowed else "[]"
        entries = []
        for item in value:
            group = item["group"]
            label = _format_group_label(group)
            entries.append(f"{label}: {float(item['value']):.{decimals}f}")
        return "[" + ", ".join(entries) + "]"
    raise ValueError(f"unsupported analysis output kind: {kind}")


def _format_group_label(group: dict) -> str:
    if len(group) == 1:
        return str(next(iter(group.values())))
    return "; ".join(f"{name}={group[name]}" for name in sorted(group))


def _decimal_places(guidelines: str) -> int | None:
    match = re.search(r"rounded to (?P<places>\d+) decimals?", guidelines, flags=re.IGNORECASE)
    return int(match.group("places")) if match else None


def _list_shape(guidelines: str) -> str | None:
    lowered = guidelines.lower()
    if re.search(r"\[[^\]]+\s*:\s*[^\]]+\]", guidelines) or "[group: value]" in lowered:
        return "bracketed_group_value"
    if "comma" in lowered and "list" in lowered:
        return "comma"
    return None


def _case_sensitive_choices(guidelines: str) -> tuple[str, ...]:
    match = re.search(
        r"\b(?:must be\s+exactly|exactly|must be)\s+(?P<left>yes|no)\s+or\s+(?P<right>yes|no)\b",
        guidelines,
        flags=re.IGNORECASE,
    )
    if not match:
        return ()
    return (match.group("left"), match.group("right"))


def _expects_bare_number(guidelines: str, *, decimals: int | None, list_shape: str | None) -> bool:
    if list_shape is not None:
        return False
    if "{" in guidelines or ":" in guidelines:
        return False
    lowered = guidelines.lower()
    if "list" in lowered:
        return False
    return decimals is not None or "just a number" in lowered or "with a number" in lowered


def _is_bracketed_group_value_list(answer: str) -> bool:
    item = r"\[[^\[\]:]+:\s*[^\[\]]+\]"
    if re.fullmatch(rf"{item}(?:\s*,\s*{item})*", answer):
        return True
    if not (answer.startswith("[") and answer.endswith("]")):
        return False
    entries = [entry.strip() for entry in answer[1:-1].split(",")]
    return bool(entries) and all(re.fullmatch(r"[^:]+:\s*.+", entry) for entry in entries)
