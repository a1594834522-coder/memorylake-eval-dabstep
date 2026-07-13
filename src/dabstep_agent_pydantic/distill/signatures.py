"""Template -> trigger regex + parameter signature (mechanical, zero-model).

A normalized template (placeholders like <N>, <MONTH>, <MERCHANT>) is compiled
into a full-anchored regex with named capture groups plus per-parameter
parsers. Placeholder meaning is disambiguated by the literal text immediately
preceding it (e.g. "account_type = <LETTER>" binds to the ``account_type``
parameter). Templates containing a placeholder whose context cannot be
resolved are not signature-compilable — the learn pipeline then leaves that
template to the LLM path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from dabstep_agent_pydantic.dabstep_core import DABStepData, mcc_code_for_description
from dabstep_agent_pydantic.runtime_util import MONTHS


class SignatureError(Exception):
    """Template cannot be compiled into an executable signature."""


PLACEHOLDER = re.compile(r"<(N|MONTH|YEAR|MERCHANT|SCHEME|LETTER|MCC_DESC|ORDINAL|Z_SCORE_THRESHOLD)>")

_GROUP_PATTERNS = {
    "N": r"(\d+(?:\.\d+)?)",
    "MONTH": r"([A-Za-z]+)",
    "YEAR": r"(\d{4})",
    "MERCHANT": r"([A-Z][A-Za-z0-9']*(?:_[A-Za-z0-9']+)+)",
    "SCHEME": r"(\w+)",
    "LETTER": r"([A-Za-z])",
    "MCC_DESC": r"(.+?)",
    "ORDINAL": r"(\d+)(?:st|nd|rd|th)?",
    "Z_SCORE_THRESHOLD": r"(\d+(?:\.\d+)?)",
}

# (placeholder kind, regex over the preceding literal text) -> canonical param.
# First match wins; order encodes specificity.
_CONTEXT_BINDINGS: list[tuple[str, str, str]] = [
    ("LETTER", r"account[_ ]type\s*(?:=|:)?\s*$", "account_type"),
    ("LETTER", r"\baci\s*(?:=|:)?\s*$", "aci"),
    ("N", r"ID\s*=?\s*$", "fee_id"),
    ("N", r"Fee with ID\s*$", "fee_id"),
    ("N", r"changed to\s*$", "new_value"),
    ("N", r"MCC code to\s*$", "new_mcc"),
    ("N", r"transaction value of\s*$", "amount"),
    ("N", r"transaction of\s*$", "amount"),
    ("ORDINAL", r"above the\s*$", "percentile"),
    ("ORDINAL", r"For the\s*$", "day_of_year"),
    ("N", r"EUR and\s*$", "decimals"),
    ("MONTH", r".*", "month"),
    ("YEAR", r".*", "year"),
    ("MERCHANT", r".*", "merchant"),
    ("SCHEME", r".*", "card_scheme"),
    ("MCC_DESC", r".*", "merchant_category_code"),
]

# Literal words in the template that bind constant params (not placeholders).
_LITERAL_BINDINGS: list[tuple[str, str, Any]] = [
    (r"\bFor credit transactions\b", "is_credit", True),
    (r"\bFor debit transactions\b", "is_credit", False),
    (r"\bcredit transaction\b", "is_credit", True),
    (r"\bdebit transaction\b", "is_credit", False),
    (r"\bmost expensive\b", "objective", "max"),
    (r"\bleast expensive\b", "objective", "min"),
]

_PARSERS: dict[str, Callable[[str, DABStepData], Any]] = {
    "amount": lambda s, d: float(s),
    "new_value": lambda s, d: float(s),
    "fee_id": lambda s, d: int(s),
    "new_mcc": lambda s, d: int(s),
    "year": lambda s, d: int(s),
    "percentile": lambda s, d: float(s),
    "day_of_year": lambda s, d: int(s),
    "decimals": lambda s, d: int(s),
    "month": lambda s, d: MONTHS[s.lower()],
    "account_type": lambda s, d: s.upper(),
    "aci": lambda s, d: s.upper(),
    "merchant": lambda s, d: s,
    "card_scheme": lambda s, d: s,
    "merchant_category_code": lambda s, d: mcc_code_for_description(d.merchant_category_codes, s),
}


def parse_raw_param(param: str, raw: str, data: DABStepData) -> Any:
    """Parse one raw parameter value with the same typed parser the regex
    path uses, so structured (non-textual) skill invocation cannot drift
    from template parsing. Raises ValueError on an unknown parameter.

    Structured callers (the solver model) may supply already-normalized
    values that never occur in question text — a month number instead of a
    month name, an MCC code instead of its description. Accept those
    directly; everything else goes through the textual parser."""
    parser = _PARSERS.get(param)
    if parser is None:
        raise ValueError(f"no parser for parameter {param!r}")
    raw = raw.strip()
    if raw.isdigit():
        if param == "month" and 1 <= int(raw) <= 12:
            return int(raw)
        if param == "merchant_category_code":
            return int(raw)
    return parser(raw, data)


@dataclass(frozen=True)
class TemplateSignature:
    template: str
    regex: re.Pattern
    group_params: tuple[str, ...]  # param name per capture group, in order
    constant_params: tuple[tuple[str, Any], ...]

    def parse(self, question: str, data: DABStepData) -> dict[str, Any] | None:
        text = " ".join(question.split())
        match = self.regex.search(text)
        if not match:
            return None
        params: dict[str, Any] = dict(self.constant_params)
        for index, param in enumerate(self.group_params, start=1):
            raw = match.group(index)
            if raw is None:
                continue
            value = _PARSERS[param](raw.strip(), data)
            if param in params and params[param] != value:
                # Same parameter captured twice with different values: not our template.
                return None
            params[param] = value
        return params


def compile_signature(template: str) -> TemplateSignature:
    pattern_parts: list[str] = []
    group_params: list[str] = []
    seen: set[str] = set()
    cursor = 0
    for match in PLACEHOLDER.finditer(template):
        literal = template[cursor:match.start()]
        pattern_parts.append(re.escape(literal))
        kind = match.group(1)
        param = _bind_placeholder(kind, template[:match.start()])
        seen.add(param)
        group_params.append(param)
        pattern_parts.append(_GROUP_PATTERNS[kind])
        cursor = match.end()
    pattern_parts.append(re.escape(template[cursor:]))

    constants = tuple(
        (param, value)
        for literal_pattern, param, value in _LITERAL_BINDINGS
        if re.search(literal_pattern, template, flags=re.IGNORECASE)
    )
    regex = re.compile("".join(pattern_parts), flags=re.IGNORECASE)
    return TemplateSignature(
        template=template,
        regex=regex,
        group_params=tuple(group_params),
        constant_params=constants,
    )


def _bind_placeholder(kind: str, preceding: str) -> str:
    window = preceding[-40:]
    for bind_kind, context_pattern, param in _CONTEXT_BINDINGS:
        if bind_kind != kind:
            continue
        if re.search(context_pattern, window, flags=re.IGNORECASE):
            return param
    raise SignatureError(f"cannot bind <{kind}> after ...{window!r}")
