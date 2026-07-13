"""Shared runtime utilities (formerly in the hand-written deterministic solver)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from dabstep_agent_pydantic.dabstep_core import load_dabstep_data

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def cached_load_dabstep_data(data_dir: Path):
    return _cached_load_dabstep_data_by_path(str(Path(data_dir).expanduser().resolve()))


@lru_cache(maxsize=4)
def _cached_load_dabstep_data_by_path(data_dir: str):
    return load_dabstep_data(Path(data_dir))


def decimal_places(guidelines: str, *, default: int) -> int:
    match = re.search(r"rounded to (?P<places>\d+) decimals?", guidelines, flags=re.IGNORECASE)
    return int(match.group("places")) if match else default
