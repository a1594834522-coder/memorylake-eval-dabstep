"""Learn-run scoped memoization for expensive per-(merchant, month) computations.

Discrimination evaluates 4-8 candidate specs over the same instances; the
underlying merchant-month payment slice and monthly context are identical
across candidates and across candidates' repeated evaluations. This memo is
explicitly enabled only for the duration of a learn run (single-threaded);
the solver path never uses it, so no cross-task mutation risk is introduced.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

_ENABLED = False
_STORE: dict[tuple, Any] = {}


@contextmanager
def learn_memo():
    global _ENABLED
    _ENABLED = True
    _STORE.clear()
    try:
        yield
    finally:
        _ENABLED = False
        _STORE.clear()


def memo_get_or_compute(key: tuple, compute):
    if not _ENABLED:
        return compute()
    if key not in _STORE:
        _STORE[key] = compute()
    return _STORE[key]
