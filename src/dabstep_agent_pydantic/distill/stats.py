"""Exact binomial evidence gate for skill adoption.

A hard agreement-rate floor (``rate >= 0.60``) is blind to sample size: it
treats 3/3 and 12/12 as equally adoptable, though the first is weak evidence
and the second is overwhelming. The gate here asks the size-aware question
instead - *is the observed agreement significantly above what a coin-flip
null would produce?* - via the exact one-sided binomial tail, so a template
earns a deterministic skill only when its evidence would be unlikely by
chance.
"""

from __future__ import annotations

from math import comb


def binom_sf_at_least(k: int, n: int, p: float) -> float:
    """Exact P(X >= k) for X ~ Binomial(n, p). No approximation, no deps."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    # Sum the upper tail; summing the smaller side would be nicer numerically
    # but n is tiny here (sampled instances), so the direct sum is exact enough.
    return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def passes_binomial_gate(agree: int, total: int, *, null_rate: float, alpha: float) -> bool:
    """True when agree/total is significant evidence that the true agreement
    rate exceeds ``null_rate`` at level ``alpha`` (one-sided).

    Examples at null_rate=0.5, alpha=0.05:
      3/3 -> p=0.125  (fail: too few trials)
      5/5 -> p=0.031  (pass)
      6/6 -> p=0.016  (pass)
      9/12 -> p=0.073 (fail: not clean enough for the sample)
      10/12 -> p=0.019 (pass)
    """
    if total <= 0:
        return False
    return binom_sf_at_least(agree, total, null_rate) <= alpha
