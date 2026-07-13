"""Exact binomial gate: values and adoption semantics."""

from __future__ import annotations

import pytest

from dabstep_agent_pydantic.distill.stats import binom_sf_at_least, passes_binomial_gate


@pytest.mark.parametrize("k,n,expected", [
    (3, 3, 0.125),
    (4, 4, 0.0625),
    (5, 5, 0.03125),
    (6, 6, 0.015625),
    (9, 12, 0.072998),
    (10, 12, 0.019287),
])
def test_binom_sf_matches_known_tail(k, n, expected):
    assert binom_sf_at_least(k, n, 0.5) == pytest.approx(expected, abs=1e-5)


def test_binom_sf_edges():
    assert binom_sf_at_least(0, 5, 0.5) == 1.0   # X >= 0 is certain
    assert binom_sf_at_least(6, 5, 0.5) == 0.0   # more successes than trials


def test_gate_is_sample_size_aware():
    # Same perfect rate, different evidence: 3/3 fails, 5/5 passes at alpha=.05.
    assert not passes_binomial_gate(3, 3, null_rate=0.5, alpha=0.05)
    assert passes_binomial_gate(5, 5, null_rate=0.5, alpha=0.05)


def test_gate_rejects_diluted_large_sample():
    # 9/12 (rate .75) is not significant; 10/12 (rate .83) is.
    assert not passes_binomial_gate(9, 12, null_rate=0.5, alpha=0.05)
    assert passes_binomial_gate(10, 12, null_rate=0.5, alpha=0.05)


def test_gate_zero_trials_fails_closed():
    assert not passes_binomial_gate(0, 0, null_rate=0.5, alpha=0.05)
