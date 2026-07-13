from __future__ import annotations

import os
from pathlib import Path

from dabstep_agent_pydantic.distill.matrix_cache import MatrixCache
from dabstep_agent_pydantic.distill.matrix_cache import MatrixCacheKey


def _key(**overrides) -> MatrixCacheKey:
    values = {
        "template_hash": "template",
        "spec_hash": "spec",
        "instances_hash": "instances",
        "data_hash": "data",
        "documents_hash": "documents",
        "artifact_version": "v1",
    }
    values.update(overrides)
    return MatrixCacheKey(**values)


def test_matrix_cache_round_trip_and_fingerprint_invalidation(tmp_path):
    cache = MatrixCache(tmp_path)
    value = {
        "matrix": {"candidate": {"1": "2.0"}},
        "targeting": {"unanimous": True, "sep_tids": None},
    }

    cache.put(_key(), value)

    assert cache.get(_key()) == value
    assert cache.get(_key(data_hash="changed")) is None


def test_matrix_cache_replaces_entries_atomically(tmp_path, monkeypatch):
    cache = MatrixCache(tmp_path)
    key = _key()
    cache.put(key, {"matrix": {"candidate": {"1": "old"}}})
    real_replace = os.replace
    replacements: list[tuple[Path, Path]] = []

    def recording_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.exists()
        replacements.append((source_path, destination_path))
        real_replace(source, destination)

    monkeypatch.setattr(
        "dabstep_agent_pydantic.distill.matrix_cache.os.replace",
        recording_replace,
    )

    cache.put(key, {"matrix": {"candidate": {"1": "new"}}})

    assert replacements
    assert cache.get(key) == {"matrix": {"candidate": {"1": "new"}}}
    assert not list(tmp_path.glob("*.tmp"))


def test_matrix_cache_treats_malformed_payload_as_a_miss(tmp_path):
    cache = MatrixCache(tmp_path)
    key = _key()
    path = tmp_path / f"{key.digest}.json"
    path.write_text("[]", encoding="utf-8")

    assert cache.get(key) is None
