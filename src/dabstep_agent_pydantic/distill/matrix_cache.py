from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from typing import Iterable

from pydantic import BaseModel
from pydantic import ConfigDict


MATRIX_CACHE_VERSION = "matrix-cache-v1"


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_files(root: Path, paths: Iterable[Path]) -> str:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(paths)
    ]
    return stable_hash(entries)


class MatrixCacheKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    template_hash: str
    spec_hash: str
    instances_hash: str
    data_hash: str
    documents_hash: str
    artifact_version: str = MATRIX_CACHE_VERSION

    @classmethod
    def from_inputs(
        cls,
        *,
        template: str,
        candidates: list[Any],
        instances: list[dict[str, Any]],
        data_hash: str,
        document_fingerprints: dict[str, str],
    ) -> "MatrixCacheKey":
        specs = sorted(
            [
                candidate.model_dump(mode="json")
                if hasattr(candidate, "model_dump")
                else candidate
                for candidate in candidates
            ],
            key=lambda value: json.dumps(value, sort_keys=True),
        )
        instance_inputs = sorted(
            (
                {
                    "task_id": str(instance.get("task_id", "")),
                    "question": str(instance.get("question", "")),
                    "guidelines": str(instance.get("guidelines") or ""),
                }
                for instance in instances
            ),
            key=lambda value: (value["task_id"], value["question"]),
        )
        return cls(
            template_hash=stable_hash(template),
            spec_hash=stable_hash(specs),
            instances_hash=stable_hash(instance_inputs),
            data_hash=data_hash,
            documents_hash=stable_hash(document_fingerprints),
        )

    @property
    def digest(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


class MatrixCache:
    def __init__(self, root: Path):
        self.root = Path(root)

    def get(self, key: MatrixCacheKey) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            if payload.get("cache_key") != key.model_dump(mode="json"):
                return None
            value = payload.get("value")
            return value if isinstance(value, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return None

    def put(self, key: MatrixCacheKey, value: dict[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._path(key)
        payload = {
            "cache_key": key.model_dump(mode="json"),
            "value": value,
        }
        fd, temporary_name = tempfile.mkstemp(
            dir=self.root,
            prefix=f".{key.digest}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def _path(self, key: MatrixCacheKey) -> Path:
        return self.root / f"{key.digest}.json"
