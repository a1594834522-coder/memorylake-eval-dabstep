from __future__ import annotations

from pathlib import Path


def resolve_standard_data_path(path: Path, *, data_root: Path = Path("data")) -> Path:
    """Resolve a bare benchmark path against the downloader's data directory."""
    path = Path(path)
    if path.exists() or path.is_absolute() or path.parts[:1] == data_root.parts[:1]:
        return path
    candidate = data_root / path
    return candidate if candidate.exists() else path
