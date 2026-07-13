from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

from dabstep_agent_pydantic.asset_compiler import RouteCard
from dabstep_agent_pydantic.asset_compiler import format_route_cards
from dabstep_agent_pydantic.asset_compiler import load_route_cards


class RuntimeAssets(BaseModel):
    patterns: str = ""
    route_card_count: int = 0
    route_cards: list[RouteCard] = Field(default_factory=list)
    asset_fingerprint: str | None = None


def load_runtime_assets(path: Path | None, max_pattern_chars: int = 8000) -> RuntimeAssets:
    if path is None or not path.exists():
        return RuntimeAssets()

    asset_fingerprint = compute_asset_fingerprint(path)
    route_cards_path = path / "route_cards.json"
    if not route_cards_path.exists():
        return RuntimeAssets(asset_fingerprint=asset_fingerprint)

    route_cards = load_route_cards(route_cards_path)
    return RuntimeAssets(
        patterns=format_route_cards(route_cards, max_chars=max_pattern_chars),
        route_card_count=len(route_cards),
        route_cards=route_cards,
        asset_fingerprint=asset_fingerprint,
    )


def compute_asset_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    digests: list[str] = []
    for asset_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        digests.append(f"{asset_path.relative_to(path).as_posix()}:{digest}")
    if not digests:
        return None
    return "sha256:" + hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()
