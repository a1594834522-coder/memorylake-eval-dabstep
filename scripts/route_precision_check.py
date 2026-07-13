from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.distill_demo import match_route


def validate_precision_fixture(path: Path) -> dict[str, Any]:
    variants = _load_variant_rows(path)
    violations = []
    for item in variants:
        route_id = str(item["route_id"])
        variant = str(item["variant"])
        matched = match_route(variant)
        if matched == route_id:
            violations.append({"route_id": route_id, "matched_route": matched, "variant": variant})
    return {"checked": len(variants), "violations": violations}


def _load_variant_rows(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict) and isinstance(raw.get("variants"), list):
        rows = raw["variants"]
    elif isinstance(raw, dict) and isinstance(raw.get("routes"), dict):
        rows = []
        for route_id, variants in raw["routes"].items():
            if not isinstance(variants, list):
                continue
            for variant in variants:
                if isinstance(variant, dict):
                    rows.append({"route_id": str(route_id), "variant": str(variant.get("variant", ""))})
                else:
                    rows.append({"route_id": str(route_id), "variant": str(variant)})
    else:
        raise ValueError(f"expected a JSON list, variants list, or routes mapping: {path}")
    normalized = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        normalized.append({"route_id": str(item["route_id"]), "variant": str(item["variant"])})
    return normalized


def build_prompt() -> str:
    return (
        "Generate near-but-different DABStep question variants for each route. "
        "Each variant should be semantically different from the route it resembles."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or prepare route precision adversarial variants.")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-model", action="store_true", help="Generate variants with a configured model endpoint.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.run_model:
        raise SystemExit("model execution is intentionally gated until variants are generated and reviewed")
    report = validate_precision_fixture(args.fixture) if args.fixture else {"prompt": build_prompt()}
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
