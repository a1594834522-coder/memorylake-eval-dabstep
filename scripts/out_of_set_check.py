from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.candidate_interpretations import route_ids


SYNTHETIC_AMOUNTS = (7.3, 333, 8888, 42.42, 901.25)
SYNTHETIC_MONTHS = ("2026-02", "2026-03", "2026-05", "2027-01", "2027-04")


def parameter_signature(route_id: str, index: int) -> str:
    amount = SYNTHETIC_AMOUNTS[index % len(SYNTHETIC_AMOUNTS)]
    month = SYNTHETIC_MONTHS[index % len(SYNTHETIC_MONTHS)]
    return f"amount={amount}|month={month}"


def plan_out_of_set_matrix(
    routes: Sequence[str] | None = None,
    *,
    samples_per_route: int = 5,
    teacher_samples: int = 10,
    run_model: bool = False,
) -> dict[str, Any]:
    selected_routes = list(routes or route_ids())
    cases = [
        {"route_id": route_id, "case_index": index, "signature": parameter_signature(route_id, index)}
        for route_id in selected_routes
        for index in range(samples_per_route)
    ]
    return {
        "routes": selected_routes,
        "samples_per_route": samples_per_route,
        "teacher_samples": teacher_samples,
        "will_call_model": bool(run_model),
        "cases": cases,
    }


def assert_no_parameter_collisions(cases: Iterable[dict[str, Any]], existing_signatures: set[tuple[str, str]]) -> None:
    for case in cases:
        key = (str(case["route_id"]), str(case["signature"]))
        if key in existing_signatures:
            raise ValueError(f"synthetic parameter case collides with an existing task: {key[0]} {key[1]}")


def load_existing_parameter_signatures(path: Path) -> set[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        rows = raw.get("tasks") or raw.get("cases") or []
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON list or object with tasks/cases: {path}")
    signatures: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        route_id = row.get("route_id")
        signature = row.get("signature")
        if route_id is not None and signature is not None:
            signatures.add((str(route_id), str(signature)))
    return signatures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run out-of-set route generalization checks.")
    parser.add_argument("--route", action="append", dest="routes")
    parser.add_argument("--samples-per-route", type=int, default=5)
    parser.add_argument("--teacher-samples", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--existing-signatures", type=Path)
    parser.add_argument("--run-model", action="store_true", help="Actually call the configured teacher endpoint.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.run_model:
        raise SystemExit("model execution is intentionally gated for the post-benchmark phase")
    plan = plan_out_of_set_matrix(
        args.routes,
        samples_per_route=args.samples_per_route,
        teacher_samples=args.teacher_samples,
        run_model=args.run_model,
    )
    if args.existing_signatures:
        assert_no_parameter_collisions(plan["cases"], load_existing_parameter_signatures(args.existing_signatures))
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
