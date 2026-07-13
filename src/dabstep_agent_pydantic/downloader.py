from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ID = "adyen/DABstep"
EXPECTED_CONTEXT_FILES = [
    "payments.csv",
    "fees.json",
    "manual.md",
    "acquirer_countries.csv",
    "merchant_category_codes.csv",
    "merchant_data.json",
    "payments-readme.md",
]
TASK_SPLITS = {"default": "tasks.json", "dev": "tasks_dev.json"}


def download_dabstep_data(output_dir: Path, overwrite: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context_dir = output_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)

    for name in EXPECTED_CONTEXT_FILES:
        destination = context_dir / name
        if destination.exists() and not overwrite:
            continue
        source = _download_hf_file(f"data/context/{name}")
        shutil.copyfile(source, destination)

    for split, filename in TASK_SPLITS.items():
        destination = output_dir / filename
        if destination.exists() and not overwrite:
            continue
        rows = _load_hf_tasks_rows(split)
        destination.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _download_hf_file(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename))


def _load_hf_tasks_rows(split: str) -> list[dict[str, object]]:
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(REPO_ID, "tasks", split=split)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download DABStep tasks and context files")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Local output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    download_dabstep_data(args.output_dir, overwrite=args.overwrite)
    print(f"DABStep data ready at {args.output_dir}")


if __name__ == "__main__":
    main()
