#!/usr/bin/env python3
"""Compare complete Hugging Face and MNN tokenizer outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, list[int]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "meetnote.token_ids.v1":
        raise ValueError(f"unsupported token ID format: {path}")
    return {
        "prompt_ids": [int(value) for value in document["prompt_ids"]],
        "reference_ids": [int(value) for value in document["reference_ids"]],
    }


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--mnn", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf = load(args.hf)
    mnn = load(args.mnn)
    for field in ("prompt_ids", "reference_ids"):
        difference = first_difference(hf[field], mnn[field])
        if difference is not None:
            hf_value = hf[field][difference] if difference < len(hf[field]) else None
            mnn_value = mnn[field][difference] if difference < len(mnn[field]) else None
            raise ValueError(
                f"{field} differs at index {difference}: HF={hf_value} MNN={mnn_value}; "
                f"lengths HF={len(hf[field])} MNN={len(mnn[field])}"
            )
    print(
        json.dumps(
            {
                "prompt_tokens": len(hf["prompt_ids"]),
                "reference_tokens": len(hf["reference_ids"]),
                "status": "identical",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
