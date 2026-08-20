#!/usr/bin/env python3
"""Compare Android runner prompt IDs with a host full-prompt contract report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(result: dict[str, object], contract: dict[str, object]) -> dict[str, object]:
    if result.get("format") != "meetnote.moss_qnn_result.v1":
        raise ValueError("unsupported MOSS device result format")
    if contract.get("format") != "meetnote.moss_full_prompt_contract.v1":
        raise ValueError("unsupported MOSS full prompt contract format")
    expected = [int(value) for value in contract.get("runtime_input_ids", [])]
    actual = [int(value) for value in result.get("prompt_token_ids", [])]
    first_mismatch = min(len(expected), len(actual))
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            first_mismatch = index
            break
    complete = result.get("prompt_token_ids_complete") is True
    count_matches = result.get("prompt_tokens") == len(actual) == len(expected)
    exact = complete and count_matches and actual == expected
    return {
        "format": "meetnote.moss_device_prompt_alignment.v1",
        "status": "valid" if exact else "invalid",
        "sample_count": contract.get("sample_count"),
        "expected_token_count": len(expected),
        "actual_token_count": len(actual),
        "prompt_token_ids_complete": complete,
        "first_mismatch": first_mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    report = verify(
        json.loads(args.result.read_text(encoding="utf-8")),
        json.loads(args.contract.read_text(encoding="utf-8")),
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
