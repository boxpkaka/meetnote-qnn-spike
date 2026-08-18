#!/usr/bin/env python3
"""Turn QNN/HTP runtime warnings and fatal signals into a promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

PATTERNS = {
    "shared_weight_mapping_failure": re.compile(
        r"(?:map result\s+8003|err(?:or)?\s*[=:]?\s*1002)", re.IGNORECASE
    ),
    "dsp_ssr": re.compile(r"(?:subsystem restart|\bSSR\b|adsprpc.*fatal)", re.IGNORECASE),
    "process_crash": re.compile(
        r"(?:fatal signal|segmentation fault|SIGSEGV|Abort message)", re.IGNORECASE
    ),
    "non_finite_output": re.compile(
        r"(?:non[- ]finite|\bNaN\b|[+-]?\bInf(?:inity)?\b)", re.IGNORECASE
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(paths: list[Path]) -> dict[str, object]:
    counts = {name: 0 for name in PATTERNS}
    samples: dict[str, list[str]] = {name: [] for name in PATTERNS}
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                for name, pattern in PATTERNS.items():
                    if pattern.search(line):
                        counts[name] += 1
                        if len(samples[name]) < 5:
                            samples[name].append(f"{path}:{line_number}:{line.rstrip()}")
    failures = [name for name, count in counts.items() if count]
    return {
        "format": "meetnote.qnn_runtime_log_gate.v1",
        "files": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in paths
        ],
        "counts": counts,
        "samples": {name: lines for name, lines in samples.items() if lines},
        "failures": failures,
        "status": "valid" if not failures else "invalid",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--allow-known-mapping-warning",
        action="store_true",
        help="permit only the known experimental shared-weight mapping warning",
    )
    args = parser.parse_args()
    result = analyze(args.logs)
    failures = list(result["failures"])
    if args.allow_known_mapping_warning:
        failures = [name for name in failures if name != "shared_weight_mapping_failure"]
        result["promotion_failures"] = failures
        result["status"] = "valid" if not failures else "invalid"
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
