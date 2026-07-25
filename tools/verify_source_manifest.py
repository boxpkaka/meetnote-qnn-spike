#!/usr/bin/env python3
"""Verify a local model snapshot against a pinned source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "meetnote.model_source_manifest.v1":
        raise ValueError("unsupported model source manifest format")

    total_bytes = 0
    for relative, expected in manifest["files"].items():
        path = args.model_dir / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"source checksum mismatch for {relative}: "
                f"expected={expected} actual={actual}"
            )
        total_bytes += path.stat().st_size

    print(
        json.dumps(
            {
                "model": manifest["model"],
                "files": len(manifest["files"]),
                "bytes": total_bytes,
                "status": "valid",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
