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
    parser.add_argument(
        "--allow-extra",
        action="store_true",
        help=(
            "allow payload files not listed in the manifest "
            "(cache directories are always ignored)"
        ),
    )
    return parser.parse_args()


def safe_manifest_path(model_dir: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe source manifest path: {relative}")
    path = model_dir / candidate
    resolved_root = model_dir.resolve()
    if not path.resolve().is_relative_to(resolved_root):
        raise ValueError(f"source manifest path escapes model directory: {relative}")
    return path


def payload_files(model_dir: Path) -> set[str]:
    return {
        path.relative_to(model_dir).as_posix()
        for path in model_dir.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(model_dir).parts
    }


def verify(model_dir: Path, manifest_path: Path, allow_extra: bool = False) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "meetnote.model_source_manifest.v1":
        raise ValueError("unsupported model source manifest format")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source manifest files must be a non-empty object")
    total_bytes = 0
    expected_files: set[str] = set()
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("source manifest paths and hashes must be strings")
        path = safe_manifest_path(model_dir, relative)
        expected_files.add(Path(relative).as_posix())
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"source checksum mismatch for {relative}: expected={expected} actual={actual}"
            )
        total_bytes += path.stat().st_size

    extras = sorted(payload_files(model_dir) - expected_files)
    if extras and not allow_extra:
        raise ValueError(f"unmanifested model payload files: {extras}")
    return {
        "model": manifest["model"],
        "files": len(files),
        "bytes": total_bytes,
        "extra_files": extras,
        "status": "valid",
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            verify(args.model_dir, args.manifest, args.allow_extra),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
