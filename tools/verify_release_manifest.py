#!/usr/bin/env python3

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
    parser = argparse.ArgumentParser(
        description="Verify a QNN assembly manifest against its wrapper and contexts."
    )
    parser.add_argument("qnn_dir", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="defaults to <qnn_dir>/assembly-manifest.json",
    )
    parser.add_argument("--allow-extra", action="store_true")
    return parser.parse_args()


def verify(qnn_dir: Path, manifest_path: Path, allow_extra: bool = False) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("format") != "meetnote.qnn_cpu_rope_assembly.v1":
        raise ValueError(f"unsupported manifest format: {manifest.get('format')}")
    contexts = manifest.get("contexts")
    if not isinstance(contexts, list):
        raise ValueError("manifest contexts must be a list")
    if any(not isinstance(item, dict) for item in contexts):
        raise ValueError("manifest contexts must contain objects")
    expected_count = int(manifest["output_graph_count"])
    if [item.get("output") for item in contexts] != list(range(expected_count)):
        raise ValueError("manifest context outputs are not contiguous")

    wrapper = qnn_dir / "llm.mnn"
    actual_wrapper_hash = sha256(wrapper)
    if actual_wrapper_hash != manifest["wrapper_sha256"]:
        raise ValueError(
            "wrapper checksum mismatch: "
            f"expected={manifest['wrapper_sha256']} actual={actual_wrapper_hash}"
        )

    context_bytes = 0
    expected_files = {"llm.mnn"}
    if manifest_path.parent.resolve() == qnn_dir.resolve():
        expected_files.add(manifest_path.name)
    for item in contexts:
        if not isinstance(item.get("output"), int):
            raise ValueError("manifest context output must be an integer")
        context_name = f"graph{item['output']}.bin"
        expected_files.add(context_name)
        context = qnn_dir / context_name
        actual_hash = sha256(context)
        if actual_hash != item["sha256"]:
            raise ValueError(
                f"{context.name} checksum mismatch: expected={item['sha256']} actual={actual_hash}"
            )
        context_bytes += context.stat().st_size

    actual_files = {path.name for path in qnn_dir.iterdir()}
    extras = sorted(actual_files - expected_files)
    if extras and not allow_extra:
        raise ValueError(f"unmanifested QNN payload files: {extras}")
    return {
        "manifest": str(manifest_path),
        "contexts": len(contexts),
        "context_bytes": context_bytes,
        "wrapper_sha256": actual_wrapper_hash,
        "extra_files": extras,
        "status": "valid",
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or args.qnn_dir / "assembly-manifest.json"
    print(
        json.dumps(
            verify(args.qnn_dir, manifest_path, args.allow_extra),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
