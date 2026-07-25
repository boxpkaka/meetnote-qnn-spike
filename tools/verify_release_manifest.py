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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or args.qnn_dir / "assembly-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("format") != "meetnote.qnn_cpu_rope_assembly.v1":
        raise ValueError(f"unsupported manifest format: {manifest.get('format')}")
    contexts = manifest.get("contexts")
    if not isinstance(contexts, list):
        raise ValueError("manifest contexts must be a list")
    expected_count = int(manifest["output_graph_count"])
    if [item.get("output") for item in contexts] != list(range(expected_count)):
        raise ValueError("manifest context outputs are not contiguous")

    wrapper = args.qnn_dir / "llm.mnn"
    actual_wrapper_hash = sha256(wrapper)
    if actual_wrapper_hash != manifest["wrapper_sha256"]:
        raise ValueError(
            "wrapper checksum mismatch: "
            f"expected={manifest['wrapper_sha256']} actual={actual_wrapper_hash}"
        )

    context_bytes = 0
    for item in contexts:
        context = args.qnn_dir / f"graph{item['output']}.bin"
        actual_hash = sha256(context)
        if actual_hash != item["sha256"]:
            raise ValueError(
                f"{context.name} checksum mismatch: "
                f"expected={item['sha256']} actual={actual_hash}"
            )
        context_bytes += context.stat().st_size

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "contexts": len(contexts),
                "context_bytes": context_bytes,
                "wrapper_sha256": actual_wrapper_hash,
                "status": "valid",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
