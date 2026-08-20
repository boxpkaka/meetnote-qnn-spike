#!/usr/bin/env python3
"""Write the deterministic manifest for an assembled MOSS runtime payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from moss_runtime import MOSS_AUDIO_CONTRACT


RELEASE_ID = "moss-transcribe-diarize-0.9b-sm8850-v81-poc-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path, logits_range: float, ndk_revision: str | None) -> dict[str, object]:
    manifest_path = root / "artifact-manifest.json"
    files = {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    manifest = {
        "format": "meetnote.moss_qnn_artifacts.v1",
        "release_id": RELEASE_ID,
        "target": {
            "soc": "SM8850",
            "soc_id": 87,
            "dsp_arch": "v81",
            "vtcm_mb": 8,
            "hvx_threads": 8,
            "qairt_version": "2.48.40.260702",
        },
        "runtime_performance": {
            "profile": "burst",
            "dcvs_enabled": False,
            "sleep_disabled": True,
            "voltage_corner": "max",
        },
        "audio_token_contract": MOSS_AUDIO_CONTRACT,
        "android_ndk_revision": ndk_revision,
        "logits_range": [-logits_range, logits_range],
        "files": files,
        "status": "server-build-complete-device-validation-required",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--logits-range", type=float, required=True)
    parser.add_argument("--android-ndk-revision")
    args = parser.parse_args()
    manifest = write_manifest(args.release, args.logits_range, args.android_ndk_revision)
    print(json.dumps({"file_count": len(manifest["files"]), "status": manifest["status"]}))


if __name__ == "__main__":
    main()
