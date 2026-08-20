#!/usr/bin/env python3
"""Extract a MOSS result and write hashes for one immutable device run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_result(log: Path) -> dict[str, object] | None:
    for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            document = json.loads(line)
        except json.JSONDecodeError:
            continue
        if document.get("format") == "meetnote.moss_qnn_result.v1":
            return document
    return None


def record(
    release: Path,
    input_wav: Path,
    run_log: Path,
    output_dir: Path,
    *,
    exit_code: int,
    device_model: str,
    soc_model: str,
    android_version: str,
) -> dict[str, object]:
    result = extract_result(run_log)
    result_path = output_dir / "result.json"
    if result is not None:
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    evidence = {
        "format": "meetnote.moss_device_run_evidence.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "device": {
            "model": device_model,
            "soc": soc_model,
            "android": android_version,
        },
        "process_exit_code": exit_code,
        "result_present": result is not None,
        "result_status": None if result is None else result.get("status"),
        "files": {},
    }
    files = {
        "artifact_manifest": release / "artifact-manifest.json",
        "runner": release / "moss_qnn_runner",
        "input_wav": input_wav,
        "run_log": run_log,
    }
    if result is not None:
        files["result"] = result_path
    prompt_alignment = output_dir / "prompt-alignment.json"
    if prompt_alignment.is_file():
        files["prompt_alignment"] = prompt_alignment
    for name, path in files.items():
        evidence["files"][name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--input-wav", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--device-model", required=True)
    parser.add_argument("--soc-model", required=True)
    parser.add_argument("--android-version", required=True)
    args = parser.parse_args()
    evidence = record(
        args.release,
        args.input_wav,
        args.run_log,
        args.output_dir,
        exit_code=args.exit_code,
        device_model=args.device_model,
        soc_model=args.soc_model,
        android_version=args.android_version,
    )
    print(json.dumps({"result_present": evidence["result_present"], "result_status": evidence["result_status"]}))


if __name__ == "__main__":
    main()
