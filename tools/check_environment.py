#!/usr/bin/env python3
"""Validate the pinned host, Python, MNN, QAIRT, model, and workspace inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from verify_source_manifest import verify as verify_source_model

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HOST_PYTHON = (3, 10)
EXPECTED_EXPORT_PYTHON = (3, 13)
EXPECTED_MNN_REVISION = "0bff03cbef43c783f44e41484b9f8a0b28bd758d"
EXPECTED_QAIRT_VERSION = "2.48.0"
EXPECTED_EXPORTED_WEIGHT_SHA256 = "b096a22a1d61fff84a77c2277202eeafba3802df941b7eef711f3faf570558db"
EXPECTED_EXPORT_PACKAGES = {
    "datasets": "5.0.0",
    "numpy": "2.3.3",
    "onnx": "1.22.0",
    "onnxruntime": "1.27.0",
    "onnxslim": "0.1.94",
    "peft": "0.19.1",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.1",
    "torch": "2.8.0",
    "transformers": "4.57.6",
}
EXPECTED_HOST_PACKAGES = {"cmake": "4.4.2", "ninja": "1.13.0"}
QNN_FILES = {
    "bin/x86_64-linux-clang/qnn-model-lib-generator": "be751c726210013cc4fee4fbe18e9cd67ab3a7f508bd8cd2bd33772a1e144a00",  # noqa: E501
    "bin/x86_64-linux-clang/qnn-context-binary-generator": "b9400731a5b4a3f4d108f1ab44d501894a1691fc382b2e6c3d33737694d2c82b",  # noqa: E501
    "lib/x86_64-linux-clang/libQnnHtp.so": "c26e65d0eaae6e1fd6d6faa6d4f1726606cebc317b8bb0a690151e19a246f4f7",  # noqa: E501
    "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so": "e944c16d6ea9329411bd01ffd7db0ba8be5b61cca8fa94b73ccfc4742d8b34f0",  # noqa: E501
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def check_python(
    fresh_export: bool,
    export_device: str,
    errors: list[str],
    report: dict[str, object],
) -> None:
    expected_python = EXPECTED_EXPORT_PYTHON if fresh_export else EXPECTED_HOST_PYTHON
    actual_python = sys.version_info[:2]
    report["python"] = platform.python_version()
    report["python_profile"] = "export" if fresh_export else "host"
    if actual_python != expected_python:
        errors.append(
            f"Python {expected_python[0]}.{expected_python[1]} required for "
            f"{'export' if fresh_export else 'host'} profile, got {platform.python_version()}"
        )
    packages: dict[str, str] = {}
    expected_packages = EXPECTED_EXPORT_PACKAGES if fresh_export else EXPECTED_HOST_PACKAGES
    for name, expected in expected_packages.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Python package missing: {name}=={expected}")
            continue
        packages[name] = actual
        accepted_versions = {expected}
        if fresh_export and export_device == "cpu" and name == "torch":
            accepted_versions.add(f"{expected}+cpu")
        if actual not in accepted_versions:
            errors.append(f"Python package mismatch: {name} expected={expected} actual={actual}")
    report["packages"] = packages


def check_export_accelerator(
    export_device: str, errors: list[str], report: dict[str, object]
) -> None:
    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    report["export_accelerator"] = {
        "requested_device": export_device,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
    }
    if export_device == "cuda" and not cuda_available:
        errors.append(
            "fresh OmniQuant export requires CUDA to preserve the validated mixed-precision "
            "execution path; select --export-device cpu only for an export-only validation"
        )


def patch_state(mnn_root: Path) -> tuple[str, list[str]]:
    patch_dir = ROOT / "third_party/patches/mnn"
    states: list[str] = []
    for patch in sorted(patch_dir.glob("*.patch")):
        reverse = subprocess.run(
            ["git", "apply", "--unidiff-zero", "--reverse", "--check", str(patch)],
            cwd=mnn_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        forward = subprocess.run(
            ["git", "apply", "--unidiff-zero", "--check", str(patch)],
            cwd=mnn_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if reverse.returncode == 0:
            states.append("applied")
        elif forward.returncode == 0:
            states.append("applicable")
        else:
            states.append("invalid")
    if states and len(set(states)) == 1:
        return states[0], states
    return "mixed", states


def check_mnn(
    mnn_root: Path,
    build_mnn_tools: str,
    mnn_build_dir: Path | None,
    errors: list[str],
    report: dict[str, object],
) -> None:
    try:
        revision = command_output(["git", "rev-parse", "HEAD"], cwd=mnn_root)
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"MNN is not a readable Git checkout: {mnn_root}: {exc}")
        return
    state, per_patch = patch_state(mnn_root)
    report["mnn"] = {"root": str(mnn_root), "revision": revision, "patch_state": state}
    if revision != EXPECTED_MNN_REVISION:
        errors.append(f"MNN revision mismatch: expected={EXPECTED_MNN_REVISION} actual={revision}")
    expected_states = {"applicable"} if build_mnn_tools == "true" else {"applied"}
    if build_mnn_tools == "resume":
        expected_states = {"applicable", "applied"}
    if state not in expected_states:
        errors.append(
            f"MNN patch state must be one of {sorted(expected_states)} when "
            f"BUILD_MNN_TOOLS={build_mnn_tools}; "
            f"state={state} per_patch={per_patch}"
        )
    if build_mnn_tools in {"false", "resume"} and mnn_build_dir:
        for name in ("MNNConvert", "generateIO", "compilefornpu"):
            path = mnn_build_dir / name
            if not path.is_file() or not os.access(path, os.X_OK):
                errors.append(f"MNN host executable missing: {path}")


def check_qnn(
    qnn_root: Path,
    require_host_build: bool,
    errors: list[str],
    report: dict[str, object],
) -> None:
    notes = qnn_root / "QAIRT_ReleaseNotes.txt"
    if not notes.is_file():
        errors.append(f"QAIRT release notes missing: {notes}")
        return
    match = re.search(
        r"^([0-9]+\.[0-9]+\.[0-9]+)\n=+$", notes.read_text(errors="replace"), re.MULTILINE
    )
    version = match.group(1) if match else "unknown"
    hashes: dict[str, str] = {}
    runnable_tools: dict[str, bool] = {}
    if version != EXPECTED_QAIRT_VERSION:
        errors.append(f"QAIRT version mismatch: expected={EXPECTED_QAIRT_VERSION} actual={version}")
    for relative, expected in QNN_FILES.items():
        path = qnn_root / relative
        if not path.is_file():
            errors.append(f"QAIRT file missing: {path}")
            continue
        actual = sha256(path)
        hashes[relative] = actual
        if actual != expected:
            errors.append(
                f"QAIRT checksum mismatch: {relative} expected={expected} actual={actual}"
            )
        if require_host_build and relative.startswith("bin/"):
            result = subprocess.run(
                [str(path), "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            runnable_tools[relative] = result.returncode == 0
            if result.returncode != 0:
                errors.append(f"QAIRT host tool is not runnable with the current libraries: {path}")
    compiler = shutil.which("clang++-9") or shutil.which("clang++")
    if require_host_build and compiler is None:
        errors.append("QAIRT host model compilation requires clang++-9 or clang++ on PATH")
    compiler_version = None
    if compiler:
        try:
            compiler_version = command_output([compiler, "--version"]).splitlines()[0]
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"QAIRT host compiler is not runnable: {compiler}: {exc}")
    report["qairt"] = {
        "root": str(qnn_root),
        "version": version,
        "sha256": hashes,
        "runnable_tools": runnable_tools,
        "host_compiler": compiler,
        "host_compiler_version": compiler_version,
    }


def check_model(model_dir: Path, errors: list[str], report: dict[str, object]) -> None:
    manifest_path = ROOT / "releases/qwen3-4b-sm8850-v81-c64-rope-cpu/source-manifest.json"
    try:
        verification = verify_source_model(model_dir, manifest_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"model verification failed: {exc}")
        return
    report["model"] = {"root": str(model_dir), **verification}


def check_calibration(path: Path, errors: list[str], report: dict[str, object]) -> None:
    manifest_path = ROOT / "releases/qwen3-4b-sm8850-v81-c64-rope-cpu/calibration-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["file"]
    if not path.is_file():
        errors.append(f"OmniQuant calibration file missing: {path}")
        return
    actual_hash = sha256(path)
    actual_bytes = path.stat().st_size
    if actual_hash != expected["sha256"] or actual_bytes != expected["bytes"]:
        errors.append(
            "OmniQuant calibration mismatch: "
            f"expected_sha256={expected['sha256']} actual_sha256={actual_hash} "
            f"expected_bytes={expected['bytes']} actual_bytes={actual_bytes}"
        )
    report["calibration"] = {
        "path": str(path),
        "sha256": actual_hash,
        "bytes": actual_bytes,
        "samples": manifest["selection"]["non_empty_samples"],
        "dataset_revision": manifest["dataset"]["revision"],
    }


def existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return candidate


def check_storage(
    work_dir: Path,
    cache_root: Path | None,
    exported_model_source: Path | None,
    expected_exported_weight_sha256: str,
    min_free_gib: int,
    errors: list[str],
    report: dict[str, object],
) -> None:
    work_parent = existing_parent(work_dir.parent)
    free_bytes = shutil.disk_usage(work_parent).free
    required = min_free_gib * 1024**3
    if free_bytes < required:
        errors.append(
            f"insufficient free space at {work_parent}: required={min_free_gib}GiB "
            f"actual={free_bytes / 1024**3:.1f}GiB"
        )
    storage: dict[str, object] = {
        "work_parent": str(work_parent),
        "free_gib": round(free_bytes / 1024**3, 1),
        "minimum_gib": min_free_gib,
    }
    if cache_root:
        storage["cache_parent"] = str(existing_parent(cache_root.parent))
    if exported_model_source:
        if not exported_model_source.is_dir():
            errors.append(f"EXPORTED_MODEL_SOURCE is not a directory: {exported_model_source}")
        else:
            same_device = exported_model_source.stat().st_dev == work_parent.stat().st_dev
            storage["export_source_hardlink_compatible"] = same_device
            if not same_device:
                errors.append("EXPORTED_MODEL_SOURCE and WORK_DIR are on different filesystems")
            exported_weight = exported_model_source / "llm.mnn.weight"
            if not exported_weight.is_file():
                errors.append(f"exported MNN weight missing: {exported_weight}")
            else:
                actual_weight_hash = sha256(exported_weight)
                storage["exported_weight_sha256"] = actual_weight_hash
                if actual_weight_hash != expected_exported_weight_sha256:
                    errors.append(
                        "exported MNN weight checksum mismatch: "
                        f"expected={expected_exported_weight_sha256} actual={actual_weight_hash}"
                    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="meetnote-link-check-", dir=work_parent
        ) as directory:
            source = Path(directory) / "source"
            target = Path(directory) / "target"
            source.touch()
            os.link(source, target)
        storage["hardlinks"] = "supported"
    except OSError as exc:
        errors.append(f"WORK_DIR filesystem does not support hard links: {exc}")
    report["storage"] = storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--fresh-export", action="store_true")
    parser.add_argument("--require-qnn-host-build", action="store_true")
    parser.add_argument("--export-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--qnn-sdk-root", type=Path)
    parser.add_argument("--hf-model-dir", type=Path)
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument("--mnn-root", type=Path, default=ROOT / "third_party/MNN")
    parser.add_argument("--mnn-build-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--exported-model-source", type=Path)
    parser.add_argument(
        "--expected-exported-weight-sha256",
        default=EXPECTED_EXPORTED_WEIGHT_SHA256,
        help="Pinned llm.mnn.weight hash accepted for EXPORTED_MODEL_SOURCE",
    )
    parser.add_argument("--build-mnn-tools", choices=("true", "false", "resume"), default="false")
    parser.add_argument("--min-free-gib", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    report: dict[str, object] = {"format": "meetnote.environment_check.v1"}
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_exported_weight_sha256):
        errors.append("--expected-exported-weight-sha256 must be 64 lowercase hex characters")
    check_python(args.fresh_export, args.export_device, errors, report)
    if args.fresh_export:
        check_export_accelerator(args.export_device, errors, report)
    if not args.python_only:
        required = {
            "--qnn-sdk-root": args.qnn_sdk_root,
            "--hf-model-dir": args.hf_model_dir,
            "--calibration-data": args.calibration_data,
            "--work-dir": args.work_dir,
        }
        for option, value in required.items():
            if value is None:
                errors.append(f"{option} is required unless --python-only is used")
        required_present = all(value is not None for value in required.values())
        if required_present:
            check_mnn(
                args.mnn_root.resolve(),
                args.build_mnn_tools,
                args.mnn_build_dir.resolve() if args.mnn_build_dir else None,
                errors,
                report,
            )
            check_qnn(args.qnn_sdk_root.resolve(), args.require_qnn_host_build, errors, report)
            check_model(args.hf_model_dir.resolve(), errors, report)
            check_calibration(args.calibration_data.resolve(), errors, report)
            check_storage(
                args.work_dir,
                args.cache_root,
                args.exported_model_source,
                args.expected_exported_weight_sha256,
                args.min_free_gib,
                errors,
                report,
            )
    report["status"] = "valid" if not errors else "invalid"
    report["errors"] = errors
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
