#!/usr/bin/env python3
"""Run checks that do not require model weights or the proprietary QNN SDK."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 1_000_000
FORBIDDEN_ROOTS = {"ablation", "local-assets", "sdk", "server-teacher", "work"}
FORBIDDEN_SUFFIXES = {
    ".a",
    ".bin",
    ".f32",
    ".log",
    ".mnn",
    ".o",
    ".so",
    ".strace",
    ".weight",
}
EXPECTED_MNN_REVISION = "0bff03cbef43c783f44e41484b9f8a0b28bd758d"


def run(command: list[str], cwd: Path = ROOT) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
    )
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tracked_files(files: list[Path]) -> None:
    errors: list[str] = []
    secret_patterns = [
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile("BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY"),
        re.compile(r"ghp_[A-Za-z0-9]{36}"),
    ]
    for path in files:
        relative = path.relative_to(ROOT)
        if relative.parts[0] in FORBIDDEN_ROOTS:
            errors.append(f"generated root is tracked: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"generated binary is tracked: {relative}")
        if path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds {MAX_TRACKED_BYTES} bytes: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in secret_patterns:
            if pattern.search(text):
                errors.append(f"credential-like content in {relative}")
                break
    if errors:
        raise ValueError("\n".join(errors))


def validate_fixtures() -> None:
    checksums_path = ROOT / "fixtures/checksums.json"
    document = json.loads(checksums_path.read_text(encoding="utf-8"))
    if document.get("format") != "meetnote.fixture_checksums.v1":
        raise ValueError("unsupported fixture checksum format")
    for relative, expected in document["files"].items():
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"fixture checksum mismatch for {relative}: expected={expected} actual={actual}"
            )


def validate_release_metadata() -> None:
    release = ROOT / "releases/qwen3-4b-sm8850-v81-c64-rope-cpu"
    manifest = json.loads((release / "assembly-manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((release / "provenance.json").read_text(encoding="utf-8"))

    if manifest.get("format") != "meetnote.qnn_cpu_rope_assembly.v1":
        raise ValueError("unsupported assembly manifest format")
    if provenance.get("format") != "meetnote.qnn_release_provenance.v1":
        raise ValueError("unsupported provenance format")

    count = int(manifest["output_graph_count"])
    contexts = manifest["contexts"]
    outputs = [int(item["output"]) for item in contexts]
    if outputs != list(range(count)):
        raise ValueError("release context outputs are not contiguous")
    if any(not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in contexts):
        raise ValueError("release context checksum is not SHA-256")
    if provenance["assembly"]["context_count"] != count:
        raise ValueError("provenance context count differs from the manifest")
    if provenance["assembly"]["wrapper_sha256"] != manifest["wrapper_sha256"]:
        raise ValueError("provenance wrapper checksum differs from the manifest")
    if provenance["graph"].get("vtcm_mb") != 4:
        raise ValueError("release VTCM budget must be pinned to 4 MiB")
    exported_weight = provenance.get("export", {}).get("llm_weight_sha256")
    if not isinstance(exported_weight, str) or not re.fullmatch(r"[0-9a-f]{64}", exported_weight):
        raise ValueError("release exported MNN weight checksum is not pinned")

    validation = json.loads((release / "validation.json").read_text(encoding="utf-8"))
    if validation.get("format") != "meetnote.qnn_release_validation.v1":
        raise ValueError("unsupported release validation format")
    tokenizer = validation["tokenizer_parity"]
    if not tokenizer["exact_token_id_match"]:
        raise ValueError("release tokenizer parity is not exact")
    if tokenizer["model_revision"] != provenance["model"]["revision"]:
        raise ValueError("tokenizer validation used a different model revision")
    rebuild = validation["clean_rebuild"]
    if rebuild["context_count"] != count:
        raise ValueError("clean rebuild context count differs from the manifest")
    if not rebuild["all_context_sizes_match"]:
        raise ValueError("clean rebuild context sizes were not validated")
    if not rebuild["realistic_logits_match_retained_release"]:
        raise ValueError("clean rebuild device logits differ from the retained release")
    if rebuild["process_crash"] or rebuild["dsp_ssr"] or rebuild["non_finite_logits"]:
        raise ValueError("clean rebuild device gate contains a fatal runtime failure")

    source = json.loads((release / "source-manifest.json").read_text(encoding="utf-8"))
    if source.get("format") != "meetnote.model_source_manifest.v1":
        raise ValueError("unsupported model source manifest format")
    if source["model"] != provenance["model"]:
        raise ValueError("source manifest model differs from provenance")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in source["files"].values()):
        raise ValueError("model source checksum is not SHA-256")

    calibration = json.loads((release / "calibration-manifest.json").read_text(encoding="utf-8"))
    if calibration.get("format") != "meetnote.omni_calibration_manifest.v1":
        raise ValueError("unsupported calibration manifest format")
    if calibration["selection"].get("non_empty_samples") != 128:
        raise ValueError("OmniQuant calibration sample count must be 128")
    if not re.fullmatch(r"[0-9a-f]{64}", calibration["file"].get("sha256", "")):
        raise ValueError("OmniQuant calibration checksum is not SHA-256")


def validate_cpu_candidate_metadata() -> None:
    release = ROOT / "releases/qwen3-4b-sm8850-v81-c64-rope-cpu-export-v1"
    candidate = json.loads((release / "candidate.json").read_text(encoding="utf-8"))
    if candidate.get("format") != "meetnote.cpu_export_candidate.v1":
        raise ValueError("unsupported CPU candidate format")
    if candidate.get("release_id") != release.name:
        raise ValueError("CPU candidate release_id differs from its directory")
    if candidate.get("export", {}).get("device") != "cpu":
        raise ValueError("CPU candidate does not pin the CPU export path")
    weight = candidate["export"].get("llm_weight_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", weight):
        raise ValueError("CPU candidate exported MNN weight checksum is not pinned")
    policy = candidate.get("policy", {})
    if int(policy.get("determinism_min_runs", 0)) < 2:
        raise ValueError("CPU candidate requires fewer than two deterministic exports")
    quality = policy.get("quality", {})
    expected_quality = {
        "minimum_teacher_steps": 255,
        "minimum_cpu_cuda_top1": 0.9,
        "maximum_hf_top1_regression": 0.02,
        "maximum_hf_cosine_regression": 0.03,
    }
    if quality != expected_quality:
        raise ValueError("CPU candidate quality policy differs from the reviewed thresholds")
    if not policy.get("require_qnn_manifest"):
        raise ValueError("CPU candidate does not require a QNN manifest")
    if not policy.get("require_sm8850_device_validation"):
        raise ValueError("CPU candidate does not require SM8850 device validation")

    validation = json.loads((release / "server-validation.json").read_text(encoding="utf-8"))
    if validation.get("format") != "meetnote.cpu_candidate_server_validation.v1":
        raise ValueError("unsupported CPU candidate server validation format")
    if validation.get("release_id") != candidate["release_id"]:
        raise ValueError("CPU candidate server validation targets another release")
    if validation["export_determinism"]["llm_weight_sha256"] != weight:
        raise ValueError("CPU candidate server validation used another exported weight")
    if validation.get("stage") != "device-validation-required":
        raise ValueError("CPU candidate server validation has an unexpected stage")
    if validation.get("production_ready") is not False:
        raise ValueError("server-only CPU candidate must not be production-ready")
    assembly = validation.get("qnn_assembly", {})
    if assembly.get("contexts") != 39 or assembly.get("manifest_status") != "valid":
        raise ValueError("CPU candidate QNN assembly was not server-validated")
    for key in ("wrapper_sha256", "assembly_manifest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", assembly.get(key, "")):
            raise ValueError(f"CPU candidate {key} is not SHA-256")


def validate_source_syntax(files: list[Path]) -> None:
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        elif path.suffix == ".sh":
            run(["bash", "-n", str(path)])
        elif path.suffix in {".js", ".mjs"}:
            if shutil.which("node") is None:
                raise RuntimeError("node is required to check JavaScript syntax")
            run(["node", "--check", str(path)])


def validate_mnn_patches(mnn_root: Path, expected_state: str) -> None:
    revision = run(["git", "rev-parse", "HEAD"], cwd=mnn_root).strip()
    if revision != EXPECTED_MNN_REVISION:
        raise ValueError(
            f"MNN revision differs: expected={EXPECTED_MNN_REVISION} actual={revision}"
        )
    patches = sorted((ROOT / "third_party/patches/mnn").glob("*.patch"))
    if not patches:
        raise ValueError("no MNN patches found")
    command = ["git", "apply", "--unidiff-zero"]
    if expected_state == "applied":
        command.append("--reverse")
    command.extend(["--check", *(str(path) for path in patches)])
    if expected_state == "either":
        forward = subprocess.run(
            ["git", "apply", "--unidiff-zero", "--check", *(str(path) for path in patches)],
            cwd=mnn_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        reverse = subprocess.run(
            [
                "git",
                "apply",
                "--unidiff-zero",
                "--reverse",
                "--check",
                *(str(path) for path in patches),
            ],
            cwd=mnn_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if forward.returncode and reverse.returncode:
            raise ValueError("MNN patches are neither cleanly applicable nor fully applied")
    else:
        run(command, cwd=mnn_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnn-root",
        type=Path,
        help="also verify patches against the pinned clean MNN checkout",
    )
    parser.add_argument(
        "--mnn-patch-state",
        choices=("applicable", "applied", "either"),
        default="applicable",
        help="expected state when --mnn-root is supplied (default: applicable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = tracked_files()
    validate_tracked_files(files)
    validate_fixtures()
    validate_release_metadata()
    validate_cpu_candidate_metadata()
    validate_source_syntax(files)
    empty_tree = run(["git", "hash-object", "-t", "tree", "/dev/null"]).strip()
    run(["git", "diff", "--check", empty_tree, "HEAD", "--"])
    if args.mnn_root:
        validate_mnn_patches(args.mnn_root.resolve(), args.mnn_patch_state)
    print(
        json.dumps(
            {
                "tracked_files": len(files),
                "tracked_bytes": sum(path.stat().st_size for path in files),
                "mnn_patches": "valid" if args.mnn_root else "not-requested",
                "status": "valid",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
