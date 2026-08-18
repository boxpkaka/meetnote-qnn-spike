#!/usr/bin/env python3
"""Gate an independent CPU-export release candidate without weakening CUDA gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

from verify_release_manifest import verify as verify_qnn_release

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def require_finite(summary: dict, keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        value = summary.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label} has invalid {key}: {value!r}")


def canonical_mnn_sha256(model: Path, mnn_convert: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="meetnote-canonical-mnn-") as directory:
        output = Path(directory) / "model.json"
        subprocess.run(
            [
                str(mnn_convert),
                "-f",
                "MNN",
                "--modelFile",
                str(model),
                "--JsonFile",
                str(output),
                "--bizCode",
                "MNNTest",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        document = json.loads(output.read_text(encoding="utf-8"))
    document.pop("mnn_uuid", None)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_export_runs(
    candidate: dict,
    paths: list[Path],
    export_dirs: list[Path],
    mnn_convert: Path,
) -> dict:
    expected_hash = candidate["export"]["llm_weight_sha256"]
    minimum_runs = int(candidate["policy"]["determinism_min_runs"])
    if len(paths) < minimum_runs:
        raise ValueError(f"at least {minimum_runs} CPU export records are required")
    if len(export_dirs) != len(paths):
        raise ValueError("each CPU export record requires a matching --export-dir")
    hashes = []
    for path in paths:
        record = read_json(path)
        if record.get("format") != "meetnote.export_verification.v1":
            raise ValueError(f"unsupported export record: {path}")
        if record.get("device") != "cpu":
            raise ValueError(f"export record is not CPU: {path}")
        actual = record.get("llm_weight_sha256")
        if actual != expected_hash:
            raise ValueError(
                f"CPU export hash differs: expected={expected_hash} actual={actual} path={path}"
            )
        hashes.append(actual)
    deterministic_files = (
        "config.json",
        "llm_config.json",
        "llm.mnn.weight",
        "tokenizer.mtok",
        "embeddings_bf16.bin",
    )
    payloads = []
    for export_dir in export_dirs:
        payloads.append(
            {
                "path": str(export_dir),
                "files": {name: sha256(export_dir / name) for name in deterministic_files},
                "canonical_mnn_sha256": canonical_mnn_sha256(export_dir / "llm.mnn", mnn_convert),
            }
        )
    for payload in payloads:
        actual_weight_hash = payload["files"]["llm.mnn.weight"]
        if actual_weight_hash != expected_hash:
            raise ValueError(
                "CPU export payload weight differs from candidate: "
                f"expected={expected_hash} actual={actual_weight_hash} "
                f"path={payload['path']}"
            )
    for name in (*deterministic_files, "canonical_mnn_sha256"):
        values = {
            payload[name] if name == "canonical_mnn_sha256" else payload["files"][name]
            for payload in payloads
        }
        if len(values) != 1:
            raise ValueError(f"CPU export payload is not deterministic: {name}")
    return {
        "runs": len(paths),
        "unique_hashes": len(set(hashes)),
        "sha256": expected_hash,
        "records": [{"path": str(path), "sha256": sha256(path)} for path in paths],
        "payloads": payloads,
    }


def token_ids_contract(paths: list[Path]) -> tuple[dict, dict[str, list[int]]]:
    if len(paths) < 2:
        raise ValueError("at least two token-ID records are required")
    records = [read_json(path) for path in paths]
    if any(record.get("format") != "meetnote.token_ids.v1" for record in records):
        raise ValueError("unsupported token-ID record")
    normalized = []
    for record in records:
        prompt_ids = record.get("prompt_ids")
        reference_ids = record.get("reference_ids")
        if not isinstance(prompt_ids, list) or not all(
            isinstance(item, int) for item in prompt_ids
        ):
            raise ValueError("token-ID prompt_ids must be an integer list")
        if not isinstance(reference_ids, list) or not all(
            isinstance(item, int) for item in reference_ids
        ):
            raise ValueError("token-ID reference_ids must be an integer list")
        if not prompt_ids or len(reference_ids) < 2:
            raise ValueError("token-ID record is empty or has no teacher-forcing steps")
        normalized.append({"prompt_ids": prompt_ids, "reference_ids": reference_ids})
    first = normalized[0]
    if any(record != first for record in normalized[1:]):
        raise ValueError("CPU/CUDA/HF token IDs are not identical")
    result = {
        "records": len(records),
        "prompt_tokens": len(first["prompt_ids"]),
        "reference_tokens": len(first["reference_ids"]),
        "exact_match": True,
        "evidence": [{"path": str(path), "sha256": sha256(path)} for path in paths],
    }
    return result, first


def validate_token_ids(paths: list[Path]) -> dict:
    result, _ = token_ids_contract(paths)
    return result


def comparison_summary(
    tool: Path,
    left_prefix: Path,
    right_prefix: Path,
    minimum_steps: int,
    label: str,
) -> dict:
    process = subprocess.run(
        ["node", str(tool), str(left_prefix), str(right_prefix)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    summary = json.loads(process.stdout)
    if not isinstance(summary, dict):
        raise ValueError(f"comparison did not produce an object: {label}")
    if int(summary.get("steps", 0)) < minimum_steps:
        raise ValueError(f"{label} requires at least {minimum_steps} steps")
    require_finite(
        summary,
        (
            "top1_agreement",
            "cpu_target_top1_rate",
            "qnn_target_top1_rate",
            "cosine_mean",
            "cosine_min",
            "rmse_mean",
            "top_k_overlap_mean",
            "cpu_global_min",
            "cpu_global_max",
            "qnn_global_min",
            "qnn_global_max",
        ),
        label,
    )
    return summary


def probe_evidence(
    prefix: Path,
    minimum_steps: int,
    token_contract: dict[str, list[int]],
) -> dict:
    metadata_path = Path(f"{prefix}.jsonl")
    logits_path = Path(f"{prefix}.f32")
    with metadata_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) < 3:
        raise ValueError(f"teacher probe metadata is incomplete: {prefix}")
    header = records[0]
    footer = records[-1]
    if header.get("format") != "meetnote.teacher_logits.v1":
        raise ValueError(f"unsupported teacher probe: {prefix}")
    steps = int(header.get("steps", 0))
    vocab_size = int(header.get("vocab_size", 0))
    if steps < minimum_steps or vocab_size <= 0:
        raise ValueError(f"teacher probe is too short or has no vocabulary: {prefix}")
    prompt_ids = header.get("prompt_ids")
    reference_ids = header.get("reference_ids")
    if prompt_ids != token_contract["prompt_ids"]:
        raise ValueError(f"teacher probe prompt IDs differ from tokenizer evidence: {prefix}")
    if reference_ids != token_contract["reference_ids"]:
        raise ValueError(f"teacher probe reference IDs differ from tokenizer evidence: {prefix}")
    if header.get("prompt_tokens") != len(prompt_ids) or header.get("reference_tokens") != len(
        reference_ids
    ):
        raise ValueError(f"teacher probe token counts differ from embedded IDs: {prefix}")
    step_records = [record for record in records[1:-1] if record.get("type") == "step"]
    if len(step_records) != steps or footer != {"type": "footer", "steps_written": steps}:
        raise ValueError(f"teacher probe step/footer contract differs: {prefix}")
    if steps > len(reference_ids) - 1:
        raise ValueError(f"teacher probe has more steps than reference IDs: {prefix}")
    for index, record in enumerate(step_records):
        expected = (index, reference_ids[index], reference_ids[index + 1])
        actual = (record.get("step"), record.get("input_token"), record.get("target_token"))
        if actual != expected:
            raise ValueError(
                f"teacher probe token contract differs at step {index}: "
                f"expected={expected} actual={actual} prefix={prefix}"
            )
    expected_bytes = steps * vocab_size * 4
    if logits_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"teacher probe logits size differs: expected={expected_bytes} "
            f"actual={logits_path.stat().st_size} prefix={prefix}"
        )
    return {
        "prefix": str(prefix),
        "steps": steps,
        "vocab_size": vocab_size,
        "metadata_sha256": sha256(metadata_path),
        "logits_sha256": sha256(logits_path),
    }


def validate_quality(
    candidate: dict,
    args: argparse.Namespace,
    token_contract: dict[str, list[int]],
) -> dict:
    policy = candidate["policy"]["quality"]
    minimum_steps = int(policy["minimum_teacher_steps"])
    probes = {
        "cpu": probe_evidence(args.cpu_probe_prefix, minimum_steps, token_contract),
        "cuda": probe_evidence(args.cuda_probe_prefix, minimum_steps, token_contract),
        "hf": probe_evidence(args.hf_probe_prefix, minimum_steps, token_contract),
    }
    dimensions = {(item["steps"], item["vocab_size"]) for item in probes.values()}
    if len(dimensions) != 1:
        raise ValueError("CPU/CUDA/HF teacher probe dimensions differ")
    cpu_cuda = comparison_summary(
        args.comparison_tool,
        args.cpu_probe_prefix,
        args.cuda_probe_prefix,
        minimum_steps,
        "CPU vs CUDA",
    )
    cpu_hf = comparison_summary(
        args.comparison_tool,
        args.cpu_probe_prefix,
        args.hf_probe_prefix,
        minimum_steps,
        "CPU vs HF",
    )
    cuda_hf = comparison_summary(
        args.comparison_tool,
        args.cuda_probe_prefix,
        args.hf_probe_prefix,
        minimum_steps,
        "CUDA vs HF",
    )

    if cpu_cuda["top1_agreement"] < float(policy["minimum_cpu_cuda_top1"]):
        raise ValueError("CPU vs CUDA top-1 agreement is below policy")
    top1_floor = cuda_hf["top1_agreement"] - float(policy["maximum_hf_top1_regression"])
    if cpu_hf["top1_agreement"] < top1_floor:
        raise ValueError(
            "CPU export regresses against HF beyond policy: "
            f"cpu={cpu_hf['top1_agreement']} floor={top1_floor}"
        )
    cosine_floor = cuda_hf["cosine_mean"] - float(policy["maximum_hf_cosine_regression"])
    if cpu_hf["cosine_mean"] < cosine_floor:
        raise ValueError(
            "CPU export cosine against HF regresses beyond policy: "
            f"cpu={cpu_hf['cosine_mean']} floor={cosine_floor}"
        )
    return {
        "comparison_tool": {
            "path": str(args.comparison_tool),
            "sha256": sha256(args.comparison_tool),
        },
        "raw_probes": probes,
        "cpu_vs_cuda": {
            "summary": cpu_cuda,
        },
        "cpu_vs_hf": {
            "summary": cpu_hf,
        },
        "cuda_vs_hf": {
            "summary": cuda_hf,
        },
    }


def validate_runtime_gate(path: Path) -> dict:
    document = read_json(path)
    if document.get("format") != "meetnote.qnn_runtime_log_gate.v1":
        raise ValueError("unsupported QNN runtime log gate format")
    if document.get("status") != "valid" or document.get("failures") != []:
        raise ValueError("QNN runtime log gate contains promotion failures")
    counts = document.get("counts")
    required_counts = {
        "shared_weight_mapping_failure",
        "dsp_ssr",
        "process_crash",
        "non_finite_output",
    }
    if not isinstance(counts, dict) or set(counts) != required_counts:
        raise ValueError("QNN runtime log gate has an unexpected count contract")
    if any(value != 0 for value in counts.values()):
        raise ValueError("QNN runtime log gate contains non-zero failure counts")
    files = document.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("QNN runtime log gate has no bound log files")
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("QNN runtime log gate file evidence is invalid")
        log_path = Path(item["path"])
        if not log_path.is_file() or sha256(log_path) != item.get("sha256"):
            raise ValueError(f"QNN runtime log differs from gate evidence: {log_path}")
        if log_path.stat().st_size != item.get("bytes"):
            raise ValueError(f"QNN runtime log size differs from gate evidence: {log_path}")
    return {"path": str(path), "sha256": sha256(path), "record": document}


def validate_device(path: Path, candidate: dict, qnn_dir: Path) -> dict:
    document = read_json(path)
    if document.get("format") != "meetnote.qnn_release_validation.v1":
        raise ValueError("unsupported device validation format")
    rebuild = document.get("clean_rebuild")
    if not isinstance(rebuild, dict):
        raise ValueError("device validation has no clean_rebuild record")
    if rebuild.get("process_crash") or rebuild.get("dsp_ssr"):
        raise ValueError("device validation contains a fatal runtime failure")
    if rebuild.get("non_finite_logits"):
        raise ValueError("device validation contains non-finite logits")
    if rebuild.get("shared_weight_mapping_failure") is not False:
        raise ValueError("device validation contains or omits shared-weight mapping failures")
    if not rebuild.get("all_context_sizes_match"):
        raise ValueError("device contexts were not fully validated")
    expected_weight = candidate["export"]["llm_weight_sha256"]
    expected_manifest = sha256(qnn_dir / "assembly-manifest.json")
    if document.get("release_id") != candidate["release_id"]:
        raise ValueError("device validation targets a different release")
    if document.get("llm_weight_sha256") != expected_weight:
        raise ValueError("device validation targets a different exported weight")
    if document.get("assembly_manifest_sha256") != expected_manifest:
        raise ValueError("device validation targets a different QNN assembly")
    if rebuild.get("soc") != "SM8850" or rebuild.get("dsp_arch") != "v81":
        raise ValueError("device validation is not the pinned SM8850/v81 target")
    return {"path": str(path), "sha256": sha256(path), "record": document}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--export-record", type=Path, action="append", required=True)
    parser.add_argument("--export-dir", type=Path, action="append", required=True)
    parser.add_argument("--mnn-convert", type=Path, required=True)
    parser.add_argument("--token-ids", type=Path, action="append", required=True)
    parser.add_argument(
        "--comparison-tool",
        type=Path,
        default=ROOT / "tools/compare_teacher_forced_logits.mjs",
    )
    parser.add_argument("--cpu-probe-prefix", type=Path, required=True)
    parser.add_argument("--cuda-probe-prefix", type=Path, required=True)
    parser.add_argument("--hf-probe-prefix", type=Path, required=True)
    parser.add_argument("--qnn-dir", type=Path)
    parser.add_argument("--device-validation", type=Path)
    parser.add_argument("--device-runtime-gate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = read_json(args.candidate)
    if candidate.get("format") != "meetnote.cpu_export_candidate.v1":
        raise ValueError("unsupported CPU candidate format")

    tokenizer, token_contract = token_ids_contract(args.token_ids)
    result = {
        "format": "meetnote.cpu_release_gate.v1",
        "candidate": candidate["release_id"],
        "candidate_manifest": {
            "path": str(args.candidate),
            "sha256": sha256(args.candidate),
        },
        "export_determinism": validate_export_runs(
            candidate, args.export_record, args.export_dir, args.mnn_convert
        ),
        "tokenizer": tokenizer,
        "quality": validate_quality(candidate, args, token_contract),
        "qnn": None,
        "device": None,
        "device_runtime": None,
        "stage": "qnn-eligible",
        "production_ready": False,
    }
    if args.qnn_dir:
        result["qnn"] = verify_qnn_release(args.qnn_dir, args.qnn_dir / "assembly-manifest.json")
        result["stage"] = "device-validation-required"
    if args.device_validation:
        if not args.qnn_dir:
            raise ValueError("--device-validation requires --qnn-dir")
        if not args.device_runtime_gate:
            raise ValueError("--device-validation requires --device-runtime-gate")
        result["device_runtime"] = validate_runtime_gate(args.device_runtime_gate)
        result["device"] = validate_device(args.device_validation, candidate, args.qnn_dir)
        result["stage"] = "production-ready"
        result["production_ready"] = True
    elif args.device_runtime_gate:
        raise ValueError("--device-runtime-gate requires --device-validation")

    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
