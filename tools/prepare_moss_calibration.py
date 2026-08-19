#!/usr/bin/env python3
"""Extract 128 deterministic C64 audio-conditioned MOSS decoder windows.

Outputs are external build inputs and may contain model embeddings; they must
not be committed.  The input manifest contains ``cases`` with ``id``, ``wav``,
``transcript``, ``split`` and ``sha256`` fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

MODEL_REVISION = "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"
WINDOW_SIZE = 64
WINDOW_COUNT = 128
PHASES = ("audio_start", "audio_middle", "audio_end", "decode")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def all_hashes(value: Any) -> set[str]:
    hashes: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().endswith("sha256") and isinstance(item, str):
                hashes.add(item)
            hashes.update(all_hashes(item))
    elif isinstance(value, list):
        for item in value:
            hashes.update(all_hashes(item))
    return hashes


def candidate_starts(prompt_tokens: int, total_tokens: int) -> dict[str, list[int]]:
    """Create phase-stratified C64 starts without crossing sequence bounds."""
    if prompt_tokens < WINDOW_SIZE or total_tokens < WINDOW_SIZE:
        return {phase: [] for phase in PHASES}
    last_prompt = max(0, prompt_tokens - WINDOW_SIZE)
    candidates = {
        "audio_start": list(range(0, min(last_prompt, WINDOW_SIZE * 4) + 1, WINDOW_SIZE)),
        "audio_middle": [],
        "audio_end": [],
        "decode": list(range(prompt_tokens, total_tokens - WINDOW_SIZE + 1, WINDOW_SIZE)),
    }
    middle = max(0, prompt_tokens // 2 - WINDOW_SIZE // 2)
    candidates["audio_middle"] = [min(middle, last_prompt)]
    candidates["audio_end"] = [last_prompt]
    return candidates


def select_balanced(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select exactly 32 lexically stable, round-robin windows per phase."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["phase"]].append(candidate)
    selected = []
    quota = WINDOW_COUNT // len(PHASES)
    for phase in PHASES:
        rows = sorted(grouped[phase], key=lambda row: (row["case_id"], row["start"]))
        if len(rows) < quota:
            raise ValueError(f"calibration phase {phase} has {len(rows)} windows; need {quota}")
        # Evenly cover the complete sorted set instead of taking only early meetings.
        indexes = [(index * len(rows)) // quota for index in range(quota)]
        selected.extend(rows[index] for index in indexes)
    return selected


def load_audio(path: Path):
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16_000 or audio.shape[1] != 1:
        raise ValueError(f"calibration WAV must be 16 kHz mono: {path}")
    return audio[:, 0]


def extract_case(model, processor, case: dict[str, Any], device):
    import torch
    from moss_runtime import DEFAULT_PROMPT

    wav = Path(case["wav"])
    if sha256(wav) != case["sha256"]:
        raise ValueError(f"calibration WAV checksum mismatch: {wav}")
    audio = load_audio(wav)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(wav)},
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, audio=[audio], max_length=8192, return_tensors="pt").to(device)
    with torch.inference_mode():
        token_embeds = model.get_input_embeddings()(inputs["input_ids"])
        fused = model.model.inject_audio_features(
            inputs["input_ids"],
            token_embeds,
            inputs["input_features"],
            inputs["audio_feature_lengths"],
            inputs["audio_chunk_mapping"],
        )
        transcript_ids = processor.tokenizer.encode(
            case["transcript"], add_special_tokens=False, return_tensors="pt"
        ).to(device)
        transcript_embeds = model.get_input_embeddings()(transcript_ids)
        complete = torch.cat((fused, transcript_embeds), dim=1).squeeze(0).float().cpu()
    return complete, int(fused.shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    args = parse_args()
    train = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    acceptance = json.loads(args.acceptance_manifest.read_text(encoding="utf-8"))
    acceptance_hashes = all_hashes(acceptance)
    cases = [case for case in train["cases"] if case.get("split") == "train"]
    overlap = sorted(case["sha256"] for case in cases if case["sha256"] in acceptance_hashes)
    if overlap:
        raise ValueError(f"calibration/acceptance overlap: {overlap}")

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, dtype=torch.bfloat16
    ).to(args.device).eval()
    tensors: dict[str, Any] = {}
    candidates = []
    for case in sorted(cases, key=lambda item: item["id"]):
        embeddings, prompt_tokens = extract_case(model, processor, case, args.device)
        tensors[case["id"]] = embeddings
        for phase, starts in candidate_starts(prompt_tokens, embeddings.shape[0]).items():
            for start in starts:
                candidates.append({"case_id": case["id"], "phase": phase, "start": start})
    selected = select_balanced(candidates)
    windows = np.stack(
        [
            tensors[row["case_id"]][row["start"] : row["start"] + WINDOW_SIZE].numpy()
            for row in selected
        ]
    ).astype(np.float16)
    logits_max_abs = 0.0
    with torch.inference_mode():
        for window, row in zip(windows, selected):
            inputs_embeds = torch.from_numpy(window).unsqueeze(0).to(args.device, torch.bfloat16)
            start = int(row["start"])
            position_ids = torch.arange(
                start, start + WINDOW_SIZE, device=args.device, dtype=torch.long
            ).unsqueeze(0)
            logits = model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                use_cache=False,
            ).logits
            logits_max_abs = max(logits_max_abs, float(logits.abs().max()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        embeddings=windows,
        starts=np.asarray([row["start"] for row in selected], dtype=np.int64),
    )
    record = {
        "format": "meetnote.moss_decoder_calibration.v1",
        "model_revision": MODEL_REVISION,
        "source_manifest_sha256": sha256(args.train_manifest),
        "acceptance_manifest_sha256": sha256(args.acceptance_manifest),
        "no_acceptance_overlap": True,
        "window_size": WINDOW_SIZE,
        "window_count": len(selected),
        "phase_counts": {phase: sum(row["phase"] == phase for row in selected) for phase in PHASES},
        "hf_bf16_logits_max_abs": logits_max_abs,
        "windows": selected,
        "file": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
    }
    args.manifest_output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
