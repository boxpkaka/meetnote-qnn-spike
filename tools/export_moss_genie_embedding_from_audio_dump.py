#!/usr/bin/env python3
"""Build and decode a MOSS prompt from native audio-encoder tensor dumps."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--audio-back-prefix", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--prompt")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_audio_embeddings(torch, inputs: dict, prefix: Path, hidden_size: int):
    lengths = inputs["audio_feature_lengths"].cpu().tolist()
    mapping = inputs["audio_chunk_mapping"].cpu().tolist()
    per_audio = [[] for _ in range(max(mapping) + 1)]
    paths = []
    for chunk_index, (token_length, audio_index) in enumerate(zip(lengths, mapping)):
        path = Path(f"{prefix}-chunk{chunk_index}-back.f32")
        values = torch.from_file(
            str(path), dtype=torch.float32, size=375 * hidden_size
        ).reshape(1, 375, hidden_size).clone()
        per_audio[audio_index].append(values[:, : int(token_length)])
        paths.append(path)
    features = [torch.cat(parts, dim=1) for parts in per_audio]
    return features, paths


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from run_moss_transcribe_diarize_eval import replace_text_decoder_from_gguf
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        prepare_inputs,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype="auto",
    ).to(dtype=torch.float32).cpu().eval()
    replace_text_decoder_from_gguf(model, args.gguf, sha256(args.gguf), torch.float32)
    model = model.cpu().eval()
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    messages = (
        build_transcription_messages(args.audio, prompt=args.prompt)
        if args.prompt is not None
        else build_transcription_messages(args.audio)
    )
    inputs = prepare_inputs(
        processor,
        messages,
        max_length=8192,
        device=torch.device("cpu"),
    )
    prompt_len = int(inputs["attention_mask"][0].sum().item())
    input_ids = inputs["input_ids"][:, :prompt_len]
    attention_mask = inputs["attention_mask"][:, :prompt_len]
    hidden_size = int(model.config.text_config.hidden_size)
    audio_features, dump_paths = load_audio_embeddings(
        torch, inputs, args.audio_back_prefix, hidden_size
    )

    with torch.inference_mode():
        prompt_embeddings = model.model.get_input_embeddings()(input_ids)
        audio_embeds = torch.cat([part.squeeze(0) for part in audio_features], dim=0)
        audio_mask = model.model.get_placeholder_mask(
            input_ids, prompt_embeddings, audio_embeds
        )
        prompt_embeddings = prompt_embeddings.masked_scatter(
            audio_mask, audio_embeds.to(prompt_embeddings.dtype)
        ).float().contiguous()

        generation_config = copy.deepcopy(model.generation_config)
        generation_config.max_new_tokens = args.max_new_tokens
        generation_config.do_sample = False
        outputs = model.generate(
            inputs_embeds=prompt_embeddings,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    generated_ids = outputs[0].cpu().tolist()
    generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    prompt_fp32 = args.out_dir / "prompt-embeddings-fp32.bin"
    prompt_fp16 = args.out_dir / "prompt-embeddings-fp16.bin"
    prompt_embeddings.numpy().tofile(prompt_fp32)
    prompt_embeddings.half().numpy().tofile(prompt_fp16)
    metadata = {
        "model": args.model,
        "gguf": str(args.gguf.resolve()),
        "ggufSha256": sha256(args.gguf),
        "audio": str(args.audio.resolve()),
        "audioSha256": sha256(args.audio),
        "prompt": args.prompt,
        "audioBackDumps": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in dump_paths
        ],
        "audioFeatureLengths": inputs["audio_feature_lengths"].cpu().tolist(),
        "audioChunkMapping": inputs["audio_chunk_mapping"].cpu().tolist(),
        "promptTokens": prompt_len,
        "embeddingSize": hidden_size,
        "promptEmbeddingShape": list(prompt_embeddings.shape),
        "promptEmbeddingFp32Path": str(prompt_fp32.resolve()),
        "promptEmbeddingFp32Sha256": sha256(prompt_fp32),
        "promptEmbeddingFp16Path": str(prompt_fp16.resolve()),
        "promptEmbeddingFp16Sha256": sha256(prompt_fp16),
        "generatedTokenIds": generated_ids,
        "generatedText": generated_text,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
