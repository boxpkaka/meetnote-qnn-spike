#!/usr/bin/env python3
"""Export a real MOSS audio prompt for Genie embedding-to-token validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    with torch.inference_mode():
        prompt_embeddings = model.model.get_input_embeddings()(input_ids)
        prompt_embeddings = model.model.inject_audio_features(
            input_ids=input_ids,
            inputs_embeds=prompt_embeddings,
            input_features=inputs["input_features"],
            audio_feature_lengths=inputs["audio_feature_lengths"],
            audio_chunk_mapping=inputs["audio_chunk_mapping"],
        ).float().contiguous()

        generation_config = model.generation_config
        generation_config.max_new_tokens = args.max_new_tokens
        generation_config.do_sample = False
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            audio_feature_lengths=inputs["audio_feature_lengths"],
            audio_chunk_mapping=inputs["audio_chunk_mapping"],
            generation_config=generation_config,
        )

    prompt_path = args.out_dir / "prompt-embeddings-fp32.bin"
    prompt_fp16_path = args.out_dir / "prompt-embeddings-fp16.bin"
    table_path = args.out_dir / "token-embedding-table-fp32.bin"
    prompt_embeddings.numpy().tofile(prompt_path)
    prompt_embeddings.half().numpy().tofile(prompt_fp16_path)
    model.model.get_input_embeddings().weight.detach().float().contiguous().numpy().tofile(table_path)

    generated_ids = outputs[0, prompt_len:].cpu().tolist()
    metadata = {
        "model": args.model,
        "gguf": str(args.gguf.resolve()),
        "ggufSha256": sha256(args.gguf),
        "audio": str(args.audio.resolve()),
        "audioSha256": sha256(args.audio),
        "prompt": args.prompt,
        "promptTokens": prompt_len,
        "embeddingSize": prompt_embeddings.shape[-1],
        "promptEmbeddingDtype": "float32",
        "promptEmbeddingShape": list(prompt_embeddings.shape),
        "promptEmbeddingPath": str(prompt_path.resolve()),
        "promptEmbeddingSha256": sha256(prompt_path),
        "promptEmbeddingFp16Path": str(prompt_fp16_path.resolve()),
        "promptEmbeddingFp16Sha256": sha256(prompt_fp16_path),
        "tokenEmbeddingTableShape": list(model.model.get_input_embeddings().weight.shape),
        "tokenEmbeddingTablePath": str(table_path.resolve()),
        "tokenEmbeddingTableSha256": sha256(table_path),
        "generatedTokenIds": generated_ids,
        "generatedText": processor.tokenizer.decode(generated_ids, skip_special_tokens=True),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
