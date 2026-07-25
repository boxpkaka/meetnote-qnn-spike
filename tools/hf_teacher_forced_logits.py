#!/usr/bin/env python3
"""Dump Hugging Face causal-LM logits with the MeetNote teacher-forcing contract."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


SYSTEM_PROMPT = (
    "你是 MeetNote 的端侧会议纪要结构化抽取引擎。"
    "不要输出思考过程，不要输出 Markdown，只输出用户要求的 JSON。"
)


def qwen_chat_prompt(user_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def read_expected_metadata(path: Path) -> tuple[dict, list[dict]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("type") != "header":
        raise ValueError(f"invalid expected metadata: {path}")
    return records[0], [record for record in records[1:] if record.get("type") == "step"]


def validate_token_contract(
    prompt_ids: list[int],
    reference_ids: list[int],
    expected_path: Path,
) -> None:
    header, steps = read_expected_metadata(expected_path)
    expected_prompt_tokens = int(header["prompt_tokens"])
    expected_reference_tokens = int(header["reference_tokens"])
    if len(prompt_ids) != expected_prompt_tokens:
        raise ValueError(
            f"prompt token count differs: HF={len(prompt_ids)} "
            f"expected={expected_prompt_tokens}"
        )
    if len(reference_ids) != expected_reference_tokens:
        raise ValueError(
            f"reference token count differs: HF={len(reference_ids)} "
            f"expected={expected_reference_tokens}"
        )
    for step in steps:
        index = int(step["step"])
        if index + 1 >= len(reference_ids):
            raise ValueError(f"expected step is outside reference tokens: {index}")
        actual = (reference_ids[index], reference_ids[index + 1])
        expected = (int(step["input_token"]), int(step["target_token"]))
        if actual != expected:
            raise ValueError(
                f"reference token ids differ at step {index}: "
                f"HF={actual} expected={expected}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--expected-metadata", type=Path)
    parser.add_argument("--token-ids-output", type=Path)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.validate_only and args.output_prefix is None:
        raise ValueError("--output-prefix is required unless --validate-only is set")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        use_fast=True,
    )
    user_prompt = args.prompt.read_text(encoding="utf-8")
    reference = args.reference.read_text(encoding="utf-8")
    prompt_ids = tokenizer.encode(qwen_chat_prompt(user_prompt), add_special_tokens=False)
    reference_ids = tokenizer.encode(reference, add_special_tokens=False)

    if args.token_ids_output:
        args.token_ids_output.parent.mkdir(parents=True, exist_ok=True)
        args.token_ids_output.write_text(
            json.dumps(
                {
                    "format": "meetnote.token_ids.v1",
                    "prompt_ids": prompt_ids,
                    "reference_ids": reference_ids,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.expected_metadata:
        validate_token_contract(prompt_ids, reference_ids, args.expected_metadata)
    print(
        json.dumps(
            {
                "prompt_tokens": len(prompt_ids),
                "reference_tokens": len(reference_ids),
                "token_contract": "valid" if args.expected_metadata else "unchecked",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.validate_only:
        return

    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()

    steps = min(len(reference_ids) - 1, args.max_steps)
    if steps <= 0:
        raise ValueError("reference must encode to at least two tokens")

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(f"{prefix}.jsonl")
    logits_path = Path(f"{prefix}.f32")
    with (
        metadata_path.open("w", encoding="utf-8") as metadata,
        logits_path.open("wb") as binary,
        torch.inference_mode(),
    ):
        prefill_start = time.perf_counter()
        result = model(
            input_ids=torch.tensor([prompt_ids], dtype=torch.long),
            use_cache=True,
        )
        cache = result.past_key_values
        prefill_ms = round((time.perf_counter() - prefill_start) * 1000)

        header = {
            "type": "header",
            "format": "meetnote.teacher_logits.v1",
            "dtype": "float32_le",
            "runtime": "transformers",
            "model_path": str(args.model),
            "model_revision": args.model_revision,
            "model_type": model.config.model_type,
            "model_dtype": str(model.dtype),
            "attention": "sdpa",
            "prompt_tokens": len(prompt_ids),
            "reference_tokens": len(reference_ids),
            "steps": steps,
            "vocab_size": model.config.vocab_size,
            "top_k": args.top_k,
            "prefill_ms": prefill_ms,
        }
        metadata.write(json.dumps(header, ensure_ascii=False) + "\n")

        for step in range(steps):
            input_token = reference_ids[step]
            target_token = reference_ids[step + 1]
            start = time.perf_counter()
            result = model(
                input_ids=torch.tensor([[input_token]], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )
            forward_ms = round((time.perf_counter() - start) * 1000)
            cache = result.past_key_values
            logits = result.logits[0, -1].float().contiguous()
            binary.write(logits.numpy().astype("<f4", copy=False).tobytes())

            selected = min(args.top_k, logits.numel())
            top_values, top_indices = torch.topk(logits, selected)
            target_logit = float(logits[target_token])
            record = {
                "type": "step",
                "step": step,
                "input_token": input_token,
                "input_piece": tokenizer.decode(
                    [input_token],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "target_token": target_token,
                "target_piece": tokenizer.decode(
                    [target_token],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "target_logit": target_logit,
                "target_rank": int((logits > target_logit).sum()) + 1,
                "finite_logits": int(torch.isfinite(logits).sum()),
                "forward_ms": forward_ms,
                "top": [
                    {
                        "token": int(token),
                        "piece": tokenizer.decode(
                            [int(token)],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                        "logit": float(value),
                    }
                    for value, token in zip(top_values, top_indices)
                ],
            }
            metadata.write(json.dumps(record, ensure_ascii=False) + "\n")
            metadata.flush()
            print(
                f"step={step + 1}/{steps} forward_ms={forward_ms} "
                f"top1={int(top_indices[0])} target={target_token}",
                flush=True,
            )


if __name__ == "__main__":
    main()
