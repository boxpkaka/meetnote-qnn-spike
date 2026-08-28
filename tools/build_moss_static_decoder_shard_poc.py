#!/usr/bin/env python3
"""Build and validate a small MOSS/Qwen3 static decoder shard on QNN."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable

import torch

from build_moss_static_attention_poc import (
    AttentionConfig,
    export_qnn,
    make_inputs,
    parse_positions,
    sha256_file,
    tensor_metrics,
)
from build_moss_static_decoder_layer_poc import (
    WEIGHT_MAP as LAYER_WEIGHT_MAP,
    MossStaticDecoderLayer,
    manual_reference as layer_manual_reference,
)


class MossStaticDecoderShard(torch.nn.Module):
    def __init__(self, config: AttentionConfig, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.config = config
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList(
            MossStaticDecoderLayer(config) for _ in range(num_layers)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position: torch.Tensor,
        *layer_caches: torch.Tensor,
    ):
        if len(layer_caches) != self.num_layers * 2:
            raise ValueError(
                f"expected {self.num_layers * 2} cache tensors, got {len(layer_caches)}"
            )
        outputs = []
        for layer_index, layer in enumerate(self.layers):
            k_cache = layer_caches[layer_index * 2]
            v_cache = layer_caches[layer_index * 2 + 1]
            hidden_states, new_k, new_v = layer(
                hidden_states,
                attention_mask,
                position,
                k_cache,
                v_cache,
            )
            outputs.extend((new_k, new_v))
        return (hidden_states, *outputs)


def make_shard_inputs(
    config: AttentionConfig,
    position: int,
    num_layers: int,
    *,
    seed: int = 20260825,
    active_cache_tokens: int | None = 32,
    dtype: torch.dtype = torch.float32,
):
    hidden, mask, position_tensor, _, _ = make_inputs(
        config,
        position,
        seed=seed,
        active_cache_tokens=active_cache_tokens,
        dtype=dtype,
    )
    caches = []
    for layer_index in range(num_layers):
        _, _, _, k_cache, v_cache = make_inputs(
            config,
            position,
            seed=seed + 1000 * (layer_index + 1),
            active_cache_tokens=active_cache_tokens,
            dtype=dtype,
        )
        caches.extend((k_cache, v_cache))
    return hidden, mask, position_tensor, *caches


def load_weights(model: MossStaticDecoderShard, checkpoint: Path) -> Dict[str, dict]:
    from safetensors import safe_open

    state = model.state_dict()
    report: Dict[str, dict] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        available = set(source.keys())
        for layer_index in range(model.num_layers):
            for local_target, layer_zero_source in LAYER_WEIGHT_MAP.items():
                target = f"layers.{layer_index}.{local_target}"
                source_key = layer_zero_source.replace(
                    ".layers.0.", f".layers.{layer_index}."
                )
                if source_key not in available:
                    raise KeyError(f"checkpoint is missing decoder weight: {source_key}")
                value = source.get_tensor(source_key).float()
                expected = state[target]
                if value.shape != expected.shape:
                    raise ValueError(
                        f"shape mismatch for {source_key}: {tuple(value.shape)} != "
                        f"{tuple(expected.shape)}"
                    )
                state[target] = value
                report[target] = {
                    "source": source_key,
                    "shape": list(value.shape),
                    "source_dtype": str(source.get_tensor(source_key).dtype),
                }
    model.load_state_dict(state, strict=True)
    return report


def manual_reference(model: MossStaticDecoderShard, inputs):
    hidden, mask, position, *layer_caches = inputs
    outputs = []
    for layer_index, layer in enumerate(model.layers):
        layer_inputs = (
            hidden,
            mask,
            position,
            layer_caches[layer_index * 2],
            layer_caches[layer_index * 2 + 1],
        )
        hidden, new_k, new_v = layer_manual_reference(layer, layer_inputs)
        outputs.extend((new_k, new_v))
    return (hidden, *outputs)


@torch.no_grad()
def validate_cpu(
    model: MossStaticDecoderShard,
    positions: Iterable[int],
    *,
    active_cache_tokens: int | None = 32,
) -> dict:
    results = {}
    for position in positions:
        inputs = make_shard_inputs(
            model.config,
            position,
            model.num_layers,
            seed=20260825 + position,
            active_cache_tokens=active_cache_tokens,
        )
        actual = model(*inputs)
        expected = manual_reference(model, inputs)
        metrics = [tensor_metrics(a, e) for a, e in zip(actual, expected)]
        if not all(metric["finite"] and metric["max_abs"] <= 2e-5 for metric in metrics):
            raise AssertionError(
                f"CPU reference mismatch at position {position}: {metrics}"
            )
        results[str(position)] = {
            "hidden": metrics[0],
            "cache_outputs": metrics[1:],
        }
    return results


def save_fixture(
    model: MossStaticDecoderShard,
    position: int,
    directory: Path,
    *,
    dtype: torch.dtype,
    fp32_reference=None,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    inputs = make_shard_inputs(
        model.config,
        position,
        model.num_layers,
        active_cache_tokens=32,
        dtype=dtype,
    )
    with torch.no_grad():
        outputs = model(*inputs)
    input_names = ["hidden", "mask", "position"]
    output_names = ["hidden"]
    for layer_index in range(model.num_layers):
        input_names.extend(
            (f"layer{layer_index}_k_cache", f"layer{layer_index}_v_cache")
        )
        output_names.extend(
            (f"layer{layer_index}_new_k", f"layer{layer_index}_new_v")
        )
    manifest = {
        "position": position,
        "num_layers": model.num_layers,
        "inputs": [],
        "outputs": [],
    }
    for name, tensor in zip(input_names, inputs):
        path = directory / f"input_{name}.raw"
        tensor.contiguous().numpy().tofile(path)
        manifest["inputs"].append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "file": path.name,
            }
        )
    for name, tensor in zip(output_names, outputs):
        path = directory / f"reference_{name}.raw"
        tensor.contiguous().numpy().tofile(path)
        manifest["outputs"].append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "file": path.name,
            }
        )
    if fp32_reference is not None:
        manifest["fp32_outputs"] = []
        for name, tensor in zip(output_names, fp32_reference):
            path = directory / f"reference_fp32_{name}.raw"
            tensor.float().contiguous().numpy().tofile(path)
            manifest["fp32_outputs"].append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": "torch.float32",
                    "file": path.name,
                }
            )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--soc-model", default="SM8850")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--positions", type=parse_positions, default=(2047, 2048, 3996))
    parser.add_argument("--fixture-position", type=int, default=3996)
    parser.add_argument("--max-context-len", type=int, default=4096)
    parser.add_argument("--use-mha2sha", action="store_true")
    parser.add_argument("--io-fp16", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    config = AttentionConfig(max_context_len=args.max_context_len)
    invalid_positions = [
        position
        for position in (*args.positions, args.fixture_position)
        if not 0 <= position < config.max_context_len
    ]
    if invalid_positions:
        raise ValueError(
            f"positions exceed max context {config.max_context_len}: {invalid_positions}"
        )
    model = MossStaticDecoderShard(config, args.num_layers).eval()
    weight_report = load_weights(model, args.checkpoint)
    cpu_validation = validate_cpu(model, args.positions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    io_dtype = torch.float16 if args.io_fp16 else torch.float32
    fp32_inputs = make_shard_inputs(
        config,
        args.fixture_position,
        args.num_layers,
        active_cache_tokens=32,
        dtype=torch.float32,
    )
    with torch.no_grad():
        fp32_reference = model(*fp32_inputs)
    if args.io_fp16:
        model.half()
    fixture = save_fixture(
        model,
        args.fixture_position,
        args.output_dir / "fixture",
        dtype=io_dtype,
        fp32_reference=fp32_reference if args.io_fp16 else None,
    )
    pte_path = args.output_dir / (
        f"moss_layers0_{args.num_layers - 1}_static_decoder_fp16_sm8850.pte"
    )
    if not args.skip_export:
        for layer in model.layers:
            layer.mlp.prepare_feed_forward_conv()
        export_inputs = make_shard_inputs(
            config,
            0,
            args.num_layers,
            active_cache_tokens=0,
            dtype=io_dtype,
        )
        export_qnn(
            model,
            export_inputs,
            pte_path,
            args.soc_model,
            use_mha2sha=args.use_mha2sha,
        )

    report = {
        "config": asdict(config),
        "num_layers": args.num_layers,
        "positions": list(args.positions),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "weights": weight_report,
        "cpu_validation": cpu_validation,
        "fixture": fixture,
        "use_mha2sha": args.use_mha2sha,
        "io_dtype": str(io_dtype),
        "pte": str(pte_path) if pte_path.exists() else None,
        "pte_sha256": sha256_file(pte_path) if pte_path.exists() else None,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
