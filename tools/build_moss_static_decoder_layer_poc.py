#!/usr/bin/env python3
"""Build and validate one complete MOSS/Qwen3 static decoder layer on QNN."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable

import torch

from build_moss_static_attention_poc import (
    DEFAULT_POSITIONS,
    AttentionConfig,
    Qwen3Attention,
    _rms_norm,
    export_qnn,
    make_inputs,
    parse_positions,
    sha256_file,
    tensor_metrics,
)


WEIGHT_MAP = {
    "attention.wq.weight": "model.language_model.layers.0.self_attn.q_proj.weight",
    "attention.wk.weight": "model.language_model.layers.0.self_attn.k_proj.weight",
    "attention.wv.weight": "model.language_model.layers.0.self_attn.v_proj.weight",
    "attention.wo.weight": "model.language_model.layers.0.self_attn.o_proj.weight",
    "attention.q_norm_fn.weight": "model.language_model.layers.0.self_attn.q_norm.weight",
    "attention.k_norm_fn.weight": "model.language_model.layers.0.self_attn.k_norm.weight",
    "input_layernorm.weight": "model.language_model.layers.0.input_layernorm.weight",
    "post_attention_layernorm.weight": (
        "model.language_model.layers.0.post_attention_layernorm.weight"
    ),
    "mlp.gate_proj.weight": "model.language_model.layers.0.mlp.gate_proj.weight",
    "mlp.up_proj.weight": "model.language_model.layers.0.mlp.up_proj.weight",
    "mlp.down_proj.weight": "model.language_model.layers.0.mlp.down_proj.weight",
}


class Qwen3MLP(torch.nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.gate_proj = torch.nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = torch.nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = torch.nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.use_conv = False

    def prepare_feed_forward_conv(self) -> None:
        """Match Qualcomm's production static-Llama feed-forward layout."""
        config = self.config
        projection_dtype = self.gate_proj.weight.dtype
        self.gate_proj_conv = torch.nn.Conv2d(
            config.dim, config.hidden_dim, 1, bias=False
        ).to(dtype=projection_dtype)
        self.up_proj_conv = torch.nn.Conv2d(
            config.dim, config.hidden_dim, 1, bias=False
        ).to(dtype=projection_dtype)
        self.down_proj_conv = torch.nn.Conv2d(
            config.hidden_dim, config.dim, 1, bias=False
        ).to(dtype=projection_dtype)
        self.gate_proj_conv.weight.data.copy_(self.gate_proj.weight[:, :, None, None])
        self.up_proj_conv.weight.data.copy_(self.up_proj.weight[:, :, None, None])
        self.down_proj_conv.weight.data.copy_(self.down_proj.weight[:, :, None, None])
        del self.gate_proj, self.up_proj, self.down_proj
        self.use_conv = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_conv:
            return self.down_proj(
                torch.nn.functional.silu(self.gate_proj(hidden_states))
                * self.up_proj(hidden_states)
            )
        config = self.config
        projected = hidden_states.reshape(1, config.ar_len, 1, config.dim)
        projected = projected.transpose(1, 3)
        projected = torch.nn.functional.silu(self.gate_proj_conv(projected)) * self.up_proj_conv(
            projected
        )
        output = self.down_proj_conv(projected).transpose(1, 3)
        return output.reshape(1, config.ar_len, config.dim)


class MossStaticDecoderLayer(torch.nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.attention = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = torch.nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = torch.nn.RMSNorm(
            config.dim, eps=config.norm_eps
        )
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.int64).float()
                / config.head_dim
            )
        )
        angles = torch.outer(torch.arange(config.max_context_len).float(), inv_freq)
        self.register_buffer("freqs_cos", torch.cos(angles), persistent=False)
        self.register_buffer("freqs_sin", torch.sin(angles), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        residual = hidden_states
        normalized = _rms_norm(
            hidden_states, self.input_layernorm.weight, self.config.norm_eps
        )
        freqs_cos = self.freqs_cos[position][0]
        freqs_sin = self.freqs_sin[position][0]
        attention_output, new_k, new_v = self.attention(
            normalized,
            freqs_cos,
            freqs_sin,
            attention_mask,
            k_cache,
            v_cache,
        )
        hidden_states = residual + attention_output
        normalized = _rms_norm(
            hidden_states,
            self.post_attention_layernorm.weight,
            self.config.norm_eps,
        )
        return hidden_states + self.mlp(normalized), new_k, new_v


def load_weights(model: MossStaticDecoderLayer, checkpoint: Path) -> Dict[str, dict]:
    from safetensors import safe_open

    state = model.state_dict()
    report: Dict[str, dict] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        available = set(source.keys())
        missing = sorted(set(WEIGHT_MAP.values()) - available)
        if missing:
            raise KeyError(f"checkpoint is missing layer-0 decoder weights: {missing}")
        for target, source_key in WEIGHT_MAP.items():
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


def manual_reference(model: MossStaticDecoderLayer, inputs):
    hidden, mask, position, k_cache, v_cache = inputs
    config = model.config
    residual = hidden
    normalized = _rms_norm(hidden, model.input_layernorm.weight, config.norm_eps)
    attention = model.attention
    q = torch.nn.functional.linear(normalized, attention.wq.weight)
    k = torch.nn.functional.linear(normalized, attention.wk.weight)
    v = torch.nn.functional.linear(normalized, attention.wv.weight)
    q = q.view(1, config.ar_len, config.n_heads, config.head_dim).transpose(1, 2)
    k = k.view(1, config.ar_len, config.n_kv_heads, config.head_dim).transpose(1, 2)
    v = v.view(1, config.ar_len, config.n_kv_heads, config.head_dim).transpose(1, 2)
    q = _rms_norm(q, attention.q_norm_fn.weight, config.norm_eps)
    k = _rms_norm(k, attention.k_norm_fn.weight, config.norm_eps)
    cos = model.freqs_cos[position][0]
    sin = model.freqs_sin[position][0]
    q_real, q_imag = q.chunk(2, dim=-1)
    k_real, k_imag = k.chunk(2, dim=-1)
    q = torch.cat((q_real * cos - q_imag * sin, q_real * sin + q_imag * cos), dim=-1)
    k = torch.cat((k_real * cos - k_imag * sin, k_real * sin + k_imag * cos), dim=-1)
    new_k = k.transpose(2, 3)
    kh = torch.repeat_interleave(torch.cat((k_cache, new_k), dim=-1), 2, dim=1)
    vh = torch.repeat_interleave(torch.cat((v_cache, v), dim=2), 2, dim=1)
    scores = torch.matmul(q, kh) / config.head_dim**0.5
    probs = torch.softmax(scores + mask, dim=-1)
    attention_output = torch.matmul(probs, vh).transpose(1, 2).reshape(1, 1, -1)
    attention_output = torch.nn.functional.linear(attention_output, attention.wo.weight)
    hidden = residual + attention_output
    normalized = _rms_norm(
        hidden, model.post_attention_layernorm.weight, config.norm_eps
    )
    mlp = model.mlp
    mlp_output = torch.nn.functional.linear(
        torch.nn.functional.silu(
            torch.nn.functional.linear(normalized, mlp.gate_proj.weight)
        )
        * torch.nn.functional.linear(normalized, mlp.up_proj.weight),
        mlp.down_proj.weight,
    )
    return hidden + mlp_output, new_k, v


@torch.no_grad()
def validate_cpu(
    model: MossStaticDecoderLayer,
    positions: Iterable[int],
    *,
    active_cache_tokens: int | None = 32,
) -> dict:
    results = {}
    for position in positions:
        inputs = make_inputs(
            model.config,
            position,
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
            "new_k": metrics[1],
            "new_v": metrics[2],
        }
    return results


def save_fixture(
    model: MossStaticDecoderLayer,
    position: int,
    directory: Path,
    *,
    dtype: torch.dtype,
    fp32_reference=None,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    inputs = make_inputs(
        model.config,
        position,
        active_cache_tokens=32,
        dtype=dtype,
    )
    with torch.no_grad():
        outputs = model(*inputs)
    names = ("hidden", "mask", "position", "k_cache", "v_cache")
    output_names = ("hidden", "new_k", "new_v")
    manifest = {"position": position, "inputs": [], "outputs": []}
    for name, tensor in zip(names, inputs):
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
    parser.add_argument("--positions", type=parse_positions, default=DEFAULT_POSITIONS)
    parser.add_argument("--fixture-position", type=int, default=3996)
    parser.add_argument("--max-context-len", type=int, default=10240)
    parser.add_argument("--use-mha2sha", action="store_true")
    parser.add_argument(
        "--grouped-gqa-bmm",
        action="store_true",
        help="preserve 8 KV heads using grouped 3-D BMM attention",
    )
    parser.add_argument(
        "--grouped-gqa-unrolled",
        action="store_true",
        help="preserve 8 KV heads using eight explicit attention groups",
    )
    parser.add_argument("--io-fp16", action="store_true")
    parser.add_argument(
        "--fp16a8w",
        action="store_true",
        help="quantize Conv/Linear weights to INT8 while keeping FP16 activations",
    )
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    grouped_modes = int(args.grouped_gqa_bmm) + int(args.grouped_gqa_unrolled)
    if grouped_modes > 1:
        parser.error("choose only one grouped GQA implementation")
    if args.grouped_gqa_bmm and args.use_mha2sha:
        parser.error("grouped BMM cannot be combined with MHA-to-SHA")
    config = AttentionConfig(
        max_context_len=args.max_context_len,
        grouped_gqa_bmm=args.grouped_gqa_bmm,
        grouped_gqa_unrolled=args.grouped_gqa_unrolled,
    )
    invalid_positions = [
        position for position in (*args.positions, args.fixture_position)
        if not 0 <= position < config.max_context_len
    ]
    if invalid_positions:
        raise ValueError(
            f"positions exceed max context {config.max_context_len}: {invalid_positions}"
        )
    model = MossStaticDecoderLayer(config).eval()
    weight_report = load_weights(model, args.checkpoint)
    cpu_validation = validate_cpu(model, args.positions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    io_dtype = torch.float16 if args.io_fp16 else torch.float32
    fp32_reference = None
    if args.io_fp16:
        fp32_inputs = make_inputs(
            config,
            args.fixture_position,
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
        fp32_reference=fp32_reference,
    )
    quant_suffix = "_fp16a8w" if args.fp16a8w else ""
    pte_path = args.output_dir / (
        f"moss_layer0_static_decoder_fp16{quant_suffix}_sm8850.pte"
    )
    if not args.skip_export:
        if grouped_modes:
            model.attention.prepare_attention_conv()
        model.mlp.prepare_feed_forward_conv()
        export_inputs = make_inputs(config, 0, active_cache_tokens=0, dtype=io_dtype)
        reference_inputs = make_inputs(
            config,
            args.fixture_position,
            active_cache_tokens=32,
            dtype=io_dtype,
        )
        quantized_reference = export_qnn(
            model,
            export_inputs,
            pte_path,
            args.soc_model,
            use_mha2sha=args.use_mha2sha,
            fp16a8w=args.fp16a8w,
            reference_inputs=reference_inputs,
        )
        if quantized_reference is not None:
            fixture["fp16a8w_outputs"] = []
            for name, tensor in zip(("hidden", "new_k", "new_v"), quantized_reference):
                path = args.output_dir / "fixture" / f"reference_fp16a8w_{name}.raw"
                tensor.contiguous().numpy().tofile(path)
                fixture["fp16a8w_outputs"].append(
                    {
                        "name": name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "file": path.name,
                    }
                )
            (args.output_dir / "fixture" / "manifest.json").write_text(
                json.dumps(fixture, indent=2) + "\n"
            )

    report = {
        "config": asdict(config),
        "positions": list(args.positions),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "weights": weight_report,
        "cpu_validation": cpu_validation,
        "fixture": fixture,
        "use_mha2sha": args.use_mha2sha,
        "io_dtype": str(io_dtype),
        "weight_quantization": "fp16a8w" if args.fp16a8w else None,
        "pte": str(pte_path) if pte_path.exists() else None,
        "pte_sha256": sha256_file(pte_path) if pte_path.exists() else None,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
