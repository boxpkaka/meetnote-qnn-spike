#!/usr/bin/env python3
"""Build and validate a one-layer MOSS/Qwen3 static attention QNN PoC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch


DEFAULT_POSITIONS = (0, 1, 63, 64, 2047, 2048, 2049, 3995, 3996, 7000, 10239)


@dataclass(frozen=True)
class AttentionConfig:
    dim: int = 1024
    hidden_dim: int = 3072
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    max_context_len: int = 10240
    norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    ar_len: int = 1
    use_qk_norm: bool = True
    grouped_gqa_bmm: bool = False
    grouped_gqa_unrolled: bool = False

    @property
    def cache_len(self) -> int:
        return self.max_context_len - self.ar_len

    def kv_bytes(self, *, layers: int, element_bytes: int) -> int:
        return (
            2
            * layers
            * self.n_kv_heads
            * self.head_dim
            * self.max_context_len
            * element_bytes
        )


WEIGHT_MAP = {
    "attention.wq.weight": "model.language_model.layers.0.self_attn.q_proj.weight",
    "attention.wk.weight": "model.language_model.layers.0.self_attn.k_proj.weight",
    "attention.wv.weight": "model.language_model.layers.0.self_attn.v_proj.weight",
    "attention.wo.weight": "model.language_model.layers.0.self_attn.o_proj.weight",
    "attention.q_norm_fn.weight": "model.language_model.layers.0.self_attn.q_norm.weight",
    "attention.k_norm_fn.weight": "model.language_model.layers.0.self_attn.k_norm.weight",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mask(
    config: AttentionConfig,
    position: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not 0 <= position < config.max_context_len:
        raise ValueError(f"position must be in [0, {config.max_context_len}), got {position}")
    mask = torch.full(
        (1, 1, config.ar_len, config.max_context_len), -255.0, dtype=dtype
    )
    mask[..., :position] = 0.0
    mask[..., -config.ar_len :] = 0.0
    return mask


class Qwen3Attention(torch.nn.Module):
    """The official static Qwen3 attention dataflow, isolated from package side effects."""

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.wq = torch.nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = torch.nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = torch.nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = torch.nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.q_norm_fn = torch.nn.RMSNorm(config.head_dim, eps=config.norm_eps)
        self.k_norm_fn = torch.nn.RMSNorm(config.head_dim, eps=config.norm_eps)
        self.use_conv = False

    def prepare_attention_conv(self) -> None:
        """Match Qualcomm's production static-Llama projection dataflow."""
        if self.use_conv:
            return
        config = self.config
        self.wq_conv = torch.nn.Conv2d(
            config.dim, config.n_heads * config.head_dim, 1, bias=False
        )
        self.wk_conv = torch.nn.Conv2d(
            config.dim, config.n_kv_heads * config.head_dim, 1, bias=False
        )
        self.wv_conv = torch.nn.Conv2d(
            config.dim, config.n_kv_heads * config.head_dim, 1, bias=False
        )
        self.wo_conv = torch.nn.Conv2d(
            config.n_heads * config.head_dim, config.dim, 1, bias=False
        )
        projection_dtype = self.wq.weight.dtype
        self.wq_conv.to(dtype=projection_dtype)
        self.wk_conv.to(dtype=projection_dtype)
        self.wv_conv.to(dtype=projection_dtype)
        self.wo_conv.to(dtype=projection_dtype)
        self.wq_conv.weight.data.copy_(self.wq.weight[:, :, None, None])
        self.wk_conv.weight.data.copy_(self.wk.weight[:, :, None, None])
        self.wv_conv.weight.data.copy_(self.wv.weight[:, :, None, None])
        self.wo_conv.weight.data.copy_(self.wo.weight[:, :, None, None])
        del self.wq, self.wk, self.wv, self.wo
        self.use_conv = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        config = self.config
        if self.use_conv:
            projected = hidden_states.reshape(1, config.ar_len, 1, config.dim)
            projected = projected.transpose(1, 3)
            q = self.wq_conv(projected).permute(0, 3, 1, 2).squeeze(-1)
            k = self.wk_conv(projected).permute(0, 3, 1, 2).squeeze(-1)
            v = self.wv_conv(projected).permute(0, 3, 1, 2).squeeze(-1)
        else:
            q = self.wq(hidden_states)
            k = self.wk(hidden_states)
            v = self.wv(hidden_states)
        q = q.view(1, config.ar_len, config.n_heads, config.head_dim).transpose(1, 2)
        k = k.view(1, config.ar_len, config.n_kv_heads, config.head_dim).transpose(1, 2)
        v = v.view(1, config.ar_len, config.n_kv_heads, config.head_dim).transpose(1, 2)
        if config.use_qk_norm:
            # Keep Q/K RMSNorm decomposed into elementwise ops. ExecuTorch's
            # MHA-to-SHA pass cannot split aten.rms_norm with its 1-D weight input.
            q = _rms_norm(q, self.q_norm_fn.weight, config.norm_eps)
            k = _rms_norm(k, self.k_norm_fn.weight, config.norm_eps)
        q = _rope(q, freqs_cos, freqs_sin)
        k = _rope(k, freqs_cos, freqs_sin)
        new_k = k.transpose(2, 3)
        kh = torch.cat((k_cache, new_k), dim=-1)
        vh = torch.cat((v_cache, v), dim=2)
        repeats = config.n_heads // config.n_kv_heads
        if config.grouped_gqa_bmm:
            if config.ar_len != 1:
                raise ValueError("grouped GQA BMM currently supports decode ar_len=1 only")
            # Batch the eight KV groups directly. This keeps each K/V head once
            # and avoids both 5-D broadcast MatMul and MHA-to-SHA KV expansion.
            q_grouped = q.reshape(
                config.n_kv_heads, repeats, config.head_dim
            )
            k_grouped = kh.reshape(
                config.n_kv_heads, config.head_dim, config.max_context_len
            )
            v_grouped = vh.reshape(
                config.n_kv_heads, config.max_context_len, config.head_dim
            )
            scores = torch.bmm(q_grouped, k_grouped) / math.sqrt(config.head_dim)
            grouped_mask = attention_mask.reshape(1, 1, config.max_context_len)
            probs = torch.softmax(scores + grouped_mask, dim=-1)
            output = torch.bmm(probs, v_grouped).reshape(
                1, config.n_heads, config.ar_len, config.head_dim
            ).transpose(1, 2)
        elif config.grouped_gqa_unrolled:
            if config.ar_len != 1:
                raise ValueError("unrolled grouped GQA currently supports decode ar_len=1 only")
            grouped_outputs = []
            for group_index in range(config.n_kv_heads):
                head_begin = group_index * repeats
                head_end = head_begin + repeats
                group_scores = torch.matmul(
                    q[:, head_begin:head_end], kh[:, group_index : group_index + 1]
                ) / math.sqrt(config.head_dim)
                group_probs = torch.softmax(group_scores + attention_mask, dim=-1)
                grouped_outputs.append(
                    torch.matmul(
                        group_probs, vh[:, group_index : group_index + 1]
                    )
                )
            output = torch.cat(grouped_outputs, dim=1).transpose(1, 2)
        else:
            kh = kh[:, :, None, :, :].expand(
                1, config.n_kv_heads, repeats, config.head_dim, config.max_context_len
            )
            kh = kh.reshape(1, config.n_heads, config.head_dim, config.max_context_len)
            vh = vh[:, :, None, :, :].expand(
                1, config.n_kv_heads, repeats, config.max_context_len, config.head_dim
            )
            vh = vh.reshape(1, config.n_heads, config.max_context_len, config.head_dim)
            scores = torch.matmul(q, kh) / math.sqrt(config.head_dim)
            probs = torch.softmax(scores + attention_mask, dim=-1)
            output = torch.matmul(probs, vh).transpose(1, 2)
        if self.use_conv:
            output = output.reshape(1, config.ar_len, 1, -1).transpose(1, 3)
            output = self.wo_conv(output).transpose(1, 3).reshape(1, config.ar_len, -1)
        else:
            output = self.wo(output.reshape(1, config.ar_len, -1))
        # Keep cache-update outputs outside the MHA-to-SHA rewrite. They are
        # also consumed by attention, and rewriting a shared producer causes
        # the QNN graph to alias every KV output head to the first head.
        return output, new_k.clone(), v.clone()


class MossStaticAttention(torch.nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.attention = Qwen3Attention(config)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.int64).float()
                / config.head_dim
            )
        )
        angles = torch.outer(torch.arange(config.max_context_len).float(), inv_freq)
        cos, sin = torch.cos(angles), torch.sin(angles)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        # Keep position integer-valued through table lookup. It must never travel via FP16.
        freqs_cos = self.freqs_cos[position][0]
        freqs_sin = self.freqs_sin[position][0]
        return self.attention(
            hidden_states,
            freqs_cos,
            freqs_sin,
            attention_mask,
            k_cache,
            v_cache,
        )


class AttentionOutputOnly(torch.nn.Module):
    """Export wrapper for runtimes that keep KV updates in internal buffers."""

    def __init__(self, model: MossStaticAttention):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        return self.model(
            hidden_states, attention_mask, position, k_cache, v_cache
        )[0]


def load_weights(model: MossStaticAttention, checkpoint: Path) -> Dict[str, dict]:
    from safetensors import safe_open

    state = model.state_dict()
    report: Dict[str, dict] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        available = set(source.keys())
        missing = sorted(set(WEIGHT_MAP.values()) - available)
        if missing:
            raise KeyError(f"checkpoint is missing layer-0 attention weights: {missing}")
        for target, source_key in WEIGHT_MAP.items():
            value = source.get_tensor(source_key).float()
            expected = state[target]
            expanded_kv_heads = False
            if (
                target in {"attention.wk.weight", "attention.wv.weight"}
                and value.shape[1:] == expected.shape[1:]
                and expected.shape[0] % value.shape[0] == 0
            ):
                repeats = expected.shape[0] // value.shape[0]
                value = (
                    value.view(-1, model.config.head_dim, value.shape[1])
                    .repeat_interleave(repeats, dim=0)
                    .reshape(expected.shape)
                )
                expanded_kv_heads = True
            if value.shape != expected.shape:
                raise ValueError(
                    f"shape mismatch for {source_key}: {tuple(value.shape)} != {tuple(expected.shape)}"
                )
            state[target] = value
            report[target] = {
                "source": source_key,
                "shape": list(value.shape),
                "source_dtype": str(source.get_tensor(source_key).dtype),
                "expanded_kv_heads": expanded_kv_heads,
            }
    model.load_state_dict(state, strict=True)
    return report


def make_inputs(
    config: AttentionConfig,
    position: int,
    *,
    seed: int = 20260825,
    active_cache_tokens: int | None = None,
    dtype: torch.dtype = torch.float32,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(
        (1, config.ar_len, config.dim), generator=generator, dtype=torch.float32
    ).to(dtype)
    k_cache = torch.zeros(
        (1, config.n_kv_heads, config.head_dim, config.cache_len), dtype=dtype
    )
    v_cache = torch.zeros(
        (1, config.n_kv_heads, config.cache_len, config.head_dim), dtype=dtype
    )
    active = position if active_cache_tokens is None else min(position, active_cache_tokens)
    if active:
        k_cache[..., :active] = torch.randn(
            (1, config.n_kv_heads, config.head_dim, active),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype)
        v_cache[..., :active, :] = torch.randn(
            (1, config.n_kv_heads, active, config.head_dim),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype)
    mask = build_mask(config, position, dtype=dtype)
    position_tensor = torch.tensor([[position]], dtype=torch.int32)
    return hidden, mask, position_tensor, k_cache, v_cache


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    real = x[..., : x.shape[-1] // 2]
    imag = x[..., x.shape[-1] // 2 :]
    return torch.cat((real * cos - imag * sin, real * sin + imag * cos), dim=-1)


def manual_reference(model: MossStaticAttention, inputs):
    hidden, mask, position, k_cache, v_cache = inputs
    attention = model.attention
    config = model.config
    q = torch.nn.functional.linear(hidden, attention.wq.weight)
    k = torch.nn.functional.linear(hidden, attention.wk.weight)
    v = torch.nn.functional.linear(hidden, attention.wv.weight)
    q = q.view(1, config.ar_len, config.n_heads, config.head_dim).transpose(1, 2)
    k = k.view(1, config.ar_len, config.n_kv_heads, config.head_dim).transpose(1, 2)
    v = v.view(1, config.ar_len, config.n_kv_heads, config.head_dim).transpose(1, 2)
    if config.use_qk_norm:
        q = _rms_norm(q, attention.q_norm_fn.weight, config.norm_eps)
        k = _rms_norm(k, attention.k_norm_fn.weight, config.norm_eps)
    cos = model.freqs_cos[position][0]
    sin = model.freqs_sin[position][0]
    q = _rope(q, cos, sin)
    k = _rope(k, cos, sin)
    new_k = k.transpose(2, 3)
    kh = torch.cat((k_cache, new_k), dim=-1)
    vh = torch.cat((v_cache, v), dim=2)
    repeats = config.n_heads // config.n_kv_heads
    kh = torch.repeat_interleave(kh, repeats, dim=1)
    vh = torch.repeat_interleave(vh, repeats, dim=1)
    scores = torch.matmul(q, kh) / math.sqrt(config.head_dim)
    probs = torch.softmax(scores + mask, dim=-1)
    output = torch.matmul(probs, vh).transpose(1, 2).reshape(1, config.ar_len, -1)
    output = torch.nn.functional.linear(output, attention.wo.weight)
    return output, new_k, v


def tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    return {
        "max_abs": float((actual - expected).abs().max()),
        "mean_abs": float((actual - expected).abs().mean()),
        "cosine": float(torch.nn.functional.cosine_similarity(actual, expected, dim=0)),
        "finite": bool(torch.isfinite(actual).all()),
    }


@torch.no_grad()
def validate_cpu(
    model: MossStaticAttention,
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
        if not all(m["finite"] and m["max_abs"] <= 2e-5 for m in metrics):
            raise AssertionError(f"CPU reference mismatch at position {position}: {metrics}")
        results[str(position)] = {
            "attention": metrics[0],
            "new_k": metrics[1],
            "new_v": metrics[2],
        }
    return results


def export_qnn(
    model: MossStaticAttention,
    sample_inputs,
    output: Path,
    soc_model: str,
    *,
    use_mha2sha: bool,
    fp16a8w: bool = False,
    reference_inputs=None,
):
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_qnn_context_binary_alignment,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    chipset = getattr(QcomChipset, soc_model)
    backend_options = generate_htp_compiler_spec(
        use_fp16=True,
        use_weight_sharing=False,
    )
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=chipset,
        backend_options=backend_options,
        shared_buffer=True,
        use_mha2sha=use_mha2sha,
    )
    if use_mha2sha:
        for module in model.modules():
            if isinstance(module, Qwen3Attention):
                module.prepare_attention_conv()

    quantized_reference = None
    if fp16a8w:
        from executorch.backends.qualcomm.quantizer.quantizer import (
            QnnQuantizer,
            QuantDtype,
        )
        from executorch.backends.qualcomm.serialization.qc_schema import (
            QnnExecuTorchBackendType,
        )
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        exported = torch.export.export(model, sample_inputs, strict=True).module()
        quantizer = QnnQuantizer(
            backend=QnnExecuTorchBackendType.kHtpBackend,
            soc_model=getattr(QcomChipset, soc_model),
        )
        quantizer.set_default_quant_config(
            QuantDtype.use_fp16a8w,
            is_conv_per_channel=True,
            is_linear_per_channel=True,
        )
        prepared = prepare_pt2e(exported, quantizer)
        prepared(*sample_inputs)
        model = convert_pt2e(prepared)
        if reference_inputs is not None:
            with torch.no_grad():
                quantized_reference = model(*reference_inputs)

    lower_model = model if fp16a8w else model.eval()
    edge = to_edge_transform_and_lower_to_qnn(
        lower_model,
        sample_inputs,
        compile_spec,
    )
    executorch_config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False,
            alloc_graph_output=False,
        ),
        segment_alignment=get_qnn_context_binary_alignment(),
    )
    program = edge.to_executorch(config=executorch_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as destination:
        program.write_to_file(destination)
    return quantized_reference


def save_fixture(
    model: MossStaticAttention,
    position: int,
    directory: Path,
    *,
    dtype: torch.dtype = torch.float32,
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
    output_names = ("attention", "new_k", "new_v")
    manifest = {"position": position, "inputs": [], "outputs": []}
    for name, tensor in zip(names, inputs):
        path = directory / f"input_{name}.raw"
        tensor.contiguous().numpy().tofile(path)
        manifest["inputs"].append(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "file": path.name}
        )
    for name, tensor in zip(output_names, outputs):
        path = directory / f"reference_{name}.raw"
        tensor.contiguous().numpy().tofile(path)
        manifest["outputs"].append(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "file": path.name}
        )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_positions(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--soc-model", default="SM8850")
    parser.add_argument(
        "--positions",
        type=parse_positions,
        default=DEFAULT_POSITIONS,
        help="comma-separated CPU validation positions",
    )
    parser.add_argument("--fixture-position", type=int, default=3996)
    parser.add_argument(
        "--use-mha2sha",
        action="store_true",
        help="enable ExecuTorch's experimental MHA-to-SHA graph rewrite",
    )
    parser.add_argument(
        "--attention-output-only",
        action="store_true",
        help="omit diagnostic KV outputs from the exported QNN graph",
    )
    parser.add_argument(
        "--disable-qk-norm",
        action="store_true",
        help="diagnostic export that omits Qwen3 Q/K RMSNorm",
    )
    parser.add_argument("--max-context-len", type=int, default=10240)
    parser.add_argument(
        "--expand-kv-heads",
        action="store_true",
        help="repeat GQA K/V weights and caches to the full query-head count",
    )
    parser.add_argument(
        "--grouped-gqa-bmm",
        action="store_true",
        help="preserve 8 KV heads using two 3-D grouped BMM operations",
    )
    parser.add_argument(
        "--grouped-gqa-unrolled",
        action="store_true",
        help="preserve 8 KV heads using eight explicit two-query-head attention groups",
    )
    parser.add_argument(
        "--io-fp16",
        action="store_true",
        help="use FP16 hidden, mask, KV cache, and output graph tensors",
    )
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    grouped_modes = int(args.grouped_gqa_bmm) + int(args.grouped_gqa_unrolled)
    if grouped_modes and args.expand_kv_heads:
        parser.error("grouped GQA cannot be combined with --expand-kv-heads")
    if args.grouped_gqa_bmm and args.use_mha2sha:
        parser.error("grouped BMM cannot be combined with MHA-to-SHA")
    config = AttentionConfig(
        max_context_len=args.max_context_len,
        n_kv_heads=16 if args.expand_kv_heads else 8,
        use_qk_norm=not args.disable_qk_norm,
        grouped_gqa_bmm=args.grouped_gqa_bmm,
        grouped_gqa_unrolled=args.grouped_gqa_unrolled,
    )
    model = MossStaticAttention(config).eval()
    weight_report = load_weights(model, args.checkpoint)
    cpu_validation = validate_cpu(model, args.positions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    io_dtype = torch.float16 if args.io_fp16 else torch.float32
    if args.io_fp16:
        model.half()
    fixture = save_fixture(
        model, args.fixture_position, args.output_dir / "fixture", dtype=io_dtype
    )
    pte_path = args.output_dir / "moss_layer0_static_attention_fp16_sm8850.pte"
    if not args.skip_export:
        if grouped_modes:
            model.attention.prepare_attention_conv()
        export_inputs = make_inputs(
            config, 0, active_cache_tokens=0, dtype=io_dtype
        )
        export_qnn(
            AttentionOutputOnly(model) if args.attention_output_only else model,
            export_inputs,
            pte_path,
            args.soc_model,
            use_mha2sha=args.use_mha2sha,
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
        "attention_output_only": args.attention_output_only,
        "io_dtype": str(io_dtype),
        "kv_memory": {
            "one_layer_fp16_bytes": config.kv_bytes(layers=1, element_bytes=2),
            "28_layers_fp16_bytes": config.kv_bytes(layers=28, element_bytes=2),
            "28_layers_fp32_bytes": config.kv_bytes(layers=28, element_bytes=4),
        },
        "pte": str(pte_path) if pte_path.exists() else None,
        "pte_sha256": sha256_file(pte_path) if pte_path.exists() else None,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
