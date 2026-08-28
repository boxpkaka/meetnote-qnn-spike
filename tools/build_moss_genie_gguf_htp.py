#!/usr/bin/env python3
"""Build a Genie HTP container from the MOSS text-decoder GGUF artifact."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from qti.aisw.graph_gen.architectures.qwen.model import Qwen3
from qti.aisw.graph_gen.mad.lib.module import ModuleMarker


class Qwen3ExternalEmbeddings(Qwen3):
    def forward(
        self,
        inputs_embeds,
        position_ids_sin,
        position_ids_cos,
        attention_mask,
        past_key_in,
        past_value_in,
    ):
        out = inputs_embeds
        outs = []
        for i in range(self.num_hidden_layers):
            out, past_key_out, past_value_out = self.decoder_blocks[i](
                out,
                position_ids_sin,
                position_ids_cos,
                attention_mask,
                past_key_in[i],
                past_value_in[i],
            )
            outs.append(past_key_out)
            outs.append(past_value_out)
            if i == self.last_layer_index:
                break
            if (i + 1) % self.split_size == 0:
                ModuleMarker.split_at(out)

        out = self.output_norm(out)
        out = self.pre_reshape(out)
        out = self.pre_tranpose(out)
        out = self.lm_head(out)
        out = self.post_transpose(out)
        out = self.logits(out)
        return out, *outs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", required=True, type=int)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--prefill-arn", type=int, default=64)
    parser.add_argument("--chipset", default="SM8850")
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument(
        "--split-embedding",
        action="store_true",
        help="Set the vendor split flag (GGUF MAD currently ignores this flag).",
    )
    parser.add_argument(
        "--external-embeddings",
        action="store_true",
        help="Build the decoder with an FP32 inputs_embeds tensor instead of token IDs.",
    )
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--debug-compile", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def write_pretrained_config(root: Path, tokenizer: Path, layers: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": layers,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_act": "silu",
        "max_position_embeddings": 10240,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "rope_theta": 1_000_000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151645,
        "eos_token_id": 151645,
        "pad_token_id": 151643,
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copyfile(tokenizer, root / "tokenizer.json")
    return root


def install_qwen3_compat(config_dir: Path, gguf_path: Path) -> None:
    # Composer writes Qwen3 tensors with the legacy GGUF architecture name
    # "qwen". QAIRT's MAD graph generator recognizes the same layout as
    # "qwen3". Alias only the reader maps and keep the vendor SDK unchanged.
    # Importing the QAIRT GGUF package installs its patched Transformers GGUF
    # reader modules. The import order is required; upstream Transformers lacks
    # QAIRT's Qwen3 tensor mappings.
    from qti.aisw.converters import gguf_builder as _qti_gguf_builder
    import transformers.integrations.ggml as ggml
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    _ = _qti_gguf_builder  # Import installs QAIRT's GGUF reader registrations.

    ggml.GGUF_CONFIG_MAPPING["qwen"] = ggml.GGUF_CONFIG_MAPPING["qwen3"]
    ggml.GGUF_TENSOR_MAPPING["qwen"] = ggml.GGUF_TENSOR_MAPPING["qwen3"]
    gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen"] = (
        gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen3"]
    )
    gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["tensors"]["qwen"] = (
        gguf_utils.GGUF_TO_TRANSFORMERS_MAPPING["tensors"]["qwen3"]
    )
    if "qwen" not in gguf_utils.GGUF_SUPPORTED_ARCHITECTURES:
        gguf_utils.GGUF_SUPPORTED_ARCHITECTURES.append("qwen")

    original_load = gguf_utils.load_gguf_checkpoint

    def load_qwen3(*args, **kwargs):
        result = original_load(*args, **kwargs)
        if result.get("config", {}).get("model_type") == "qwen":
            result["config"]["model_type"] = "qwen3"
        return result

    gguf_utils.load_gguf_checkpoint = load_qwen3

    from qti.aisw.converters.gguf_builder import gguf_parser

    gguf_parser.load_gguf_checkpoint = load_qwen3

    from qairt.modules.gguf_module import GGUFModule

    def use_moss_config(_self):
        return config_dir

    GGUFModule._extract_config = use_moss_config

    if gguf_path.name.upper().endswith("-F16.GGUF"):
        # An empty override file makes MAD enter quantization mode and abort.
        # FP16 GGUF has no quantized parameter encodings, so serialize the
        # native float graph without an overrides path.
        def no_quantization_overrides(_self, _mode=None):
            return None

        GGUFModule.get_encoding_file = no_quantization_overrides


def install_external_embedding_graph() -> None:
    """Replace only the GGUF MAD graph front end with an embeddings-input variant."""
    from qairt.modules.gguf_module.graph_gen import GraphGenerator
    from qti.aisw.graph_gen.mad.graph.components import TensorDataInfo
    from qti.aisw.graph_gen.mad.utils import update_io_names

    def update_external_embedding_io_names(ir_graph):
        update_io_names(ir_graph)
        for tensor in ir_graph.get_input_tensors_to_graph():
            name = tensor.name()
            if name == "inputs_embeds" or name.endswith(".inputs_embeds"):
                ir_graph.update_tensor_name(name, "inputs_embeds")
                break

    def generate_external_embedding_graph(
        config, encoding_file, save_path, filename_prefix, weights
    ):
        model = Qwen3ExternalEmbeddings(config)
        inputs_embeds = TensorDataInfo(
            (config.batch, config.seq_length, config.hidden_size)
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        rope_dim = int(config.head_dim * partial_rotary_factor) // 2
        positional_ids_sin = TensorDataInfo((1, 1, config.seq_length, rope_dim))
        positional_ids_cos = TensorDataInfo((1, 1, config.seq_length, rope_dim))
        attention_mask = TensorDataInfo(
            (1, 1, config.seq_length, config.max_position_embeddings)
        )
        past_keys = [
            TensorDataInfo(
                (
                    config.num_key_value_heads,
                    config.batch,
                    config.head_dim,
                    config.max_position_embeddings - config.seq_length,
                )
            )
        ] * config.num_hidden_layers
        past_values = [
            TensorDataInfo(
                (
                    config.num_key_value_heads,
                    config.batch,
                    config.max_position_embeddings - config.seq_length,
                    config.head_dim,
                )
            )
        ] * config.num_hidden_layers

        model.export(
            input_shapes=[
                inputs_embeds,
                positional_ids_sin,
                positional_ids_cos,
                attention_mask,
                past_keys,
                past_values,
            ],
            weight_dict=weights,
            pre_serialization_callback=update_external_embedding_io_names,
            save_dir=save_path,
            filename_prefix=filename_prefix,
            encoding_file_path=encoding_file,
        )
        return GraphGenerator._get_dlc_paths(
            config.split, filename_prefix, save_path
        )

    GraphGenerator._generate_graph = staticmethod(generate_external_embedding_graph)


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    config_dir = write_pretrained_config(
        args.work_dir / "pretrained-config", args.tokenizer, args.layers
    )
    install_qwen3_compat(config_dir, args.gguf)
    if args.external_embeddings:
        install_external_embedding_graph()

    from qairt.gen_ai_api.builders.gguf_builder_htp import GGUFBuilderHTP

    builder = GGUFBuilderHTP.from_pretrained(
        args.gguf, cache_root=args.work_dir / "cache"
    )
    transformer = builder._transformation_config.model_transformer_config
    transformer.arn_cl_options.auto_regression_number = (
        [1] if args.decode_only else [1, args.prefill_arn]
    )
    transformer.arn_cl_options.context_length = [args.context_length]
    builder.config.context_length = args.context_length
    transformer.split_model.num_splits = args.num_splits
    transformer.split_model.split_embedding = args.split_embedding
    builder.weight_sharing = not args.decode_only
    builder.set_targets([f"chipset:{args.chipset}"])
    if args.debug_compile:
        from qairt.utils.loggers import get_logger

        get_logger("qairt.compile").setLevel(logging.DEBUG)
        builder._compilation_config.log_level = "debug"

    summary = {
        "builder": type(builder).__name__,
        "gguf": str(args.gguf.resolve()),
        "layers": args.layers,
        "context_length": transformer.arn_cl_options.context_length,
        "auto_regression_number": transformer.arn_cl_options.auto_regression_number,
        "num_splits": transformer.split_model.num_splits,
        "split_embedding": transformer.split_model.split_embedding,
        "external_embeddings": args.external_embeddings,
        "chipset": args.chipset,
        "weight_sharing": builder.weight_sharing,
        "gen_ai_config": builder.config.model_dump(),
    }
    print(json.dumps(summary, indent=2, default=str), flush=True)
    if args.inspect_only:
        return

    container = builder.build()
    container.save(args.output_dir, exist_ok=True)
    print(f"saved_container={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
