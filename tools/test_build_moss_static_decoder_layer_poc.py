import unittest

import torch

from build_moss_static_attention_poc import AttentionConfig
from build_moss_static_decoder_layer_poc import Qwen3MLP


class StaticDecoderLayerPocTest(unittest.TestCase):
    def test_feed_forward_conv_matches_linear_layout(self):
        torch.manual_seed(7)
        config = AttentionConfig(dim=16, hidden_dim=24, n_heads=2, n_kv_heads=1, head_dim=8)
        model = Qwen3MLP(config).eval()
        hidden = torch.randn(1, 1, config.dim)
        expected = model(hidden)

        model.prepare_feed_forward_conv()
        actual = model(hidden)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_feed_forward_conv_preserves_fp16_dtype(self):
        config = AttentionConfig(dim=16, hidden_dim=24, n_heads=2, n_kv_heads=1, head_dim=8)
        model = Qwen3MLP(config).half().eval()

        model.prepare_feed_forward_conv()

        self.assertEqual(model.gate_proj_conv.weight.dtype, torch.float16)
        self.assertEqual(model.up_proj_conv.weight.dtype, torch.float16)
        self.assertEqual(model.down_proj_conv.weight.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
