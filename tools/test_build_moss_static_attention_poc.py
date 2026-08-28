import unittest

import torch

from build_moss_static_attention_poc import (
    AttentionConfig,
    Qwen3Attention,
    build_mask,
    parse_positions,
)


class StaticAttentionPocTest(unittest.TestCase):
    def test_fp16_kv_memory_matches_qwen3_shape(self):
        config = AttentionConfig()
        self.assertEqual(config.cache_len, 10239)
        self.assertEqual(config.kv_bytes(layers=1, element_bytes=2), 41_943_040)
        self.assertEqual(config.kv_bytes(layers=28, element_bytes=2), 1_174_405_120)

    def test_mask_keeps_history_and_current_token(self):
        config = AttentionConfig()
        mask = build_mask(config, 2048)
        self.assertEqual(mask.shape, (1, 1, 1, 10240))
        self.assertEqual(int((mask == 0).sum()), 2049)
        self.assertTrue(torch.all(mask[..., :2048] == 0))
        self.assertEqual(float(mask[..., 2048]), -255.0)
        self.assertEqual(float(mask[..., -1]), 0.0)

    def test_position_bounds_and_parser(self):
        config = AttentionConfig()
        with self.assertRaises(ValueError):
            build_mask(config, -1)
        with self.assertRaises(ValueError):
            build_mask(config, 10240)
        self.assertEqual(parse_positions("0,2048,10239"), (0, 2048, 10239))

    def test_grouped_gqa_bmm_matches_expanded_attention(self):
        torch.manual_seed(11)
        base_config = AttentionConfig(
            dim=16,
            hidden_dim=24,
            n_heads=2,
            n_kv_heads=1,
            head_dim=8,
            max_context_len=16,
        )
        grouped_config = AttentionConfig(
            dim=16,
            hidden_dim=24,
            n_heads=2,
            n_kv_heads=1,
            head_dim=8,
            max_context_len=16,
            grouped_gqa_bmm=True,
        )
        unrolled_config = AttentionConfig(
            dim=16,
            hidden_dim=24,
            n_heads=2,
            n_kv_heads=1,
            head_dim=8,
            max_context_len=16,
            grouped_gqa_unrolled=True,
        )
        expanded = Qwen3Attention(base_config).eval()
        grouped = Qwen3Attention(grouped_config).eval()
        unrolled = Qwen3Attention(unrolled_config).eval()
        grouped.load_state_dict(expanded.state_dict())
        unrolled.load_state_dict(expanded.state_dict())
        hidden = torch.randn(1, 1, 16)
        cos = torch.randn(4)
        sin = torch.randn(4)
        mask = build_mask(base_config, 7)
        k_cache = torch.randn(1, 1, 8, 15)
        v_cache = torch.randn(1, 1, 15, 8)

        expected = expanded(hidden, cos, sin, mask, k_cache, v_cache)
        actual = grouped(hidden, cos, sin, mask, k_cache, v_cache)
        unrolled_actual = unrolled(hidden, cos, sin, mask, k_cache, v_cache)

        for candidate in (actual, unrolled_actual):
            for grouped_tensor, expanded_tensor in zip(candidate, expected):
                torch.testing.assert_close(
                    grouped_tensor, expanded_tensor, rtol=1e-5, atol=1e-6
                )

if __name__ == "__main__":
    unittest.main()
