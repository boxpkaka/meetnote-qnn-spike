import unittest

import torch

from build_moss_static_attention_poc import AttentionConfig
from build_moss_static_decoder_shard_poc import (
    MossStaticDecoderShard,
    make_shard_inputs,
    manual_reference,
)


class StaticDecoderShardPocTest(unittest.TestCase):
    def test_two_layer_shard_matches_independent_reference(self):
        torch.manual_seed(11)
        config = AttentionConfig(
            dim=16,
            hidden_dim=24,
            n_heads=2,
            n_kv_heads=1,
            head_dim=8,
            max_context_len=8,
        )
        model = MossStaticDecoderShard(config, 2).eval()
        inputs = make_shard_inputs(config, 5, 2, active_cache_tokens=5)

        actual = model(*inputs)
        expected = manual_reference(model, inputs)

        self.assertEqual(len(actual), 5)
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_shard_input_contract(self):
        config = AttentionConfig(
            dim=16,
            hidden_dim=24,
            n_heads=2,
            n_kv_heads=1,
            head_dim=8,
            max_context_len=8,
        )
        inputs = make_shard_inputs(config, 3, 4, active_cache_tokens=3)

        self.assertEqual(len(inputs), 11)
        self.assertEqual(inputs[0].shape, (1, 1, 16))
        self.assertEqual(inputs[3].shape, (1, 1, 8, 7))
        self.assertEqual(inputs[4].shape, (1, 1, 7, 8))


if __name__ == "__main__":
    unittest.main()
