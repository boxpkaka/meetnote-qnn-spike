#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


def load_module():
    path = Path(__file__).with_name("strip_mnn_activation_quant.py")
    spec = importlib.util.spec_from_file_location("strip_mnn_activation_quant", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STRIP = load_module()


class StripMnnActivationQuantTest(unittest.TestCase):
    def test_removes_only_activation_quant_descriptions(self):
        document = {
            "extraTensorDescribe": [{"index": 1}, {"index": 2}],
            "oplists": [{"type": "Convolution", "weight": [1, 2, 3]}],
        }
        original_ops = document["oplists"]

        removed = STRIP.strip_activation_quant(document)

        self.assertEqual(2, removed)
        self.assertEqual([], document["extraTensorDescribe"])
        self.assertIs(original_ops, document["oplists"])

    def test_rejects_invalid_description_shape(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            STRIP.strip_activation_quant({"extraTensorDescribe": {}})


if __name__ == "__main__":
    unittest.main()
