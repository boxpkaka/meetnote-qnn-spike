#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentCheckTest(unittest.TestCase):
    def test_missing_required_arguments_return_structured_failure(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/check_environment.py")],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(0, result.returncode)
        self.assertEqual("", result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual("invalid", report["status"])
        self.assertIn("--work-dir is required unless --python-only is used", report["errors"])


if __name__ == "__main__":
    unittest.main()
