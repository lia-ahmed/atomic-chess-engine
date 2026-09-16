"""Tests for historical policy+value target/loss semantics.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from historical_pv import combined_loss, historical_value_target


class HistoricalPolicyValueTests(unittest.TestCase):
    def test_value_target_white_to_move(self) -> None:
        self.assertEqual(historical_value_target(1, 0), 1.0)
        self.assertEqual(historical_value_target(-1, 0), -1.0)
        self.assertEqual(historical_value_target(0, 0), 0.0)

    def test_value_target_black_to_move(self) -> None:
        self.assertEqual(historical_value_target(1, 1), -1.0)
        self.assertEqual(historical_value_target(-1, 1), 1.0)
        self.assertEqual(historical_value_target(0, 1), 0.0)

    def test_value_target_rejects_invalid_metadata(self) -> None:
        with self.assertRaises(ValueError):
            historical_value_target(2, 0)
        with self.assertRaises(ValueError):
            historical_value_target(1, 3)

    def test_combined_loss_is_finite_and_backpropagates(self) -> None:
        logits = torch.randn(4, 7, requires_grad=True)
        raw_values = torch.randn(4, requires_grad=True)
        values = torch.tanh(raw_values)
        actions = torch.tensor([0, 2, 3, 6], dtype=torch.long)
        targets = torch.tensor([1.0, -1.0, 0.0, 1.0], dtype=torch.float32)

        total, policy, value = combined_loss(
            logits,
            values,
            actions,
            targets,
            policy_weight=1.0,
            value_weight=1.0,
        )
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(policy))
        self.assertTrue(torch.isfinite(value))
        total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(raw_values.grad)


if __name__ == "__main__":
    unittest.main()
