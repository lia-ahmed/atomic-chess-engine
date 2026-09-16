"""Unit tests for policy+value model, PUCT MCTS and self-play dataset.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chess.variant
import mcts
import models
import trainer


class Phase4Tests(unittest.TestCase):
    def test_policy_value_forward_backward(self) -> None:
        net = models.PolicyValueNet(num_actions=23, width=16, blocks=2, policy_channels=8,
                                    value_channels=8, value_hidden=16)
        x = torch.randn(4, 14, 8, 8)
        logits, values = net(x)
        self.assertEqual(tuple(logits.shape), (4, 23))
        self.assertEqual(tuple(values.shape), (4,))
        self.assertTrue(torch.all(values <= 1.0))
        self.assertTrue(torch.all(values >= -1.0))
        loss = logits.mean() + values.mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in net.parameters()))

    def test_supervised_transfer_preserves_policy(self) -> None:
        torch.manual_seed(7)
        phase3 = models.MovePolicyNet(num_actions=19, width=16, blocks=1, policy_channels=8)
        phase3.eval()
        x = torch.randn(2, 14, 8, 8)
        expected = phase3(x).detach()
        checkpoint = {"model_config": asdict(phase3.config), "model_state": phase3.state_dict()}

        phase4 = models.PolicyValueNet(num_actions=19, width=16, blocks=1, policy_channels=8,
                                       value_channels=8, value_hidden=16)
        models.transfer_supervised_weights(phase4, checkpoint, zero_value=True)
        phase4.eval()
        actual, values = phase4(x)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(values, torch.zeros_like(values))

    def test_mcts_runs_ten_simulations_and_returns_legal_move(self) -> None:
        board = chess.variant.AtomicBoard()
        legal = sorted(move.uci() for move in board.legal_moves)
        action_map = {uci: idx for idx, uci in enumerate(legal)}
        net = models.PolicyValueNet(
            num_actions=len(action_map), width=8, blocks=0, policy_channels=4,
            value_channels=4, value_hidden=8,
        )
        net.zero_value_output()
        searcher = mcts.MCTS(net, action_map, num_simulations=10, c_puct=1.5, seed=3)
        result = searcher.search(board)
        self.assertIn(result.move_uci, legal)
        self.assertEqual(sum(result.visit_counts.values()), 10)
        self.assertAlmostEqual(sum(result.visit_probs.values()), 1.0, places=6)

    def test_selfplay_dataset_and_collate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "selfplay_iter000_worker000.jsonl"
            board = chess.variant.AtomicBoard()
            records = [
                {
                    "game_id": "g1",
                    "iteration": 0,
                    "ply": 0,
                    "fen": board.fen(),
                    "side_to_move": 0,
                    "policy_indices": [0, 2],
                    "policy_probs": [0.75, 0.25],
                    "played_uci": "e2e4",
                    "played_action_index": 0,
                    "root_value": 0.0,
                    "simulations": 10,
                    "value_target": 1.0,
                },
                {
                    "game_id": "g2",
                    "iteration": 0,
                    "ply": 0,
                    "fen": board.fen(),
                    "side_to_move": 0,
                    "policy_indices": [1],
                    "policy_probs": [1.0],
                    "played_uci": "d2d4",
                    "played_action_index": 1,
                    "root_value": 0.0,
                    "simulations": 10,
                    "value_target": -1.0,
                },
            ]
            with path.open("w", encoding="utf-8") as fp:
                for rec in records:
                    fp.write(json.dumps(rec) + "\n")
            dataset = trainer.SelfPlayDataset(root, split="all", val_fraction=0.0, rebuild_index=True)
            try:
                batch = trainer.collate_selfplay([dataset[0], dataset[1]], num_actions=3)
                states, policies, values = batch
                self.assertEqual(tuple(states.shape), (2, 14, 8, 8))
                self.assertEqual(tuple(policies.shape), (2, 3))
                self.assertEqual(tuple(values.shape), (2,))
                torch.testing.assert_close(policies.sum(dim=1), torch.ones(2))
            finally:
                dataset.close()


if __name__ == "__main__":
    unittest.main()
