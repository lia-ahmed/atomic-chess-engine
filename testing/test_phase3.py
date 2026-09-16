"""Tests for dataset pairing and policy model.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chess.variant
import dataset as dsmod
import models
import representations as rep


class Phase3Tests(unittest.TestCase):
    def _write_tiny_per_ply(self, path: Path) -> None:
        rows = []
        games = [
            ("gameA", ["e2e4", "e7e5", "g1f3", "b8c6"]),
            ("gameB", ["d2d4", "d7d5", "c1f4"]),
        ]
        for gameid, moves in games:
            board = chess.variant.AtomicBoard()
            for ply_index, uci in enumerate(moves, start=1):
                board.push_uci(uci)
                rows.append(
                    {
                        "gameid": gameid,
                        "ply_index": ply_index,
                        "t": np.float32(ply_index / len(moves)),
                        "side_to_move": np.int8(0 if board.turn else 1),
                        "white_elo": np.int32(1800),
                        "black_elo": np.int32(1750),
                        "result": np.int8(1),
                        "uci_move": uci,
                        "fen": board.fen(),
                        "piece_planes": rep.planes_to_flat_bytes(rep.board_to_planes(board)),
                    }
                )
        schema = pa.schema(
            [
                pa.field("gameid", pa.string()),
                pa.field("ply_index", pa.int32()),
                pa.field("t", pa.float32()),
                pa.field("side_to_move", pa.int8()),
                pa.field("white_elo", pa.int32()),
                pa.field("black_elo", pa.int32()),
                pa.field("result", pa.int8()),
                pa.field("uci_move", pa.string()),
                pa.field("fen", pa.string()),
                pa.field("piece_planes", pa.binary(14 * 8 * 8)),
            ]
        )
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, row_group_size=2)

    def test_dataset_pairs_state_with_next_move(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet = Path(tmpdir) / "per_ply.parquet"
            self._write_tiny_per_ply(parquet)
            action_map = {
                move: i
                for i, move in enumerate(
                    sorted({"e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "d7d5", "c1f4"})
                )
            }
            with dsmod.AtomicChessDataset(
                parquet,
                action_map,
                split="all",
                val_fraction=0.0,
                rebuild_index=True,
                row_group_cache_size=2,
            ) as dataset:
                # 4-ply game -> 3 samples; 3-ply game -> 2 samples.
                self.assertEqual(len(dataset), 5)

                state, action, metadata = dataset[0]
                self.assertEqual(tuple(state.shape), (14, 8, 8))
                self.assertEqual(state.dtype, torch.float32)
                self.assertEqual(metadata["gameid"], "gameA")
                self.assertEqual(metadata["ply_index"], 1)
                # State after e2e4 must predict e7e5, not e2e4.
                self.assertEqual(metadata["target_uci"], "e7e5")
                self.assertEqual(int(action), action_map["e7e5"])

    def test_row_group_sampler_visits_every_sample_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet = Path(tmpdir) / "per_ply.parquet"
            self._write_tiny_per_ply(parquet)
            action_map = {
                move: i
                for i, move in enumerate(
                    sorted({"e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "d7d5", "c1f4"})
                )
            }
            with dsmod.AtomicChessDataset(
                parquet,
                action_map,
                split="all",
                val_fraction=0.0,
                rebuild_index=True,
            ) as dataset:
                sampler = dsmod.RowGroupShuffleSampler(dataset, seed=123)
                first = list(iter(sampler))
                sampler.set_epoch(1)
                second = list(iter(sampler))
                self.assertEqual(sorted(first), list(range(len(dataset))))
                self.assertEqual(sorted(second), list(range(len(dataset))))

    def test_model_forward_and_backward(self) -> None:
        model = models.MovePolicyNet(num_actions=23, width=16, blocks=2, policy_channels=8)
        x = torch.randn(4, 14, 8, 8)
        targets = torch.tensor([0, 1, 2, 3])
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (4, 23))
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
