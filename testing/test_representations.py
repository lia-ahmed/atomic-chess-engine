"""Tests for src/representations.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import chess
import chess.variant
import representations as rep


class RepresentationTests(unittest.TestCase):
    def test_starting_position(self) -> None:
        board = chess.variant.AtomicBoard()
        planes = rep.board_to_planes(board)

        self.assertEqual(planes.shape, (14, 8, 8))
        self.assertEqual(planes.dtype, np.uint8)
        self.assertEqual(int(planes[:12].sum()), 32)
        self.assertEqual(int(planes[12].sum()), 0)
        np.testing.assert_array_equal(planes, rep.fen_to_planes(board.fen()))

        # Orientation check: a1 contains White rook, e1 White king.
        self.assertEqual(planes[3, 0, 0], 1)  # White rook on a1.
        self.assertEqual(planes[5, 0, 4], 1)  # White king on e1.

    def test_black_side_to_move_plane(self) -> None:
        board = chess.variant.AtomicBoard()
        board.push_uci("e2e4")
        planes = rep.board_to_planes(board)

        self.assertEqual(int(planes[12].sum()), 64)
        np.testing.assert_array_equal(planes, rep.fen_to_planes(board.fen()))

    def test_atomic_capture_case(self) -> None:
        board = chess.variant.AtomicBoard()
        for uci in ("e2e4", "d7d5"):
            move = chess.Move.from_uci(uci)
            self.assertIn(move, board.legal_moves)
            board.push(move)

        capture_move = chess.Move.from_uci("e4d5")
        self.assertIn(capture_move, board.legal_moves)
        self.assertTrue(board.is_capture(capture_move))
        board.push(capture_move)

        planes = rep.board_to_planes(board)
        self.assertEqual(int(planes[:12].sum()), len(board.piece_map()))
        np.testing.assert_array_equal(planes, rep.fen_to_planes(board.fen()))
        blob = rep.planes_to_flat_bytes(planes)
        self.assertEqual(len(blob), 14 * 8 * 8)
        np.testing.assert_array_equal(rep.flat_bytes_to_planes(blob), planes)

    def test_action_map_round_trip(self) -> None:
        action_map = {"a2a3": 0, "e2e4": 1, "e7e8q": 2}
        self.assertEqual(rep.uci_to_action_index("E2E4", action_map), 1)
        self.assertEqual(rep.action_index_to_uci(2, action_map), "e7e8q")

    def test_build_action_map_from_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "tiny.parquet"
            out_path = Path(tmpdir) / "action_map.json"
            table = pa.table(
                {
                    "uci_move": ["e2e4", "g1f3", "e2e4", "e7e8q"],
                    "fen": ["x", "x", "x", "x"],
                }
            )
            pq.write_table(table, parquet_path)

            action_map = rep.build_action_map(parquet_path)
            self.assertEqual(
                action_map,
                {"e2e4": 0, "e7e8q": 1, "g1f3": 2},
            )

            rep.save_action_map(action_map, out_path)
            self.assertEqual(rep.load_action_map(out_path), action_map)
            with out_path.open("r", encoding="utf-8") as fp:
                self.assertEqual(json.load(fp), action_map)


if __name__ == "__main__":
    unittest.main()
