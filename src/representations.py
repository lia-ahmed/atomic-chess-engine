#!/usr/bin/env python3
"""Representation helpers.

Defines the canonical 14x8x8 uint8 representation ie tensor representation and utilities for building/using the supervised UCI action map.

Plane order (matching 'extract_per_ply.py'):
    0..5   white P, N, B, R, Q, K
    6..11  black P, N, B, R, Q, K
    12      side to move: all 0 for White, all 1 for Black
    13      atomic explosion-threat mask for legal captures by side to move


EGS:

Build the action map from the parquet file:

    python src/representations.py \
        --build-action-map data/per_ply.parquet \
        --out action_map.json
    
    note: sometimes windows gets angry at line break syntax. 

Run built-in representation smoke tests:

    python src/representations.py --self-test

Optionally validate that stored plane blobs agree with FEN-derived planes:

    python src/representations.py \
        --validate-parquet data/per_ply.parquet \
        --sample-size 1000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
# import pyarrow.dataset as ds - was breaking things

try:
    import chess
    import chess.variant
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        "python-chess is required. Install the atomic-variant-capable chess "
        "package from your project environment."
    ) from exc


PIECE_ORDER = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

NUM_PLANES = 14
BOARD_SIZE = 8
PLANE_SIZE = BOARD_SIZE * BOARD_SIZE
FLAT_PLANE_SIZE = NUM_PLANES * PLANE_SIZE

UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)

ActionMap = Mapping[str, int]


def atomic_explosion_threat_squares(board: chess.variant.AtomicBoard) -> int:
    """Return a bitboard mask of explosion areas from legal captures.

    For each legal capture available to the side to move, the destination
    square and its king-neighbourhood are marked. This deliberately matches
    the Phase 1 extractor's tactical feature semantics.
    """

    mask = 0
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        center = move.to_square
        mask |= chess.BB_SQUARES[center]
        mask |= chess.BB_KING_ATTACKS[center]
    return mask


def board_to_planes(board: chess.variant.AtomicBoard) -> np.ndarray:
    """Convert an atomic board to a canonical 'uint8' array of shape (14,8,8).
    """

    planes = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)

    plane_index = 0
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in PIECE_ORDER:
            for square in board.pieces(piece_type, color):
                rank = chess.square_rank(square)
                file_ = chess.square_file(square)
                planes[plane_index, rank, file_] = 1
            plane_index += 1

    if board.turn == chess.BLACK:
        planes[12, :, :] = 1

    threat_mask = atomic_explosion_threat_squares(board)
    for square in chess.scan_forward(threat_mask):
        rank = chess.square_rank(square)
        file_ = chess.square_file(square)
        planes[13, rank, file_] = 1

    return planes


def fen_to_planes(fen: str) -> np.ndarray:

    if not isinstance(fen, str) or not fen.strip():
        raise ValueError("fen must be a non-empty string")
    board = chess.variant.AtomicBoard(fen.strip())
    return board_to_planes(board)


def planes_to_flat_bytes(planes: np.ndarray) -> bytes:

    array = np.asarray(planes, dtype=np.uint8)
    if array.shape != (NUM_PLANES, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            f"expected planes shape {(NUM_PLANES, BOARD_SIZE, BOARD_SIZE)}, "
            f"got {array.shape}"
        )
    return np.ascontiguousarray(array).reshape(-1).tobytes()


def flat_bytes_to_planes(blob: bytes | bytearray | memoryview) -> np.ndarray:

    view = memoryview(blob)
    if view.nbytes != FLAT_PLANE_SIZE:
        raise ValueError(
            f"piece_planes must contain exactly {FLAT_PLANE_SIZE} bytes; "
            f"got {view.nbytes}"
        )
    return np.frombuffer(view, dtype=np.uint8).reshape(
        NUM_PLANES, BOARD_SIZE, BOARD_SIZE
    ).copy()


def normalize_uci(uci: str) -> str:
    """Normalize and validate an ordinary chess/atomic UCI move string."""

    if not isinstance(uci, str):
        raise TypeError(f"uci must be str, got {type(uci).__name__}")
    normalized = uci.strip().lower()
    if not UCI_RE.fullmatch(normalized):
        raise ValueError(f"invalid UCI move syntax: {uci!r}")
    # Let python-chess perform a second syntax-level check.
    chess.Move.from_uci(normalized)
    return normalized


def uci_to_action_index(uci: str, action_map: ActionMap) -> int:
    """Return the integer action index assigned to 'uci'.

    Raises 'KeyError' when the move is syntactically valid but absent from
    the observed-move vocabulary.
    """

    normalized = normalize_uci(uci)
    try:
        return int(action_map[normalized])
    except KeyError as exc:
        raise KeyError(f"UCI move {normalized!r} is not in the action map") from exc


def invert_action_map(action_map: ActionMap) -> Dict[int, str]:
    """Return '{index: uci}' and validate that indices are unique/dense."""

    inverse: Dict[int, str] = {}
    for uci, raw_index in action_map.items():
        normalized = normalize_uci(uci)
        index = int(raw_index)
        if index < 0:
            raise ValueError(f"negative action index for {uci!r}: {index}")
        if index in inverse:
            raise ValueError(
                f"duplicate action index {index}: {inverse[index]!r} and {normalized!r}"
            )
        inverse[index] = normalized

    expected = set(range(len(inverse)))
    actual = set(inverse)
    if actual != expected:
        missing = sorted(expected - actual)
        extras = sorted(actual - expected)
        raise ValueError(
            "action-map indices must be dense 0..N-1; "
            f"missing={missing[:10]} extras={extras[:10]}"
        )
    return inverse


def action_index_to_uci(index: int, action_map: ActionMap) -> str:
    """Reverse 'uci_to_action_index' for a validated action map."""

    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError("index must be an integer")
    inverse = invert_action_map(action_map)
    try:
        return inverse[int(index)]
    except KeyError as exc:
        raise KeyError(f"action index {index} is not in the action map") from exc


def build_action_map(parquet_path: str | Path) -> Dict[str, int]:
    """Build a deterministic UCI-index map from one Parquet file or directory.

    Only the 'uci_move' column is scanned. Moves are normalized, deduplicated,
    sorted lexicographically, and assigned dense indices starting at zero.
    """
    import pyarrow.dataset as ds
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(path)

    dataset = ds.dataset(str(path), format="parquet")
    if "uci_move" not in dataset.schema.names:
        raise ValueError(
            f"Parquet dataset does not contain required column 'uci_move'; "
            f"columns={dataset.schema.names}"
        )

    observed: set[str] = set()
    scanner = dataset.scanner(columns=["uci_move"], batch_size=131_072)
    for batch in scanner.to_batches():
        column = batch.column(0)
        for value in column.to_pylist():
            if value is None:
                continue
            observed.add(normalize_uci(value))

    if not observed:
        raise ValueError("no non-null UCI moves found in parquet dataset")

    return {uci: index for index, uci in enumerate(sorted(observed))}


def save_action_map(action_map: ActionMap, out_path: str | Path) -> None:
    """Validate and save an action map as a plain JSON object."""

    inverse = invert_action_map(action_map)
    canonical = {inverse[index]: index for index in range(len(inverse))}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(canonical, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    tmp.replace(out)


def load_action_map(path: str | Path) -> Dict[str, int]:
    """Load and validate an action-map JSON file."""

    with Path(path).open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError("action-map JSON must be an object mapping UCI strings to indices")

    action_map = {str(uci): int(index) for uci, index in raw.items()}
    inverse = invert_action_map(action_map)
    return {inverse[index]: index for index in range(len(inverse))}


def validate_parquet_planes(
    parquet_path: str | Path,
    sample_size: int = 1000,
    seed: int = 0,
) -> tuple[int, int]:
    """Compare stored 'piece_planes' blobs with planes recomputed from FEN.

    Returns '(checked, mismatches)'. Reservoir sampling keeps memory bounded
    even for a large Phase 1 dataset.
    """
    import pyarrow.dataset as ds
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(path)

    dataset = ds.dataset(str(path), format="parquet")
    required = {"fen", "piece_planes"}
    missing = required - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Parquet dataset is missing columns: {sorted(missing)}")

    rng = random.Random(seed)
    reservoir: list[tuple[str, bytes]] = []
    seen = 0

    scanner = dataset.scanner(columns=["fen", "piece_planes"], batch_size=65_536)
    for batch in scanner.to_batches():
        fens = batch.column(0).to_pylist()
        blobs = batch.column(1).to_pylist()
        for fen, blob in zip(fens, blobs):
            if fen is None or blob is None:
                continue
            item = (str(fen), bytes(blob))
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(item)
            else:
                j = rng.randrange(seen)
                if j < sample_size:
                    reservoir[j] = item

    mismatches = 0
    for fen, stored_blob in reservoir:
        recomputed = planes_to_flat_bytes(fen_to_planes(fen))
        if recomputed != stored_blob:
            mismatches += 1

    return len(reservoir), mismatches


def run_self_tests() -> None:
    """Run dependency-light smoke tests without requiring pytest."""

    # 1) Starting position: 32 pieces, White to move.
    start = chess.variant.AtomicBoard()
    planes = board_to_planes(start)
    assert planes.shape == (14, 8, 8)
    assert planes.dtype == np.uint8
    assert int(planes[:12].sum()) == 32
    assert int(planes[12].sum()) == 0
    assert np.array_equal(planes, fen_to_planes(start.fen()))

    # 2) After e2e4, Black is to move, so side plane is all ones.
    after_e4 = chess.variant.AtomicBoard()
    after_e4.push_uci("e2e4")
    planes_e4 = board_to_planes(after_e4)
    assert int(planes_e4[12].sum()) == 64
    assert np.array_equal(planes_e4, fen_to_planes(after_e4.fen()))

    # 3) Atomic capture case. Replay a legal capture under AtomicBoard, then
    # verify that its resulting state is represented consistently.
    atomic_capture = chess.variant.AtomicBoard()
    for uci in ("e2e4", "d7d5"):
        move = chess.Move.from_uci(uci)
        assert move in atomic_capture.legal_moves
        atomic_capture.push(move)
    capture_move = chess.Move.from_uci("e4d5")
    assert capture_move in atomic_capture.legal_moves
    assert atomic_capture.is_capture(capture_move)
    atomic_capture.push(capture_move)
    capture_planes = board_to_planes(atomic_capture)
    assert int(capture_planes[:12].sum()) == len(atomic_capture.piece_map())
    assert np.array_equal(capture_planes, fen_to_planes(atomic_capture.fen()))
    assert flat_bytes_to_planes(planes_to_flat_bytes(capture_planes)).shape == (14, 8, 8)

    # Action-map round trip.
    action_map = {"a2a3": 0, "e2e4": 1, "e7e8q": 2}
    assert uci_to_action_index("E2E4", action_map) == 1
    assert action_index_to_uci(2, action_map) == "e7e8q"

    print("Self-tests passed (3 board/FEN cases + action-map round trip).")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomic-chess Phase 2 representation and action-map utilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--build-action-map",
        metavar="PARQUET",
        help="Build a UCI-index action map from Phase 1 parquet data",
    )
    group.add_argument(
        "--validate-parquet",
        metavar="PARQUET",
        help="Compare stored piece_planes against planes recomputed from FEN",
    )
    group.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in representation tests and exit",
    )
    parser.add_argument(
        "--out",
        default="action_map.json",
        help="Output path used with --build-action-map",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Rows sampled by --validate-parquet",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed for --validate-parquet",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        if args.self_test:
            run_self_tests()
            return 0

        if args.build_action_map:
            action_map = build_action_map(args.build_action_map)
            save_action_map(action_map, args.out)
            print(f"actions={len(action_map)} out={args.out}")
            return 0

        checked, mismatches = validate_parquet_planes(
            args.validate_parquet,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        print(f"checked={checked} mismatches={mismatches}")
        return 0 if mismatches == 0 else 1

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
