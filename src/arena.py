#!/usr/bin/env python3
"""Head-to-head INTERNAL arena for atomic-chess checkpoints.

No root Dirichlet noise and zero-temperature root move selection to enforce determinism. 
Candidate and baseline alternate colors.
Optional random opening plies are paired ie the same opening is played twice, with model colors swapped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import chess
import chess.pgn
import chess.variant

from mcts import MCTS
from models import action_map_digest, load_policy_value_checkpoint
from representations import load_action_map
from ratings import summarize_scores

LOGGER = logging.getLogger("arena")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def load_searcher(
    checkpoint_path: str | Path,
    action_map: Mapping[str, int],
    *,
    device: torch.device,
    simulations: int,
    c_puct: float,
    seed: int,
) -> tuple[MCTS, dict[str, Any]]:
    model, checkpoint = load_policy_value_checkpoint(checkpoint_path, device=device)
    digest = action_map_digest(action_map)
    expected = checkpoint.get("action_map_sha256")
    if expected not in (None, digest):
        raise ValueError(f"checkpoint action map mismatch: {checkpoint_path}")
    if int(model.config.num_actions) != len(action_map):
        raise ValueError(f"network action count mismatch: {checkpoint_path}")
    model.action_map = action_map
    searcher = MCTS(
        model,
        action_map,
        num_simulations=simulations,
        c_puct=c_puct,
        device=device,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        unknown_move_prior_mass=0.0,
        seed=seed,
    )
    return searcher, checkpoint


def paired_opening_moves(pair_index: int, opening_plies: int, seed: int) -> list[str]:
    if opening_plies <= 0:
        return []
    for attempt in range(100):
        board = chess.variant.AtomicBoard()
        rng = np.random.default_rng(seed + pair_index * 1009 + attempt * 7919)
        moves: list[str] = []
        for _ in range(opening_plies):
            if board.outcome(claim_draw=True) is not None:
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[int(rng.integers(len(legal)))]
            moves.append(move.uci())
            board.push(move)
        if board.outcome(claim_draw=True) is None:
            return moves
    raise RuntimeError("could not generate a non-terminal random opening")


def result_for_candidate(board: chess.variant.AtomicBoard, candidate_white: bool) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    candidate_won = (outcome.winner == chess.WHITE) == candidate_white
    return 1.0 if candidate_won else 0.0


def pgn_result(board: chess.variant.AtomicBoard, adjudicated: bool) -> str:
    if adjudicated:
        return "1/2-1/2"
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "1/2-1/2"
    return "1-0" if outcome.winner == chess.WHITE else "0-1"


def build_pgn(
    moves: list[str],
    *,
    candidate_name: str,
    baseline_name: str,
    candidate_white: bool,
    result: str,
    game_index: int,
    opening_plies: int,
) -> str:
    game = chess.pgn.Game()
    game.headers["Event"] = "AtomicChess Internal Arena"
    game.headers["Site"] = "local"
    game.headers["Round"] = str(game_index + 1)
    game.headers["Variant"] = "Atomic"
    game.headers["White"] = candidate_name if candidate_white else baseline_name
    game.headers["Black"] = baseline_name if candidate_white else candidate_name
    game.headers["Result"] = result
    game.headers["OpeningPlies"] = str(opening_plies)

    board = chess.variant.AtomicBoard()
    node = game
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"cannot serialize illegal arena move {uci}")
        node = node.add_variation(move)
        board.push(move)
    return str(game) + "\n\n"


def play_game(
    *,
    game_index: int,
    candidate: MCTS,
    baseline: MCTS,
    candidate_white: bool,
    opening_moves: list[str],
    max_plies: int,
    candidate_name: str,
    baseline_name: str,
) -> tuple[dict[str, Any], str]:
    board = chess.variant.AtomicBoard()
    all_moves: list[str] = []
    for uci in opening_moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"generated opening contains illegal move {uci}")
        board.push(move)
        all_moves.append(uci)

    started = time.perf_counter()
    move_times: list[float] = []
    while board.outcome(claim_draw=True) is None and board.ply() < max_plies:
        candidate_turn = (board.turn == chess.WHITE) == candidate_white
        searcher = candidate if candidate_turn else baseline
        move_started = time.perf_counter()
        result = searcher.search(board, temperature=0.0, add_root_noise=False)
        move_elapsed = time.perf_counter() - move_started
        move_times.append(move_elapsed)
        move = chess.Move.from_uci(result.move_uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"MCTS returned illegal move {result.move_uci} from {board.fen()}")
        board.push(move)
        all_moves.append(result.move_uci)

    terminal = board.outcome(claim_draw=True) is not None
    adjudicated = not terminal
    result_text = pgn_result(board, adjudicated)
    candidate_score = result_for_candidate(board, candidate_white) if terminal else 0.5
    elapsed = time.perf_counter() - started
    record = {
        "game_index": int(game_index),
        "candidate_color": "white" if candidate_white else "black",
        "candidate_score": float(candidate_score),
        "result": result_text,
        "terminal": bool(terminal),
        "adjudicated_max_plies": bool(adjudicated),
        "plies": int(board.ply()),
        "opening_moves": opening_moves,
        "elapsed_seconds": elapsed,
        "mean_move_seconds": float(np.mean(move_times)) if move_times else 0.0,
        "final_fen": board.fen(),
    }
    pgn = build_pgn(
        all_moves,
        candidate_name=candidate_name,
        baseline_name=baseline_name,
        candidate_white=candidate_white,
        result=result_text,
        game_index=game_index,
        opening_plies=len(opening_moves),
    )
    return record, pgn


def read_existing(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}; repair before --resume") from exc
            idx = int(rec["game_index"])
            records[idx] = rec
    return records


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Play internal head-to-head Atomic arena and estimate *relative* Elo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--candidate", required=True, help="candidate Phase 4 checkpoint")
    p.add_argument("--baseline", required=True, help="baseline Phase 4 checkpoint")
    p.add_argument("--action-map-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--opening-plies", type=int, default=4,
                   help="random opening plies, paired with colors reversed; use 0 for start position only")
    p.add_argument("--candidate-name", default="candidate")
    p.add_argument("--baseline-name", default="baseline")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--baseline-rating", type=float, default=None,
                   help="optional nominal internal anchor; does not create an external Elo")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.games <= 0 or args.simulations <= 0 or args.max_plies <= 0:
        raise SystemExit("games, simulations and max-plies must be positive")
    if args.opening_plies < 0:
        raise SystemExit("opening-plies must be non-negative")
    if args.resume and args.overwrite:
        raise SystemExit("choose either --resume or --overwrite, not both")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "games.jsonl"
    pgn_path = out_dir / "games.pgn"
    summary_path = out_dir / "summary.json"

    if args.overwrite:
        for path in (results_path, pgn_path, summary_path):
            if path.exists():
                path.unlink()
    elif (results_path.exists() or pgn_path.exists()) and not args.resume:
        raise SystemExit("arena output already exists; use --resume or --overwrite")

    existing = read_existing(results_path) if args.resume else {}
    device = resolve_device(args.device)
    action_map = load_action_map(args.action_map_path)

    LOGGER.info("loading candidate=%s", args.candidate)
    candidate, candidate_ckpt = load_searcher(
        args.candidate, action_map, device=device, simulations=args.simulations,
        c_puct=args.c_puct, seed=args.seed + 1000003,
    )
    LOGGER.info("loading baseline=%s", args.baseline)
    baseline, baseline_ckpt = load_searcher(
        args.baseline, action_map, device=device, simulations=args.simulations,
        c_puct=args.c_puct, seed=args.seed + 2000003,
    )

    started = time.perf_counter()
    for game_index in range(args.games):
        if game_index in existing:
            continue
        pair_index = game_index // 2
        opening_moves = paired_opening_moves(pair_index, args.opening_plies, args.seed)
        candidate_white = (game_index % 2 == 0)
        record, pgn_text = play_game(
            game_index=game_index,
            candidate=candidate,
            baseline=baseline,
            candidate_white=candidate_white,
            opening_moves=opening_moves,
            max_plies=args.max_plies,
            candidate_name=args.candidate_name,
            baseline_name=args.baseline_name,
        )
        record.update({
            "candidate_checkpoint": str(Path(args.candidate).resolve()),
            "baseline_checkpoint": str(Path(args.baseline).resolve()),
            "simulations": args.simulations,
            "c_puct": args.c_puct,
        })
        with pgn_path.open("a", encoding="utf-8") as fp:
            fp.write(pgn_text)
            fp.flush()
        with results_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, separators=(",", ":")) + "\n")
            fp.flush()
        existing[game_index] = record

        completed = len([i for i in existing if 0 <= i < args.games])
        scores = [float(existing[i]["candidate_score"]) for i in sorted(existing) if 0 <= i < args.games]
        running = summarize_scores(scores, bootstrap_samples=min(args.bootstrap_samples, 2000), seed=args.seed)
        LOGGER.info(
            "game=%d/%d result=%s candidate_score=%.1f running=%.3f elo_delta=%.1f",
            completed, args.games, record["result"], record["candidate_score"],
            running["candidate_score_rate"], running["elo_delta_smoothed"],
        )

    final_records = read_existing(results_path)
    selected = [final_records[i] for i in sorted(final_records) if 0 <= i < args.games]
    scores = [float(r["candidate_score"]) for r in selected]
    rating = summarize_scores(
        scores,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        baseline_rating=args.baseline_rating,
    )
    summary = {
        "candidate": str(Path(args.candidate).resolve()),
        "baseline": str(Path(args.baseline).resolve()),
        "candidate_stage": candidate_ckpt.get("stage"),
        "candidate_iteration": candidate_ckpt.get("iteration"),
        "candidate_epoch": candidate_ckpt.get("epoch"),
        "baseline_stage": baseline_ckpt.get("stage"),
        "baseline_iteration": baseline_ckpt.get("iteration"),
        "baseline_epoch": baseline_ckpt.get("epoch"),
        "device": str(device),
        "simulations": args.simulations,
        "c_puct": args.c_puct,
        "opening_plies": args.opening_plies,
        "max_plies": args.max_plies,
        "seed": args.seed,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "adjudicated_games": sum(bool(r.get("adjudicated_max_plies")) for r in selected),
        "rating": rating,
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
