#!/usr/bin/env python3
"""Generate atomic-chess self-play data with PUCT MCTS.

Each output JSONL line is one training position containing:
- FEN before the played move
- sparse MCTS visit distribution over the Phase 2 action map
- final game outcome from the side-to-move perspective
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
import chess.variant

from mcts import MCTS
from models import action_map_digest, load_policy_value_checkpoint
from representations import load_action_map

LOGGER = logging.getLogger("self_play")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def white_result(board: chess.variant.AtomicBoard) -> float:
    """Return +1 White win, -1 Black win, 0 draw/unfinished adjudication."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == chess.WHITE else -1.0


def sparse_policy_target(
    visit_counts: Mapping[str, int],
    action_map: Mapping[str, int],
) -> tuple[list[int], list[float], float]:
    """Convert root visits to sparse action-index probabilities.

    Returns '(indices, probs, dropped_mass)'. Moves absent from the observed action vocabulary cannot be represented in the policy head and are dropped, then the remaining probabilities are renormalized.
    """
    positive = {uci: int(n) for uci, n in visit_counts.items() if int(n) > 0}
    total = sum(positive.values())
    if total <= 0:
        return [], [], 0.0

    kept: list[tuple[int, int]] = []
    dropped = 0
    for uci, count in positive.items():
        idx = action_map.get(uci)
        if idx is None:
            dropped += count
        else:
            kept.append((int(idx), count))

    kept_total = sum(count for _, count in kept)
    if kept_total <= 0:
        return [], [], 1.0
    indices = [idx for idx, _ in kept]
    probs = [count / kept_total for _, count in kept]
    return indices, probs, dropped / total


def play_one_game(
    searcher: MCTS,
    action_map: Mapping[str, int],
    *,
    game_id: str,
    iteration: int,
    temperature: float,
    temperature_moves: int,
    max_plies: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    board = chess.variant.AtomicBoard()
    pending: list[dict[str, Any]] = []
    unknown_selected = 0
    dropped_mass_sum = 0.0

    for ply in range(max_plies):
        if board.outcome(claim_draw=True) is not None:
            break

        fen = board.fen()
        turn_white = board.turn == chess.WHITE
        move_temperature = temperature if ply < temperature_moves else 0.0
        result = searcher.search(
            board,
            temperature=move_temperature,
            add_root_noise=searcher.dirichlet_epsilon > 0.0,
        )
        policy_indices, policy_probs, dropped_mass = sparse_policy_target(
            result.visit_counts, action_map
        )
        dropped_mass_sum += dropped_mass

        played_idx = action_map.get(result.move_uci)
        if played_idx is None:
            unknown_selected += 1

        if policy_indices:
            pending.append(
                {
                    "game_id": game_id,
                    "iteration": int(iteration),
                    "ply": int(ply),
                    "fen": fen,
                    "side_to_move": 0 if turn_white else 1,
                    "policy_indices": policy_indices,
                    "policy_probs": policy_probs,
                    "played_uci": result.move_uci,
                    "played_action_index": None if played_idx is None else int(played_idx),
                    "root_value": float(result.root_value),
                    "simulations": int(result.simulations),
                }
            )

        move = chess.Move.from_uci(result.move_uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"MCTS returned illegal move {result.move_uci} from {fen}")
        board.push(move)

    result_white = white_result(board)
    for sample in pending:
        sample["value_target"] = (
            result_white if sample["side_to_move"] == 0 else -result_white
        )

    summary = {
        "game_id": game_id,
        "plies": int(board.ply()),
        "samples": len(pending),
        "result_white": result_white,
        "terminal": board.outcome(claim_draw=True) is not None,
        "unknown_selected_moves": unknown_selected,
        "mean_dropped_policy_mass": dropped_mass_sum / max(1, board.ply()),
        "final_fen": board.fen(),
    }
    return pending, summary


def save_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate atomic-chess self-play training data with MCTS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--net", required=True, help="Phase 4 policy+value checkpoint")
    p.add_argument("--action-map-path", "--action-map", dest="action_map_path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--games", type=int, default=100, help="Games generated by this worker")
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-moves", type=int, default=20)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    p.add_argument("--unknown-move-prior-mass", type=float, default=0.0)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.games <= 0 or args.simulations <= 0 or args.max_plies <= 0:
        raise SystemExit("games, simulations and max-plies must be positive")
    if args.worker_id < 0 or args.num_workers <= 0 or args.worker_id >= args.num_workers:
        raise SystemExit("require 0 <= worker-id < num-workers")

    random.seed(args.seed + args.worker_id)
    np.random.seed(args.seed + args.worker_id)
    torch.manual_seed(args.seed + args.worker_id)

    device = resolve_device(args.device)
    action_map = load_action_map(args.action_map_path)
    model, checkpoint = load_policy_value_checkpoint(args.net, device=device)
    expected_digest = checkpoint.get("action_map_sha256")
    actual_digest = action_map_digest(action_map)
    if expected_digest is not None and expected_digest != actual_digest:
        raise SystemExit("checkpoint action map does not match --action-map-path")
    model.action_map = action_map

    searcher = MCTS(
        model,
        action_map,
        num_simulations=args.simulations,
        c_puct=args.c_puct,
        device=device,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        unknown_move_prior_mass=args.unknown_move_prior_mass,
        seed=args.seed + args.worker_id,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"selfplay_iter{args.iteration:03d}_worker{args.worker_id:03d}"
    out_path = out_dir / f"{stem}.jsonl"
    summary_path = out_dir / f"{stem}.summary.json"
    if (out_path.exists() or summary_path.exists()) and not args.overwrite:
        raise SystemExit(f"output exists for worker {args.worker_id}; use --overwrite")
    if args.overwrite:
        for path in (out_path, summary_path):
            if path.exists():
                path.unlink()

    LOGGER.info(
        "device=%s games=%d simulations=%d checkpoint=%s actions=%d",
        device,
        args.games,
        args.simulations,
        args.net,
        len(action_map),
    )
    started = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    total_samples = 0

    for local_game in range(args.games):
        global_game = args.worker_id + local_game * args.num_workers
        game_id = f"i{args.iteration:03d}_w{args.worker_id:03d}_g{global_game:08d}"
        records, summary = play_one_game(
            searcher,
            action_map,
            game_id=game_id,
            iteration=args.iteration,
            temperature=args.temperature,
            temperature_moves=args.temperature_moves,
            max_plies=args.max_plies,
        )
        save_jsonl_records(out_path, records)
        summaries.append(summary)
        total_samples += len(records)

        if args.log_every > 0 and (local_game + 1) % args.log_every == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
            LOGGER.info(
                "games=%d/%d samples=%d games_per_hour=%.1f",
                local_game + 1,
                args.games,
                total_samples,
                (local_game + 1) * 3600.0 / elapsed,
            )

    elapsed = time.perf_counter() - started
    manifest = {
        "phase": 4,
        "iteration": args.iteration,
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "checkpoint": str(Path(args.net).resolve()),
        "action_map": str(Path(args.action_map_path).resolve()),
        "games": len(summaries),
        "samples": total_samples,
        "simulations": args.simulations,
        "c_puct": args.c_puct,
        "temperature": args.temperature,
        "temperature_moves": args.temperature_moves,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_epsilon": args.dirichlet_epsilon,
        "max_plies": args.max_plies,
        "seconds": elapsed,
        "white_wins": sum(s["result_white"] > 0 for s in summaries),
        "black_wins": sum(s["result_white"] < 0 for s in summaries),
        "draws_or_adjudicated": sum(s["result_white"] == 0 for s in summaries),
        "unknown_selected_moves": sum(s["unknown_selected_moves"] for s in summaries),
        "mean_game_plies": sum(s["plies"] for s in summaries) / max(1, len(summaries)),
        "games_detail": summaries,
    }
    tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
        fp.write("\n")
    os.replace(tmp, summary_path)

    print(
        f"games={len(summaries)} samples={total_samples} seconds={elapsed:.1f} "
        f"out={out_path} summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
