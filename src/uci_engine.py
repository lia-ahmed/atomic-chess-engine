#!/usr/bin/env python3
"""UCI wrapper for the Atomic PolicyValueNet + PUCT MCTS engine.

Designed for local GUI use and the maintained lichess-bot bridge. Rated play uses zero-temperature move selection and no Dirichlet noise for determinism

Timing is simple: clock/movetime information is converted to a conservative think-time budget, then to an MCTS simulation budget using an EMA of observed simulations/second. 
Because the current MCTS is not interruptible, this is a soft timing target rather than a hard deadline.
"""


from __future__ import annotations
import argparse
import logging
import shlex
import sys
import time
from pathlib import Path

import torch
import chess
import chess.variant

from mcts import MCTS
from models import action_map_digest, load_policy_value_checkpoint
from representations import load_action_map
from time_control import (
    SpeedEstimator,
    TimingConfig,
    allocate_move_time_ms,
    parse_go_parameters,
    simulations_for_budget,
)

LOGGER = logging.getLogger("uci_engine")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def parse_setoption(line: str) -> tuple[str, str]:
    tokens = shlex.split(line, posix=False)
    lower = [t.lower() for t in tokens]
    if len(tokens) < 3 or lower[0] != "setoption" or "name" not in lower:
        return "", ""
    name_i = lower.index("name") + 1
    value_i = lower.index("value") if "value" in lower else len(tokens)
    name = " ".join(tokens[name_i:value_i]).strip()
    value = " ".join(tokens[value_i + 1:]).strip() if value_i < len(tokens) else ""
    return name, value


class AtomicUCIEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = resolve_device(args.device)
        self.action_map = load_action_map(args.action_map_path)
        self.model, self.checkpoint = load_policy_value_checkpoint(args.checkpoint, device=self.device)
        expected = self.checkpoint.get("action_map_sha256")
        actual = action_map_digest(self.action_map)
        if expected not in (None, actual):
            raise RuntimeError("checkpoint action map does not match action_map.json")
        if int(self.model.config.num_actions) != len(self.action_map):
            raise RuntimeError("network action count does not match action_map.json")
        self.model.action_map = self.action_map

        self.timing = TimingConfig(
            base_simulations=args.simulations,
            min_simulations=args.min_simulations,
            max_simulations=args.max_simulations,
            fixed_move_time_ms=args.fixed_move_time_ms,
            min_move_time_ms=args.min_move_time_ms,
            max_move_time_ms=args.max_move_time_ms,
            move_overhead_ms=args.move_overhead_ms,
            reserve_ms=args.reserve_ms,
            time_fraction=args.time_fraction,
            increment_fraction=args.increment_fraction,
            simulation_safety=args.simulation_safety,
        )
        self.timing.validate()
        self.speed = SpeedEstimator(args.initial_sims_per_second, args.speed_ema_alpha)
        self.searcher = MCTS(
            self.model,
            self.action_map,
            num_simulations=args.simulations,
            c_puct=args.c_puct,
            device=self.device,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            unknown_move_prior_mass=0.0,
            seed=args.seed,
        )
        self.board = chess.variant.AtomicBoard()

    def send(self, text: str) -> None:
        print(text, flush=True)

    def uci_handshake(self) -> None:
        self.send("id name AtomicPolicyValueMCTS")
        self.send("id author AtomicChess project")
        self.send("option name UCI_Variant type combo default atomic var atomic")
        self.send(f"option name Simulations type spin default {self.timing.base_simulations} min 1 max 100000")
        self.send(f"option name MinSimulations type spin default {self.timing.min_simulations} min 1 max 100000")
        self.send(f"option name MaxSimulations type spin default {self.timing.max_simulations} min 1 max 100000")
        self.send(f"option name MaxMoveTimeMs type spin default {self.timing.max_move_time_ms} min 1 max 600000")
        self.send(f"option name MoveOverheadMs type spin default {self.timing.move_overhead_ms} min 0 max 10000")
        self.send(f"option name ReserveMs type spin default {self.timing.reserve_ms} min 0 max 60000")
        self.send(f"option name TimeFraction type string default {self.timing.time_fraction}")
        self.send(f"option name IncrementFraction type string default {self.timing.increment_fraction}")
        self.send(f"option name CPuct type string default {self.searcher.c_puct}")
        self.send("uciok")

    def set_option(self, name: str, value: str) -> None:
        key = name.strip().lower().replace(" ", "")
        if key == "uci_variant":
            if value.strip().lower() != "atomic":
                self.send(f"info string unsupported variant {value}; this engine is atomic-only")
            return
        try:
            if key == "simulations":
                self.timing.base_simulations = int(value)
            elif key == "minsimulations":
                self.timing.min_simulations = int(value)
            elif key == "maxsimulations":
                self.timing.max_simulations = int(value)
            elif key == "maxmovetimems":
                self.timing.max_move_time_ms = int(value)
            elif key == "moveoverheadms":
                self.timing.move_overhead_ms = int(value)
            elif key == "reservems":
                self.timing.reserve_ms = int(value)
            elif key == "timefraction":
                self.timing.time_fraction = float(value)
            elif key == "incrementfraction":
                self.timing.increment_fraction = float(value)
            elif key == "cpuct":
                self.searcher.c_puct = float(value)
            else:
                LOGGER.warning("ignored unknown UCI option %s=%s", name, value)
                return
            self.timing.validate()
        except (TypeError, ValueError) as exc:
            self.send(f"info string invalid option {name}: {exc}")

    def set_position(self, line: str) -> None:
        tokens = line.strip().split()
        if len(tokens) < 2:
            raise ValueError("position command missing position")
        i = 1
        if tokens[i] == "startpos":
            board = chess.variant.AtomicBoard()
            i += 1
        elif tokens[i] == "fen":
            i += 1
            fen_parts: list[str] = []
            while i < len(tokens) and tokens[i] != "moves":
                fen_parts.append(tokens[i])
                i += 1
            if len(fen_parts) < 4:
                raise ValueError("position fen command is incomplete")
            board = chess.variant.AtomicBoard(" ".join(fen_parts))
        else:
            raise ValueError("position must use startpos or fen")

        if i < len(tokens) and tokens[i] == "moves":
            i += 1
            for uci in tokens[i:]:
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    raise ValueError(f"illegal atomic move in position command: {uci}")
                board.push(move)
        self.board = board

    def go(self, line: str) -> None:
        if self.board.outcome(claim_draw=True) is not None:
            self.send("bestmove 0000")
            return

        params = parse_go_parameters(line.split()[1:])
        budget_ms = allocate_move_time_ms(
            side_white=self.board.turn == chess.WHITE,
            params=params,
            config=self.timing,
        )
        explicit_nodes = int(params["nodes"]) if "nodes" in params else None
        simulations = simulations_for_budget(
            budget_ms,
            speed=self.speed,
            config=self.timing,
            explicit_nodes=explicit_nodes,
        )
        self.searcher.num_simulations = simulations

        started = time.perf_counter()
        result = self.searcher.search(self.board, temperature=0.0, add_root_noise=False)
        elapsed = max(time.perf_counter() - started, 1e-9)
        self.speed.update(result.simulations, elapsed)

        self.send(
            "info string "
            f"budget_ms={budget_ms if budget_ms is not None else 'none'} "
            f"sims={result.simulations} elapsed_ms={elapsed * 1000.0:.1f} "
            f"ema_sps={self.speed.sims_per_second:.1f}"
        )
        self.send(f"bestmove {result.move_uci}")

    def loop(self) -> int:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            LOGGER.debug("UCI << %s", line)
            command = line.split(maxsplit=1)[0].lower()
            try:
                if command == "uci":
                    self.uci_handshake()
                elif command == "isready":
                    self.send("readyok")
                elif command == "setoption":
                    name, value = parse_setoption(line)
                    if name:
                        self.set_option(name, value)
                elif command == "ucinewgame":
                    self.board = chess.variant.AtomicBoard()
                elif command == "position":
                    self.set_position(line)
                elif command == "go":
                    self.go(line)
                elif command == "stop":
                    # Current MCTS is synchronous/non-interruptible. Normally this
                    # command is only read after search has already returned.
                    pass
                elif command == "ponderhit":
                    pass
                elif command == "quit":
                    return 0
                elif command == "debug":
                    pass
                else:
                    LOGGER.debug("ignored UCI command: %s", line)
            except Exception as exc:
                LOGGER.exception("UCI command failed: %s", line)
                self.send(f"info string ERROR {type(exc).__name__}: {exc}")
                if command == "go":
                    self.send("bestmove 0000")
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--action-map-path", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--min-simulations", type=int, default=20)
    p.add_argument("--max-simulations", type=int, default=400)
    p.add_argument("--fixed-move-time-ms", type=int, default=0,
                   help="0 uses clock-aware allocation; positive forces a target per move")
    p.add_argument("--min-move-time-ms", type=int, default=50)
    p.add_argument("--max-move-time-ms", type=int, default=1200)
    p.add_argument("--move-overhead-ms", type=int, default=150)
    p.add_argument("--reserve-ms", type=int, default=1000)
    p.add_argument("--time-fraction", type=float, default=0.025)
    p.add_argument("--increment-fraction", type=float, default=0.50)
    p.add_argument("--simulation-safety", type=float, default=0.80)
    p.add_argument("--initial-sims-per-second", type=float, default=180.0)
    p.add_argument("--speed-ema-alpha", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-file", default=None)
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return p


def main() -> int:
    args = build_parser().parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    engine = AtomicUCIEngine(args)
    LOGGER.info(
        "loaded checkpoint=%s device=%s actions=%d stage=%s iteration=%s epoch=%s",
        args.checkpoint,
        engine.device,
        len(engine.action_map),
        engine.checkpoint.get("stage"),
        engine.checkpoint.get("iteration"),
        engine.checkpoint.get("epoch"),
    )
    return engine.loop()


if __name__ == "__main__":
    raise SystemExit(main())
