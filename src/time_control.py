#!/usr/bin/env python3
"""Small, dependency-free time manager used by the UCI/Lichess wrapper.
This is decidedly half-assed and vibe coded. Should come back to rigorously in future.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimingConfig:
    base_simulations: int = 200
    min_simulations: int = 20
    max_simulations: int = 400
    fixed_move_time_ms: int = 0
    min_move_time_ms: int = 50
    max_move_time_ms: int = 1200
    move_overhead_ms: int = 150
    reserve_ms: int = 1000
    time_fraction: float = 0.025
    increment_fraction: float = 0.50
    simulation_safety: float = 0.80

    def validate(self) -> None:
        if self.base_simulations <= 0:
            raise ValueError("base_simulations must be positive")
        if self.min_simulations <= 0 or self.max_simulations < self.min_simulations:
            raise ValueError("invalid simulation bounds")
        if self.fixed_move_time_ms < 0:
            raise ValueError("fixed_move_time_ms must be non-negative")
        if self.min_move_time_ms < 0 or self.max_move_time_ms <= 0:
            raise ValueError("invalid move-time bounds")
        if self.max_move_time_ms < self.min_move_time_ms:
            raise ValueError("max_move_time_ms must be >= min_move_time_ms")
        if self.move_overhead_ms < 0 or self.reserve_ms < 0:
            raise ValueError("overhead/reserve must be non-negative")
        if not 0.0 <= self.time_fraction <= 1.0:
            raise ValueError("time_fraction must lie in [0,1]")
        if not 0.0 <= self.increment_fraction <= 2.0:
            raise ValueError("increment_fraction outside sensible range")
        if not 0.0 < self.simulation_safety <= 1.0:
            raise ValueError("simulation_safety must lie in (0,1]")


class SpeedEstimator:
    def __init__(self, initial_sims_per_second: float = 180.0, alpha: float = 0.20) -> None:
        if initial_sims_per_second <= 0:
            raise ValueError("initial_sims_per_second must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0,1]")
        self.sims_per_second = float(initial_sims_per_second)
        self.alpha = float(alpha)

    def update(self, simulations: int, elapsed_seconds: float) -> None:
        if simulations <= 0 or elapsed_seconds <= 0:
            return
        observed = simulations / elapsed_seconds
        self.sims_per_second = (
            self.alpha * observed + (1.0 - self.alpha) * self.sims_per_second
        )


def parse_go_parameters(tokens: list[str]) -> dict[str, int | bool]:
    """Parse the simple numeric UCI ``go`` parameters used by lichess-bot."""
    params: dict[str, int | bool] = {}
    numeric = {"wtime", "btime", "winc", "binc", "movetime", "nodes", "depth", "movestogo"}
    flags = {"infinite", "ponder"}
    i = 0
    while i < len(tokens):
        key = tokens[i].lower()
        if key in flags:
            params[key] = True
            i += 1
            continue
        if key in numeric and i + 1 < len(tokens):
            try:
                params[key] = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        # Ignore unsupported UCI keywords, including searchmoves.
        i += 1
    return params


def allocate_move_time_ms(
    *,
    side_white: bool,
    params: dict[str, int | bool],
    config: TimingConfig,
) -> int | None:
    """Return a conservative target think-time in milliseconds.

    'None' means no clock information was supplied, in which case the UCI
    engine should fall back to 'base_simulations'.
    """
    config.validate()

    if "movetime" in params:
        requested = max(1, int(params["movetime"]))
        safe = max(1, requested - config.move_overhead_ms)
        return min(safe, config.max_move_time_ms)

    clock_key = "wtime" if side_white else "btime"
    inc_key = "winc" if side_white else "binc"
    if clock_key not in params:
        if config.fixed_move_time_ms > 0:
            return min(config.fixed_move_time_ms, config.max_move_time_ms)
        return None

    remaining = max(0, int(params[clock_key]))
    increment = max(0, int(params.get(inc_key, 0)))

    # Never deliberately consume the reserve/transport margin.
    hard_safe = max(1, remaining - config.reserve_ms - config.move_overhead_ms)

    if config.fixed_move_time_ms > 0:
        target = config.fixed_move_time_ms
    else:
        target = int(
            remaining * config.time_fraction
            + increment * config.increment_fraction
        )

    target = min(target, config.max_move_time_ms, hard_safe)
    if hard_safe >= config.min_move_time_ms:
        target = max(target, config.min_move_time_ms)
    else:
        target = hard_safe
    return max(1, int(target))


def simulations_for_budget(
    budget_ms: int | None,
    *,
    speed: SpeedEstimator,
    config: TimingConfig,
    explicit_nodes: int | None = None,
) -> int:
    config.validate()
    if explicit_nodes is not None and explicit_nodes > 0:
        return max(1, min(int(explicit_nodes), config.max_simulations))
    if budget_ms is None:
        return max(config.min_simulations, min(config.base_simulations, config.max_simulations))

    estimated = int(
        speed.sims_per_second
        * (budget_ms / 1000.0)
        * config.simulation_safety
    )
    estimated = max(1, estimated)

    # For extremely low clocks, do not force the normal minimum simulation count.
    if budget_ms < config.min_move_time_ms:
        return min(estimated, config.max_simulations)
    return max(config.min_simulations, min(estimated, config.max_simulations))
