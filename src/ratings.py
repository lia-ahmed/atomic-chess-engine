#!/usr/bin/env python3
"""Internal arena rating utilities for the atomic-chess project.
Note: internal elo. Not like official elo. 
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def score_rate(scores: Sequence[float]) -> float:
    if not scores:
        raise ValueError("at least one game score is required")
    arr = np.asarray(scores, dtype=np.float64)
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("game scores must lie in [0, 1]")
    return float(arr.mean())


def elo_from_probability(p: float) -> float:
    """Convert expected score probability to Elo difference.

    Returns +/- infinity at exact 0/1, matching the maximum-likelihood Elo
    transform. Use :func:`smoothed_elo_delta` for finite estimates.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0,1]")
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf
    return 400.0 * math.log10(p / (1.0 - p))


def smoothed_score(total_points: float, games: int, prior: float = 0.5) -> float:
    if games <= 0:
        raise ValueError("games must be positive")
    if prior < 0:
        raise ValueError("prior must be non-negative")
    if total_points < 0 or total_points > games:
        raise ValueError("total_points outside [0, games]")
    if prior == 0:
        return total_points / games
    return (total_points + prior) / (games + 2.0 * prior)


def smoothed_elo_delta(scores: Sequence[float], prior: float = 0.5) -> float:
    if not scores:
        raise ValueError("at least one game score is required")
    total = float(np.sum(np.asarray(scores, dtype=np.float64)))
    p = smoothed_score(total, len(scores), prior=prior)
    return elo_from_probability(p)


def bootstrap_elo_interval(
    scores: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 10000,
    seed: int = 42,
    prior: float = 0.5,
) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not scores:
        raise ValueError("at least one game score is required")

    arr = np.asarray(scores, dtype=np.float64)
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("game scores must lie in [0,1]")

    rng = np.random.default_rng(seed)
    n = len(arr)
    estimates = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        sample = arr[rng.integers(0, n, size=n)]
        p = smoothed_score(float(sample.sum()), n, prior=prior)
        estimates[i] = elo_from_probability(p)

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def summarize_scores(
    scores: Sequence[float],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10000,
    seed: int = 42,
    prior: float = 0.5,
    baseline_rating: float | None = None,
) -> dict[str, object]:
    if not scores:
        raise ValueError("no completed games")

    wins = sum(float(s) == 1.0 for s in scores)
    draws = sum(float(s) == 0.5 for s in scores)
    losses = sum(float(s) == 0.0 for s in scores)
    n = len(scores)
    total_points = float(sum(scores))
    raw_p = total_points / n
    raw_elo = elo_from_probability(raw_p)
    smooth_elo = smoothed_elo_delta(scores, prior=prior)
    ci_low, ci_high = bootstrap_elo_interval(
        scores,
        confidence=confidence,
        samples=bootstrap_samples,
        seed=seed,
        prior=prior,
    )

    result: dict[str, object] = {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "candidate_points": total_points,
        "candidate_score_rate": raw_p,
        "elo_delta_mle": None if not math.isfinite(raw_elo) else raw_elo,
        "elo_delta_smoothed": smooth_elo,
        "elo_confidence": confidence,
        "elo_ci_low": ci_low,
        "elo_ci_high": ci_high,
        "bootstrap_samples": bootstrap_samples,
        "elo_prior_each_side": prior,
    }
    if baseline_rating is not None:
        result["baseline_rating"] = float(baseline_rating)
        result["candidate_rating_estimate"] = float(baseline_rating + smooth_elo)
        result["candidate_rating_ci_low"] = float(baseline_rating + ci_low)
        result["candidate_rating_ci_high"] = float(baseline_rating + ci_high)
    return result


def read_arena_scores(paths: Iterable[str | Path]) -> list[float]:
    scores: list[float] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
                if "candidate_score" not in record:
                    continue
                scores.append(float(record["candidate_score"]))
    return scores


def main() -> int:
    p = argparse.ArgumentParser(description="Estimate internal Elo delta from arena JSONL games.")
    p.add_argument("results", nargs="+", help="arena game-results JSONL file(s)")
    p.add_argument("--baseline-rating", type=float, default=None)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prior", type=float, default=0.5)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    scores = read_arena_scores(args.results)
    summary = summarize_scores(
        scores,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        prior=args.prior,
        baseline_rating=args.baseline_rating,
    )
    print(json.dumps(summary, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
