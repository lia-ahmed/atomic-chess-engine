from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ratings import smoothed_elo_delta, summarize_scores  # noqa: E402
from time_control import (  # noqa: E402
    SpeedEstimator,
    TimingConfig,
    allocate_move_time_ms,
    parse_go_parameters,
    simulations_for_budget,
)


class RatingTests(unittest.TestCase):
    def test_equal_score_is_zero_elo(self):
        self.assertAlmostEqual(smoothed_elo_delta([1.0, 0.0, 0.5, 0.5]), 0.0, places=10)

    def test_winning_score_is_positive(self):
        summary = summarize_scores([1.0, 1.0, 1.0, 0.0], bootstrap_samples=1000, seed=1)
        self.assertGreater(summary["elo_delta_smoothed"], 0.0)
        self.assertEqual(summary["wins"], 3)
        self.assertEqual(summary["losses"], 1)


class TimingTests(unittest.TestCase):
    def test_go_parser(self):
        p = parse_go_parameters("wtime 180000 btime 180000 winc 2000 binc 2000".split())
        self.assertEqual(p["wtime"], 180000)
        self.assertEqual(p["winc"], 2000)

    def test_dynamic_budget_is_capped(self):
        cfg = TimingConfig(max_move_time_ms=1200, time_fraction=0.05, increment_fraction=0.5)
        p = {"wtime": 180000, "winc": 2000}
        budget = allocate_move_time_ms(side_white=True, params=p, config=cfg)
        self.assertEqual(budget, 1200)

    def test_low_clock_respects_safety(self):
        cfg = TimingConfig(reserve_ms=1000, move_overhead_ms=150, min_move_time_ms=50)
        p = {"btime": 1200, "binc": 0}
        budget = allocate_move_time_ms(side_white=False, params=p, config=cfg)
        self.assertLessEqual(budget, 50)

    def test_simulation_estimate(self):
        cfg = TimingConfig(min_simulations=20, max_simulations=400, simulation_safety=0.8)
        speed = SpeedEstimator(200.0)
        sims = simulations_for_budget(1000, speed=speed, config=cfg)
        self.assertEqual(sims, 160)


if __name__ == "__main__":
    unittest.main()
