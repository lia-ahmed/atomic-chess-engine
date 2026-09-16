#!/usr/bin/env python3
"""Validate/count completed self-play JSONL shards before training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--expected-games", type=int, default=None)
    args = p.parse_args()

    root = Path(args.data_dir)
    files = sorted(root.rglob("selfplay_*.jsonl"))
    if not files:
        raise SystemExit(f"no selfplay_*.jsonl under {root}")

    samples = 0
    game_files: dict[str, set[str]] = defaultdict(set)
    game_samples: dict[str, int] = defaultdict(int)
    missing_value = 0
    malformed_policy = 0

    for path in files:
        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"invalid JSON {path}:{line_no}: {exc}") from exc
                gid = str(rec.get("game_id", ""))
                if not gid:
                    raise SystemExit(f"missing game_id {path}:{line_no}")
                game_files[gid].add(str(path.resolve()))
                game_samples[gid] += 1
                samples += 1
                if "value_target" not in rec:
                    missing_value += 1
                inds = rec.get("policy_indices", [])
                probs = rec.get("policy_probs", [])
                if not inds or len(inds) != len(probs):
                    malformed_policy += 1

    cross_file_duplicates = {g: sorted(v) for g, v in game_files.items() if len(v) > 1}
    games = len(game_files)
    print(f"FILES={len(files)}")
    print(f"GAMES={games}")
    print(f"SAMPLES={samples}")
    print(f"MISSING_VALUE_TARGET={missing_value}")
    print(f"MALFORMED_POLICY_TARGET={malformed_policy}")
    print(f"CROSS_FILE_DUPLICATE_GAMES={len(cross_file_duplicates)}")
    if cross_file_duplicates:
        for gid, paths in list(cross_file_duplicates.items())[:10]:
            print(f"DUPLICATE {gid}: {' | '.join(paths)}")

    ok = missing_value == 0 and malformed_policy == 0 and not cross_file_duplicates
    if args.expected_games is not None:
        print(f"EXPECTED_GAMES={args.expected_games}")
        if games != args.expected_games:
            ok = False
    print(f"SELFPLAY_VALID={'YES' if ok else 'NO'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
