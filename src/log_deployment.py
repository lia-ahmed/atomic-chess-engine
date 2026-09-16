#!/usr/bin/env python3
"""Append an immutable-ish deployment record for a Lichess engine checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="Record which model/config was deployed to Lichess.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--action-map-path", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", default="evaluation/lichess/deployments.jsonl")
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--max-move-time-ms", type=int, default=1200)
    p.add_argument("--time-fraction", type=float, default=0.025)
    p.add_argument("--increment-fraction", type=float, default=0.50)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    amap = Path(args.action_map_path).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    if not amap.is_file():
        raise SystemExit(f"action map not found: {amap}")

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "action_map": str(amap),
        "action_map_sha256": sha256_file(amap),
        "settings": {
            "simulations": args.simulations,
            "c_puct": args.c_puct,
            "max_move_time_ms": args.max_move_time_ms,
            "time_fraction": args.time_fraction,
            "increment_fraction": args.increment_fraction,
            "temperature": 0.0,
            "dirichlet_noise": False,
        },
        "notes": args.notes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
