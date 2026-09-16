#!/usr/bin/env python3
"""Create the initial policy+value checkpoint from best.pt.

The trunk and policy head are copied exactly. 
The new value head is initialized to produce 0.0 for every position, giving the first MCTS/self-play iteration a neutral value prior rather than random evaluations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from models import PolicyValueNet, action_map_digest, transfer_supervised_weights
from representations import load_action_map


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Warm-start a policy+value network from prev.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--supervised-ckpt", required=True, help="best.pt")
    p.add_argument("--action-map-path", "--action-map", dest="action_map_path", required=True)
    p.add_argument("--out", default="ckpts/phase4/init.pt")
    p.add_argument("--value-channels", type=int, default=32)
    p.add_argument("--value-hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {out}; use --overwrite to replace it")

    torch.manual_seed(args.seed)
    action_map = load_action_map(args.action_map_path)
    supervised = torch.load(args.supervised_ckpt, map_location="cpu")
    cfg = supervised.get("model_config")
    if not isinstance(cfg, dict):
        raise SystemExit("Checkpoint is missing model_config")
    if int(cfg.get("num_actions", -1)) != len(action_map):
        raise SystemExit(
            f"action count mismatch: checkpoint={cfg.get('num_actions')} action_map={len(action_map)}"
        )

    model = PolicyValueNet(
        num_actions=len(action_map),
        in_channels=int(cfg.get("in_channels", 14)),
        width=int(cfg.get("width", 64)),
        blocks=int(cfg.get("blocks", 6)),
        policy_channels=int(cfg.get("policy_channels", 32)),
        value_channels=args.value_channels,
        value_hidden=args.value_hidden,
        dropout=float(cfg.get("dropout", 0.0)),
    )
    transfer_supervised_weights(model, supervised, zero_value=True)

    payload = {
        "phase": 4,
        "stage": "initialized_from_supervised",
        "iteration": 0,
        "epoch": 0,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "action_map_sha256": action_map_digest(action_map),
        "action_map_path": str(Path(args.action_map_path).resolve()),
        "source_supervised_checkpoint": str(Path(args.supervised_ckpt).resolve()),
        "source_supervised_epoch": supervised.get("epoch"),
        "source_supervised_val_metrics": supervised.get("val_metrics"),
        "value_initialization": "zero_output",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    print(
        f"created={out} actions={len(action_map)} "
        f"source_epoch={supervised.get('epoch')} value_init=zero"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
