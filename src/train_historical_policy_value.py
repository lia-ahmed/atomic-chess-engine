#!/usr/bin/env python3
"""Historical policy+value warm-start trainer.


We reuse the Parquet examples and trains the PolicyValueNet against two historical targets for every position:

    policy target = the next move actually played in the historical game
    value target  = the final game result from the side-to-move perspective

The historical value target is a warm start, NOT a claim of perfect minimax value. 
Self-play/MCTS training hence follows this stage to gradually replace/refine that signal.

example usage:

    python src/train_historical_policy_value.py \
      --data-dir data \
      --action-map-path action_map.json \
      --init-checkpoint ckpts/phase4/init.pt \
      --ckpt-dir ckpts/phase4/historical_pv \
      --epochs 3 --batch-size 256 --lr 0.0003 \
      --num-workers 4 --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from dataset import AtomicChessDataset, RowGroupShuffleSampler
from models import action_map_digest, count_parameters, load_policy_value_checkpoint
from representations import load_action_map
from historical_pv import combined_loss, historical_value_target

LOGGER = logging.getLogger("train_historical_policy_value")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device



class HistoricalPolicyValueDataset(Dataset):
    """Adapt AtomicChessDataset to return (state, action, value_target).

    The wrapped base dataset already performs the critical Phase 1 -> Phase 3
    pairing: state after ply N is paired with the move played at ply N+1. The
    metadata on that state also contains the final game result and side to move.
    """

    def __init__(self, base: AtomicChessDataset) -> None:
        if not getattr(base, "include_metadata", False):
            raise ValueError("HistoricalPolicyValueDataset requires include_metadata=True")
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        state, action, metadata = self.base[index]
        value = historical_value_target(metadata["result"], metadata["side_to_move"])
        return state, action, torch.tensor(value, dtype=torch.float32)


def _close_dataset(dataset: AtomicChessDataset) -> None:
    """Close Parquet handles on both old and Windows-fixed dataset.py versions."""
    close = getattr(dataset, "close", None)
    if callable(close):
        close()
        return

    cache = getattr(dataset, "_row_group_cache", None)
    if cache is not None:
        cache.clear()
    handles = getattr(dataset, "_parquet_files", None)
    if handles is not None:
        for pf in list(handles.values()):
            try:
                pf.close()
            except Exception:
                pass
        handles.clear()


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)



def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    scaler,
    amp_enabled: bool,
    policy_weight: float,
    value_weight: float,
    max_batches: Optional[int],
    log_every: int,
    epoch: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    started = time.perf_counter()

    total_examples = 0
    sums = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "policy_top1": 0.0,
        "policy_top3": 0.0,
        "value_abs_error": 0.0,
    }
    sign_correct = 0
    sign_total = 0
    value_target_sum = 0.0
    value_prediction_sum = 0.0

    grad_context = contextlib.nullcontext() if training else torch.no_grad()
    with grad_context:
        for batch_idx, (states, actions, values_target) in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            states = states.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            values_target = values_target.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, amp_enabled):
                logits, values_pred = model(states)
                loss, policy_loss, value_loss = combined_loss(
                    logits,
                    values_pred,
                    actions,
                    values_target,
                    policy_weight=policy_weight,
                    value_weight=value_weight,
                )

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            n = int(actions.shape[0])
            top_k = min(3, int(logits.shape[1]))
            predictions = logits.detach().topk(top_k, dim=1).indices
            top1 = int((predictions[:, 0] == actions).sum().item())
            top3 = int((predictions == actions[:, None]).any(dim=1).sum().item())

            detached_values = values_pred.detach()
            sums["loss"] += float(loss.detach().item()) * n
            sums["policy_loss"] += float(policy_loss.detach().item()) * n
            sums["value_loss"] += float(value_loss.detach().item()) * n
            sums["policy_top1"] += top1
            sums["policy_top3"] += top3
            sums["value_abs_error"] += float((detached_values - values_target).abs().sum().item())
            value_target_sum += float(values_target.sum().item())
            value_prediction_sum += float(detached_values.sum().item())

            decisive = values_target != 0
            if decisive.any():
                sign_correct += int(
                    (torch.sign(detached_values[decisive]) == torch.sign(values_target[decisive]))
                    .sum()
                    .item()
                )
                sign_total += int(decisive.sum().item())

            total_examples += n

            if training and log_every > 0 and batch_idx % log_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                LOGGER.info(
                    "epoch=%d batch=%d examples=%d loss=%.4f policy=%.4f value=%.4f "
                    "top1=%.4f sign=%.4f ex_per_s=%.1f",
                    epoch,
                    batch_idx,
                    total_examples,
                    sums["loss"] / total_examples,
                    sums["policy_loss"] / total_examples,
                    sums["value_loss"] / total_examples,
                    sums["policy_top1"] / total_examples,
                    sign_correct / sign_total if sign_total else 0.0,
                    total_examples / elapsed,
                )

    if total_examples == 0:
        raise RuntimeError("loader produced no examples")

    return {
        "loss": sums["loss"] / total_examples,
        "policy_loss": sums["policy_loss"] / total_examples,
        "value_loss": sums["value_loss"] / total_examples,
        "policy_top1": sums["policy_top1"] / total_examples,
        "policy_top3": sums["policy_top3"] / total_examples,
        "value_mae": sums["value_abs_error"] / total_examples,
        "value_sign_accuracy": sign_correct / sign_total if sign_total else 0.0,
        "mean_value_target": value_target_sum / total_examples,
        "mean_value_prediction": value_prediction_sum / total_examples,
        "examples": float(total_examples),
        "seconds": time.perf_counter() - started,
    }


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Preferred Phase 4 warm start: train policy+value jointly on historical "
            "per-ply data before large-scale MCTS self-play."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True, help="per_ply.parquet or directory containing it")
    p.add_argument("--action-map-path", "--action-map", dest="action_map_path", required=True)
    p.add_argument(
        "--init-checkpoint",
        required=True,
        help="Phase 4 init.pt created from the Phase 3 best.pt policy checkpoint",
    )
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--policy-weight", type=float, default=1.0)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--row-group-cache-size", type=int, default=4)
    p.add_argument("--index-path", default=None)
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--shuffle-mode", choices=("grouped", "global", "none"), default="grouped")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    amp = p.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-train-batches", type=int, default=None, help="smoke-test limiter")
    p.add_argument("--max-val-batches", type=int, default=None, help="smoke-test limiter")
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise SystemExit("epochs, batch-size and lr must be positive")
    if args.policy_weight < 0 or args.value_weight < 0:
        raise SystemExit("policy-weight and value-weight must be non-negative")
    if args.policy_weight == 0 and args.value_weight == 0:
        raise SystemExit("at least one of policy-weight/value-weight must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_enabled = (device.type == "cuda") if args.amp is None else bool(args.amp)
    if amp_enabled and device.type != "cuda":
        raise SystemExit("AMP is only supported on CUDA in this script")

    action_map = load_action_map(args.action_map_path)
    digest = action_map_digest(action_map)
    source = args.resume if args.resume else args.init_checkpoint
    model, source_checkpoint = load_policy_value_checkpoint(source, device=device)

    if int(model.config.num_actions) != len(action_map):
        raise SystemExit("network action count does not match action_map.json")
    if source_checkpoint.get("action_map_sha256") not in (None, digest):
        raise SystemExit("checkpoint action map does not match action_map.json")

    common = dict(
        data_path=args.data_dir,
        action_map=action_map,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        index_path=args.index_path,
        row_group_cache_size=args.row_group_cache_size,
        include_metadata=True,
    )
    train_base = AtomicChessDataset(split="train", rebuild_index=args.rebuild_index, **common)
    val_base = AtomicChessDataset(split="val", rebuild_index=False, **common)
    train_ds = HistoricalPolicyValueDataset(train_base)
    val_ds = HistoricalPolicyValueDataset(val_base)

    if len(train_ds) == 0 or len(val_ds) == 0:
        _close_dataset(train_base)
        _close_dataset(val_base)
        raise SystemExit(f"empty split: train={len(train_ds)} val={len(val_ds)}")

    train_sampler = None
    shuffle = False
    if args.shuffle_mode == "grouped":
        train_sampler = RowGroupShuffleSampler(train_base, seed=args.seed)
    elif args.shuffle_mode == "global":
        shuffle = True

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        shuffle=shuffle,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = make_grad_scaler(amp_enabled)

    start_epoch = 1
    best_val_loss = math.inf
    if args.resume:
        if source_checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(source_checkpoint["optimizer_state"])
        if source_checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(source_checkpoint["scheduler_state"])
        if source_checkpoint.get("scaler_state"):
            scaler.load_state_dict(source_checkpoint["scaler_state"])
        start_epoch = int(source_checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(source_checkpoint.get("best_val_loss", math.inf))

    summary = train_base.summary()
    LOGGER.info(
        "stage=historical_policy_value device=%s amp=%s files=%d total=%d train=%d val=%d "
        "actions=%d params=%s source=%s",
        device,
        amp_enabled,
        summary.parquet_files,
        summary.total_samples,
        len(train_ds),
        len(val_ds),
        len(action_map),
        f"{count_parameters(model):,}",
        source,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                policy_weight=args.policy_weight,
                value_weight=args.value_weight,
                max_batches=args.max_train_batches,
                log_every=args.log_every,
                epoch=epoch,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                scaler=scaler,
                amp_enabled=amp_enabled,
                policy_weight=args.policy_weight,
                value_weight=args.value_weight,
                max_batches=args.max_val_batches,
                log_every=0,
                epoch=epoch,
            )
            scheduler.step()

            LOGGER.info(
                "epoch=%d train_loss=%.5f val_loss=%.5f val_policy=%.5f val_value=%.5f "
                "val_top1=%.4f val_top3=%.4f val_value_mae=%.4f val_sign=%.4f lr=%.6g",
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["policy_loss"],
                val_metrics["value_loss"],
                val_metrics["policy_top1"],
                val_metrics["policy_top3"],
                val_metrics["value_mae"],
                val_metrics["value_sign_accuracy"],
                optimizer.param_groups[0]["lr"],
            )

            record = {
                "stage": "historical_policy_value_warmstart",
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_jsonl(metrics_path, record)

            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]

            payload = {
                "phase": 4,
                "stage": "historical_policy_value_warmstart",
                "iteration": 0,
                "epoch": epoch,
                "model_config": asdict(model.config),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "action_map_sha256": digest,
                "action_map_path": str(Path(args.action_map_path).resolve()),
                "historical_data": str(Path(args.data_dir).resolve()),
                "dataset_index_path": summary.index_path,
                "source_checkpoint": str(Path(source).resolve()),
                "source_stage": source_checkpoint.get("stage"),
                "source_epoch": source_checkpoint.get("epoch"),
                "value_target_convention": "final_game_result_from_side_to_move_perspective",
                "best_val_loss": best_val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "args": vars(args),
            }
            save_checkpoint(payload, ckpt_dir / "latest.pt")
            if improved:
                save_checkpoint(payload, ckpt_dir / "best.pt")
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(payload, ckpt_dir / f"epoch_{epoch:03d}.pt")
    finally:
        _close_dataset(train_base)
        _close_dataset(val_base)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
