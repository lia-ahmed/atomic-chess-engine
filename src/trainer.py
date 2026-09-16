#!/usr/bin/env python3
"""Train the policy+value network from MCTS self-play JSONL shards.

Policy targets are sparse normalized visit counts produced by 'self_play.py'.
Value targets are final game outcomes in {-1, 0, +1}, always from the
side-to-move perspective of the stored FEN.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from models import (
    PolicyValueNet,
    action_map_digest,
    count_parameters,
    load_policy_value_checkpoint,
)
from representations import fen_to_planes, load_action_map

LOGGER = logging.getLogger("trainer")
INDEX_VERSION = 1


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


def split_for_game(game_id: str, val_fraction: float, seed: int) -> int:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= x < 1")
    if val_fraction == 0.0:
        return 0
    digest = hashlib.blake2b(f"{seed}:{game_id}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big")
    return 1 if bucket < int(val_fraction * (1 << 64)) else 0


def discover_selfplay_files(data_dir: str | Path) -> list[Path]:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise ValueError("self-play input file must be .jsonl")
        return [path.resolve()]
    files = sorted(p.resolve() for p in path.rglob("selfplay_*.jsonl") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no selfplay_*.jsonl files found under {path}")
    return files


def file_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def default_index_path(data_dir: str | Path) -> Path:
    p = Path(data_dir)
    return (p.parent if p.is_file() else p) / "selfplay_index.npz"


def build_selfplay_index(
    data_dir: str | Path,
    *,
    val_fraction: float = 0.1,
    split_seed: int = 42,
    index_path: str | Path | None = None,
    rebuild: bool = False,
    max_samples: Optional[int] = None,
) -> tuple[Path, list[Path]]:
    files = discover_selfplay_files(data_dir)
    out = Path(index_path) if index_path is not None else default_index_path(data_dir)
    out = out.resolve()
    meta_path = out.with_suffix(out.suffix + ".json")
    expected = {
        "version": INDEX_VERSION,
        "files": [file_fingerprint(p) for p in files],
        "val_fraction": float(val_fraction),
        "split_seed": int(split_seed),
        "max_samples": max_samples,
    }
    if out.exists() and meta_path.exists() and not rebuild:
        try:
            with meta_path.open("r", encoding="utf-8") as fp:
                if json.load(fp) == expected:
                    return out, files
        except (OSError, json.JSONDecodeError):
            pass
        raise RuntimeError("self-play index is stale; use --rebuild-index")

    file_ids: list[int] = []
    offsets: list[int] = []
    splits: list[int] = []
    count = 0
    for file_idx, path in enumerate(files):
        with path.open("rb") as fp:
            while True:
                offset = fp.tell()
                line = fp.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path} at byte {offset}: {exc}") from exc
                game_id = str(record.get("game_id", ""))
                if not game_id:
                    raise ValueError(f"missing game_id in {path} at byte {offset}")
                file_ids.append(file_idx)
                offsets.append(offset)
                splits.append(split_for_game(game_id, val_fraction, split_seed))
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
        if max_samples is not None and count >= max_samples:
            break
    if not offsets:
        raise ValueError("no self-play samples found")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(
        tmp_npz,
        file_idx=np.asarray(file_ids, dtype=np.int32),
        offset=np.asarray(offsets, dtype=np.int64),
        split=np.asarray(splits, dtype=np.uint8),
    )
    os.replace(tmp_npz, out)
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    with tmp_meta.open("w", encoding="utf-8") as fp:
        json.dump(expected, fp, indent=2)
        fp.write("\n")
    os.replace(tmp_meta, meta_path)
    return out, files


class SelfPlayDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str,
        val_fraction: float = 0.1,
        split_seed: int = 42,
        index_path: str | Path | None = None,
        rebuild_index: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val, or all")
        self.index_path, self.files = build_selfplay_index(
            data_dir,
            val_fraction=val_fraction,
            split_seed=split_seed,
            index_path=index_path,
            rebuild=rebuild_index,
            max_samples=max_samples,
        )
        with np.load(self.index_path, allow_pickle=False) as z:
            self._file_idx = z["file_idx"].copy()
            self._offset = z["offset"].copy()
            split_array = z["split"].copy()
        if split == "train":
            self._ids = np.flatnonzero(split_array == 0).astype(np.int64)
        elif split == "val":
            self._ids = np.flatnonzero(split_array == 1).astype(np.int64)
        else:
            self._ids = np.arange(split_array.size, dtype=np.int64)
        self._handles: dict[int, Any] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def close(self) -> None:
        handles = getattr(self, "_handles", {})
        for fp in handles.values():
            try:
                fp.close()
            except Exception:
                pass
        handles.clear()

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return int(self._ids.size)

    def _handle(self, file_idx: int):
        fp = self._handles.get(file_idx)
        if fp is None or fp.closed:
            fp = self.files[file_idx].open("rb")
            self._handles[file_idx] = fp
        return fp

    def __getitem__(self, index: int):
        sample_id = int(self._ids[index])
        file_idx = int(self._file_idx[sample_id])
        offset = int(self._offset[sample_id])
        fp = self._handle(file_idx)
        fp.seek(offset)
        record = json.loads(fp.readline())

        fen = str(record["fen"])
        state = torch.from_numpy(fen_to_planes(fen)).to(dtype=torch.float32)
        indices = torch.tensor(record["policy_indices"], dtype=torch.long)
        probs = torch.tensor(record["policy_probs"], dtype=torch.float32)
        value = torch.tensor(float(record["value_target"]), dtype=torch.float32)
        if indices.numel() == 0 or indices.numel() != probs.numel():
            raise ValueError(f"invalid sparse policy target in game {record.get('game_id')}")
        total = float(probs.sum().item())
        if total <= 0:
            raise ValueError("policy target has non-positive mass")
        probs = probs / total
        return state, indices, probs, value


def collate_selfplay(batch, num_actions: int):
    states = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.zeros((len(batch), num_actions), dtype=torch.float32)
    values = torch.stack([item[3] for item in batch], dim=0)
    for row, (_, indices, probs, _) in enumerate(batch):
        if int(indices.min()) < 0 or int(indices.max()) >= num_actions:
            raise ValueError("policy target action index outside action map")
        targets[row].index_add_(0, indices, probs)
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return states, targets, values


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def batch_losses(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    *,
    policy_weight: float,
    value_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(values, value_targets)
    total = policy_weight * policy_loss + value_weight * value_loss
    return total, policy_loss, value_loss


def run_epoch(
    model: PolicyValueNet,
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
    total_examples = 0
    sums = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "policy_top1": 0.0}
    value_abs_error = 0.0
    sign_correct = 0
    sign_total = 0
    started = time.perf_counter()

    grad_context = contextlib.nullcontext() if training else torch.no_grad()
    with grad_context:
        for batch_idx, (states, policy_targets, value_targets) in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            states = states.to(device, non_blocking=True)
            policy_targets = policy_targets.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                logits, values = model(states)
                loss, p_loss, v_loss = batch_losses(
                    logits,
                    values,
                    policy_targets,
                    value_targets,
                    policy_weight=policy_weight,
                    value_weight=value_weight,
                )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            n = int(states.shape[0])
            predicted = logits.argmax(dim=1)
            target_mode = policy_targets.argmax(dim=1)
            top1 = int((predicted == target_mode).sum().item())
            sums["loss"] += float(loss.detach().item()) * n
            sums["policy_loss"] += float(p_loss.detach().item()) * n
            sums["value_loss"] += float(v_loss.detach().item()) * n
            sums["policy_top1"] += top1
            value_abs_error += float((values.detach() - value_targets).abs().sum().item())
            decisive = value_targets != 0
            if decisive.any():
                sign_correct += int(
                    (torch.sign(values.detach()[decisive]) == torch.sign(value_targets[decisive])).sum().item()
                )
                sign_total += int(decisive.sum().item())
            total_examples += n

            if training and log_every > 0 and batch_idx % log_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                LOGGER.info(
                    "epoch=%d batch=%d examples=%d loss=%.4f policy=%.4f value=%.4f ex_per_s=%.1f",
                    epoch,
                    batch_idx,
                    total_examples,
                    sums["loss"] / total_examples,
                    sums["policy_loss"] / total_examples,
                    sums["value_loss"] / total_examples,
                    total_examples / elapsed,
                )

    if total_examples == 0:
        raise RuntimeError("loader produced no examples")
    return {
        "loss": sums["loss"] / total_examples,
        "policy_loss": sums["policy_loss"] / total_examples,
        "value_loss": sums["value_loss"] / total_examples,
        "policy_top1": sums["policy_top1"] / total_examples,
        "value_mae": value_abs_error / total_examples,
        "value_sign_accuracy": sign_correct / sign_total if sign_total else 0.0,
        "examples": float(total_examples),
        "seconds": time.perf_counter() - started,
    }


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Phase 4 policy+value net on MCTS self-play data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True, help="Directory of selfplay_*.jsonl shards")
    p.add_argument("--action-map-path", "--action-map", dest="action_map_path", required=True)
    p.add_argument("--init-checkpoint", required=True, help="Phase 4 init.pt or prior trained checkpoint")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--iteration", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--policy-weight", type=float, default=1.0)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--index-path", default=None)
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    amp = p.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=None)
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise SystemExit("epochs, batch-size and lr must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_enabled = (device.type == "cuda") if args.amp is None else bool(args.amp)
    if amp_enabled and device.type != "cuda":
        raise SystemExit("AMP is only supported on CUDA in this script")

    action_map = load_action_map(args.action_map_path)
    digest = action_map_digest(action_map)
    source = args.resume if args.resume else args.init_checkpoint
    model, checkpoint = load_policy_value_checkpoint(source, device=device)
    if int(model.config.num_actions) != len(action_map):
        raise SystemExit("network action count does not match action_map.json")
    if checkpoint.get("action_map_sha256") not in (None, digest):
        raise SystemExit("checkpoint action map does not match action_map.json")

    train_ds = SelfPlayDataset(
        args.data_dir,
        split="train",
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        index_path=args.index_path,
        rebuild_index=args.rebuild_index,
        max_samples=args.max_samples,
    )
    val_ds = SelfPlayDataset(
        args.data_dir,
        split="val",
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        index_path=args.index_path,
        rebuild_index=False,
        max_samples=args.max_samples,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(f"empty split: train={len(train_ds)} val={len(val_ds)}")

    collate = partial(collate_selfplay, num_actions=len(action_map))
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = make_grad_scaler(amp_enabled)
    start_epoch = 1
    best_val = math.inf
    if args.resume:
        if checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val_loss", math.inf))

    LOGGER.info(
        "device=%s amp=%s train=%d val=%d actions=%d params=%s source=%s",
        device,
        amp_enabled,
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
                "epoch=%d train_loss=%.5f val_loss=%.5f val_policy=%.5f "
                "val_value=%.5f val_policy_top1=%.4f val_value_mae=%.4f val_sign=%.4f lr=%.6g",
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["policy_loss"],
                val_metrics["value_loss"],
                val_metrics["policy_top1"],
                val_metrics["value_mae"],
                val_metrics["value_sign_accuracy"],
                optimizer.param_groups[0]["lr"],
            )

            with metrics_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps({
                    "iteration": args.iteration,
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                }) + "\n")

            improved = val_metrics["loss"] < best_val
            if improved:
                best_val = val_metrics["loss"]
            payload = {
                "phase": 4,
                "stage": "selfplay_trained",
                "iteration": args.iteration,
                "epoch": epoch,
                "model_config": asdict(model.config),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "action_map_sha256": digest,
                "action_map_path": str(Path(args.action_map_path).resolve()),
                "selfplay_data": str(Path(args.data_dir).resolve()),
                "best_val_loss": best_val,
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
        train_ds.close()
        val_ds.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
