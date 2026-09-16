#!/usr/bin/env python3
"""PyTorch dataset for supervised atomic-chess move prediction.

The parquet stores one row for the position *after* 'uci_move'.
Therefore a supervised example is formed as:

    state after ply N  ->  move played at ply N+1

The first move of each game has no corresponding starting-position row in the data and is intentionally omitted. The final position has no next move and is also omitted.

This module builds a compact sidecar index over parquet row locations. The index keeps the large board blobs on disk, enables deterministic train/validation splits by game id, and supports row-group-local shuffling for better I/O locality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from representations import (
    ActionMap,
    FLAT_PLANE_SIZE,
    NUM_PLANES,
    action_index_to_uci,
    flat_bytes_to_planes,
    invert_action_map,
    load_action_map,
    normalize_uci,
)


INDEX_VERSION = 1
INDEX_ARRAY_KEYS = ("file_idx", "row_group", "row_in_group", "action_index", "split")
READ_COLUMNS = (
    "gameid",
    "ply_index",
    "t",
    "side_to_move",
    "white_elo",
    "black_elo",
    "result",
    "uci_move",
    "fen",
    "piece_planes",
)


@dataclass(frozen=True)
class DatasetSummary:
    parquet_files: int
    total_samples: int
    train_samples: int
    val_samples: int
    action_count: int
    index_path: str


def discover_parquet_files(data_path: str | Path) -> list[Path]:
    """Resolve a clean ish parquet file or parquet directory into stable file order."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"expected a .parquet file, got {path}")
        return [path.resolve()]

    preferred = path / "per_ply.parquet"
    if preferred.is_file():
        return [preferred.resolve()]

    files = sorted(p.resolve() for p in path.rglob("*.parquet") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no parquet files found under {path}")
    return files


def default_index_path(data_path: str | Path) -> Path:
    path = Path(data_path)
    if path.is_file():
        return path.with_suffix(path.suffix + ".phase3_index.npz")
    return path / "phase3_index.npz"


def _meta_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".json")


def _action_map_digest(action_map: ActionMap) -> str:
    canonical = json.dumps(
        {str(k): int(v) for k, v in sorted(action_map.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _file_fingerprint(path: Path) -> dict[str, object]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _split_for_game(gameid: str, val_fraction: float, seed: int) -> int:
    """Return 0 for train, 1 for validation, deterministically by game id."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")
    if val_fraction == 0.0:
        return 0
    payload = f"{seed}:{gameid}".encode("utf-8", errors="replace")
    bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    threshold = int(val_fraction * (1 << 64))
    return 1 if bucket < threshold else 0


def _validate_phase1_schema(parquet_file: pq.ParquetFile, path: Path) -> None:
    names = set(parquet_file.schema_arrow.names)
    required = {"gameid", "ply_index", "uci_move", "piece_planes"}
    missing = required - names
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def _index_metadata(
    files: Sequence[Path],
    action_map: ActionMap,
    val_fraction: float,
    split_seed: int,
) -> dict[str, object]:
    return {
        "version": INDEX_VERSION,
        "files": [_file_fingerprint(p) for p in files],
        "action_map_sha256": _action_map_digest(action_map),
        "val_fraction": float(val_fraction),
        "split_seed": int(split_seed),
        "pairing": "state_after_ply_N_to_move_at_ply_N_plus_1",
    }


def _metadata_matches(current: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    return dict(current) == dict(expected)


def build_sample_index(
    data_path: str | Path,
    action_map: ActionMap,
    *,
    index_path: str | Path | None = None,
    val_fraction: float = 0.1,
    split_seed: int = 42,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """Build an on-disk index pairing each stored position with the next move.

    Each sample stores the parquet file / row-group / row location of the state row plus the action index of the next row's 'uci_move'. 
    Consecutive plies are *required* to share a game id and have 'ply_index' increasing by one.
    """
    if not action_map:
        raise ValueError("action_map must not be empty")
    invert_action_map(action_map)  # validates dense, unique indices

    files = discover_parquet_files(data_path)
    out = Path(index_path) if index_path is not None else default_index_path(data_path)
    out = out.resolve()
    meta_path = _meta_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    expected_meta = _index_metadata(files, action_map, val_fraction, split_seed)
    if out.exists() and meta_path.exists() and not overwrite:
        try:
            with meta_path.open("r", encoding="utf-8") as fp:
                existing_meta = json.load(fp)
            if _metadata_matches(existing_meta, expected_meta):
                return out
        except (OSError, json.JSONDecodeError):
            pass
        raise RuntimeError(
            f"existing index {out} does not match the current parquet/action-map/split settings; "
            "use rebuild_index=True or --rebuild-index"
        )

    file_idx_values: list[int] = []
    row_group_values: list[int] = []
    row_in_group_values: list[int] = []
    action_values: list[int] = []
    split_values: list[int] = []

    total_row_groups = 0
    parquet_handles: list[pq.ParquetFile] = []
    for path in files:
        pf = pq.ParquetFile(path)
        _validate_phase1_schema(pf, path)
        parquet_handles.append(pf)
        total_row_groups += pf.num_row_groups

    progress = tqdm(
        total=total_row_groups,
        desc="index row groups",
        unit="rg",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for file_idx, (path, pf) in enumerate(zip(files, parquet_handles)):
            previous: Optional[tuple[str, int, int, int]] = None
            # tuple = (gameid, ply_index, row_group, row_in_group)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=["gameid", "ply_index", "uci_move"])
                gameids = table.column("gameid").to_pylist()
                plies = table.column("ply_index").to_pylist()
                moves = table.column("uci_move").to_pylist()

                for row_in_group, (gameid_raw, ply_raw, move_raw) in enumerate(
                    zip(gameids, plies, moves)
                ):
                    if gameid_raw is None or ply_raw is None:
                        previous = None
                        continue

                    gameid = str(gameid_raw)
                    ply = int(ply_raw)

                    if previous is not None:
                        prev_gameid, prev_ply, prev_rg, prev_row = previous
                        if gameid == prev_gameid and ply == prev_ply + 1:
                            if move_raw is None:
                                raise ValueError(
                                    f"null uci_move for target game={gameid!r} ply={ply} in {path}"
                                )
                            move = normalize_uci(str(move_raw))
                            try:
                                action_index = int(action_map[move])
                            except KeyError as exc:
                                raise KeyError(
                                    f"move {move!r} from {path} is missing from action map"
                                ) from exc

                            file_idx_values.append(file_idx)
                            row_group_values.append(prev_rg)
                            row_in_group_values.append(prev_row)
                            action_values.append(action_index)
                            split_values.append(_split_for_game(gameid, val_fraction, split_seed))

                    previous = (gameid, ply, rg, row_in_group)
                progress.update(1)
    finally:
        progress.close()
        # Explicitly close ParquetFile handles. This matters on Windows, where
        # an open Arrow file handle prevents TemporaryDirectory (or other code)
        # from deleting/replacing the parquet file.
        for pf in parquet_handles:
            try:
                pf.close()
            except Exception:
                pass

    if not action_values:
        raise ValueError("no consecutive-ply training samples were found")

    arrays = {
        "file_idx": np.asarray(file_idx_values, dtype=np.int32),
        "row_group": np.asarray(row_group_values, dtype=np.int32),
        "row_in_group": np.asarray(row_in_group_values, dtype=np.int32),
        "action_index": np.asarray(action_values, dtype=np.int32),
        "split": np.asarray(split_values, dtype=np.uint8),
    }

    tmp_npz = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp_npz, **arrays)
    os.replace(tmp_npz, out)

    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    with tmp_meta.open("w", encoding="utf-8") as fp:
        json.dump(expected_meta, fp, indent=2)
        fp.write("\n")
    os.replace(tmp_meta, meta_path)
    return out


class AtomicChessDataset(Dataset):
    """Map-style PyTorch dataset backed by parquet row groups.

    Returns '(state_tensor, action_index, metadata)' by default, where the state tensor is float32 '[14,8,8]' and the target is the move that follows that state in the same game.
    """

    def __init__(
        self,
        data_path: str | Path,
        action_map: ActionMap | str | Path,
        *,
        split: str = "train",
        val_fraction: float = 0.1,
        split_seed: int = 42,
        index_path: str | Path | None = None,
        rebuild_index: bool = False,
        row_group_cache_size: int = 4,
        include_metadata: bool = True,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be one of: train, val, all")
        if row_group_cache_size < 1:
            raise ValueError("row_group_cache_size must be >= 1")

        self.data_path = str(data_path)
        self.files = discover_parquet_files(data_path)
        self.action_map: Dict[str, int] = (
            load_action_map(action_map) if isinstance(action_map, (str, Path)) else dict(action_map)
        )
        self.inverse_action_map = invert_action_map(self.action_map)
        self.split = split
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)
        self.row_group_cache_size = int(row_group_cache_size)
        self.include_metadata = bool(include_metadata)

        self.index_path = build_sample_index(
            data_path,
            self.action_map,
            index_path=index_path,
            val_fraction=val_fraction,
            split_seed=split_seed,
            overwrite=rebuild_index,
            show_progress=True,
        )

        with np.load(self.index_path, allow_pickle=False) as loaded:
            missing = set(INDEX_ARRAY_KEYS) - set(loaded.files)
            if missing:
                raise ValueError(f"index is missing arrays: {sorted(missing)}")
            self._file_idx = loaded["file_idx"].copy()
            self._row_group = loaded["row_group"].copy()
            self._row_in_group = loaded["row_in_group"].copy()
            self._action_index = loaded["action_index"].copy()
            split_array = loaded["split"].copy()

        if split == "train":
            self._sample_ids = np.flatnonzero(split_array == 0).astype(np.int64)
        elif split == "val":
            self._sample_ids = np.flatnonzero(split_array == 1).astype(np.int64)
        else:
            self._sample_ids = np.arange(len(split_array), dtype=np.int64)

        self._parquet_files: dict[int, pq.ParquetFile] = {}
        self._row_group_cache: OrderedDict[tuple[int, int], object] = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        # pyarrow handles/cached tables are reopened independently in DataLoader workers.
        state["_parquet_files"] = {}
        state["_row_group_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        """Release cached Arrow tables and open Parquet file handles.

        Explicit closing is important on Windows cos open file handles can prevent parquet files or temporary directories from being deleted.
        DataLoader workers receive their own handle/cache state via __getstate__.
        """
        cache = getattr(self, "_row_group_cache", None)
        if cache is not None:
            cache.clear()

        parquet_files = getattr(self, "_parquet_files", None)
        if parquet_files is not None:
            for pf in list(parquet_files.values()):
                try:
                    pf.close()
                except Exception:
                    pass
            parquet_files.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return int(self._sample_ids.size)

    @property
    def num_actions(self) -> int:
        return len(self.action_map)

    def _global_sample_id(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return int(self._sample_ids[index])

    def sample_location(self, index: int) -> tuple[int, int, int]:
        sample_id = self._global_sample_id(index)
        return (
            int(self._file_idx[sample_id]),
            int(self._row_group[sample_id]),
            int(self._row_in_group[sample_id]),
        )

    def _get_parquet_file(self, file_idx: int) -> pq.ParquetFile:
        pf = self._parquet_files.get(file_idx)
        if pf is None:
            pf = pq.ParquetFile(self.files[file_idx])
            self._parquet_files[file_idx] = pf
        return pf

    def _get_row_group(self, file_idx: int, row_group: int):
        key = (file_idx, row_group)
        cached = self._row_group_cache.get(key)
        if cached is not None:
            self._row_group_cache.move_to_end(key)
            return cached

        pf = self._get_parquet_file(file_idx)
        available = set(pf.schema_arrow.names)
        columns = [name for name in READ_COLUMNS if name in available]
        table = pf.read_row_group(row_group, columns=columns)
        self._row_group_cache[key] = table
        self._row_group_cache.move_to_end(key)
        while len(self._row_group_cache) > self.row_group_cache_size:
            self._row_group_cache.popitem(last=False)
        return table

    def __getitem__(self, index: int):
        sample_id = self._global_sample_id(index)
        file_idx = int(self._file_idx[sample_id])
        row_group = int(self._row_group[sample_id])
        row_in_group = int(self._row_in_group[sample_id])
        target = int(self._action_index[sample_id])

        table = self._get_row_group(file_idx, row_group)
        blob = table.column("piece_planes")[row_in_group].as_py()
        if blob is None or len(blob) != FLAT_PLANE_SIZE:
            raise ValueError(
                f"invalid piece_planes at file={self.files[file_idx]} row_group={row_group} "
                f"row={row_in_group}"
            )

        planes = flat_bytes_to_planes(blob)
        state = torch.from_numpy(planes).to(dtype=torch.float32)
        action = torch.tensor(target, dtype=torch.long)

        if not self.include_metadata:
            return state, action

        def scalar(name: str, default=None):
            if name not in table.column_names:
                return default
            value = table.column(name)[row_in_group].as_py()
            return default if value is None else value

        metadata = {
            "gameid": str(scalar("gameid", "")),
            "ply_index": int(scalar("ply_index", -1)),
            "t": float(scalar("t", 0.0)),
            "side_to_move": int(scalar("side_to_move", -1)),
            "white_elo": int(scalar("white_elo", 0)),
            "black_elo": int(scalar("black_elo", 0)),
            "result": int(scalar("result", 0)),
            "target_uci": self.inverse_action_map[target],
        }
        return state, action, metadata

    def summary(self) -> DatasetSummary:
        with np.load(self.index_path, allow_pickle=False) as loaded:
            split_array = loaded["split"]
            train_count = int(np.count_nonzero(split_array == 0))
            val_count = int(np.count_nonzero(split_array == 1))
            total_count = int(split_array.size)
        return DatasetSummary(
            parquet_files=len(self.files),
            total_samples=total_count,
            train_samples=train_count,
            val_samples=val_count,
            action_count=len(self.action_map),
            index_path=str(self.index_path),
        )


class RowGroupShuffleSampler(Sampler[int]):
    """Shuffle row groups and samples within groups while preserving I/O locality."""

    def __init__(self, dataset: AtomicChessDataset, seed: int = 0) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for dataset_index in range(len(dataset)):
            file_idx, row_group, _ = dataset.sample_location(dataset_index)
            grouped[(file_idx, row_group)].append(dataset_index)
        self.groups = [np.asarray(values, dtype=np.int64) for values in grouped.values()]

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        group_order = rng.permutation(len(self.groups))
        for group_idx in group_order:
            group = self.groups[int(group_idx)]
            order = rng.permutation(group.size)
            for local_idx in order:
                yield int(group[int(local_idx)])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/inspect the AtomicChessDataset index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="per_ply.parquet or directory containing it")
    parser.add_argument("--action-map", required=True, help="action_map.json")
    parser.add_argument("--index-path", default=None, help="Optional sidecar .npz index path")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--show-sample", action="store_true", help="Print one decoded training sample")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        dataset = AtomicChessDataset(
            args.data,
            args.action_map,
            split="train",
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            index_path=args.index_path,
            rebuild_index=args.rebuild_index,
        )
        summary = dataset.summary()
        print(
            f"files={summary.parquet_files} total_samples={summary.total_samples} "
            f"train_samples={summary.train_samples} val_samples={summary.val_samples} "
            f"actions={summary.action_count} index={summary.index_path}"
        )
        if args.show_sample and len(dataset):
            state, action, metadata = dataset[0]
            print(
                f"sample_shape={tuple(state.shape)} dtype={state.dtype} "
                f"action={int(action)} metadata={metadata}"
            )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
