#!/usr/bin/env python3
"""
pgn_zst_to_duckdb_atomic.py

Stream-process lichess .pgn.zst files (Atomic variant) into a DuckDB database.

Usage:
    python pgn_zst_to_duckdb_atomic.py --input-dir raw_lichess_data --output-db lichess_atomic.duckdb

Dependencies:
    pip install duckdb pandas zstandard
"""

import argparse
import os
import re
import sys
from typing import Dict, Iterable, List, Tuple

try:
    import zstandard as zstd
except Exception as e:
    raise ImportError(
        "The 'zstandard' package is required. Install with: pip install zstandard"
    ) from e

try:
    import duckdb
except Exception as e:
    raise ImportError(
        "The 'duckdb' package is required. Install with: pip install duckdb"
    ) from e

try:
    import pandas as pd
except Exception as e:
    raise ImportError("The 'pandas' package is required. Install with: pip install pandas") from e


# termination values (case-insensitive substrings) that should exclude the game
TERMINATION_EXCLUDE = {"time out", "timeout", "abandon", "abandoned", "resign", "resignation"}

# regex to extract game id from Site header
LICHESS_ID_RE = re.compile(r"lichess\.org/([A-Za-z0-9]+)")

# Batch size for inserting to DuckDB
BATCH_SIZE = 5000


def find_pgn_zst_files(input_dir: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fn in filenames:
            if fn.endswith(".pgn.zst") and fn.startswith("lichess_db_atomic_rated"):
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def extract_gameid(site_value: str) -> str:
    if not site_value:
        return ""
    m = LICHESS_ID_RE.search(site_value)
    if m:
        return m.group(1)
    # fallback: use last path component
    parts = site_value.rstrip("/").split("/")
    return parts[-1] if parts else site_value


def termination_is_excluded(termination_value: str) -> bool:
    if not termination_value:
        return False
    tv = termination_value.strip().lower()
    for excl in TERMINATION_EXCLUDE:
        if excl in tv:
            return True
    return False


def parse_pgn_stream(text_iter: Iterable[str]) -> Iterable[Dict[str, str]]:
    """
    Parse PGN from an iterable of text lines (decoded).
    Yields header dicts with a 'moves' key containing the moves text.

    Strategy:
      - accumulate header lines that start with '[' into a dict
      - after a blank line following headers, collect move lines until a blank line
      - some dumps don't always put a blank line before next header; we handle encountering '[' as start of next game
    """
    headers: Dict[str, str] = {}
    in_headers = False
    in_moves = False
    moves_chunks: List[str] = []

    def flush_game():
        if not headers and not moves_chunks:
            return None
        game = dict(headers)  # copy
        moves_text = " ".join(line.strip() for line in moves_chunks).strip()
        # remove trailing result token if present (common in PGN)
        moves_text = re.sub(r"\s(1-0|0-1|1/2-1/2)\s*$", "", moves_text)
        game["moves"] = moves_text
        return game

    for raw_line in text_iter:
        line = raw_line.rstrip("\n")
        if line.startswith("["):
            # header line
            # if we were collecting moves, this indicates a new game started unexpectedly: flush previous
            if in_moves:
                game = flush_game()
                if game:
                    yield game
                headers = {}
                moves_chunks = []
                in_moves = False
                in_headers = True

            in_headers = True
            # parse header like: [Key "Value"]
            m = re.match(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]$', line)
            if m:
                key = m.group(1)
                val = m.group(2)
                headers[key] = val
            else:
                # fallback: attempt to parse key and raw remainder
                try:
                    k = line.split()[0][1:]
                    v = line[line.find('"') + 1 : line.rfind('"')]
                    headers[k] = v
                except Exception:
                    # ignore malformed header
                    pass
            continue

        if line.strip() == "":
            # empty line
            if in_headers:
                # headers finished -> moves start next
                in_headers = False
                in_moves = True
                continue
            elif in_moves:
                # end of moves -> flush game
                game = flush_game()
                if game:
                    yield game
                headers = {}
                moves_chunks = []
                in_moves = False
                in_headers = False
                continue
            else:
                # stray blank line, skip
                continue

        # non-empty non-header line
        if in_moves:
            moves_chunks.append(line)
        else:
            # sometimes files have no blank line between headers and moves; if we see a line not starting with '[' and not blank, treat it as moves
            in_moves = True
            moves_chunks.append(line)

    # EOF reached, flush last
    if in_moves or headers:
        game = flush_game()
        if game:
            yield game


def process_file(path: str) -> Iterable[Tuple[str, int, int, str, str]]:
    """
    Stream-decompress the zst file and yield tuples (gameid, white_elo, black_elo, result, moves)
    skipping games according to rules.
    """
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            # read in chunks and yield lines
            text_stream = reader.read()
            # decode whole; this is simpler and works for typical monthly files.
            # If you run into memory issues, change to line-by-line streaming decode.
            try:
                text = text_stream.decode("utf-8", errors="replace")
            except Exception:
                text = text_stream.decode("latin-1", errors="replace")

            # iterate through parsed games
            for game in parse_pgn_stream(text.splitlines()):
                # filter variant
                variant = game.get("Variant", "").strip()
                if variant.lower() != "atomic":
                    continue

                termination = game.get("Termination", "").strip()
                if termination_is_excluded(termination):
                    continue

                # get numeric elos
                welo = game.get("WhiteElo", "").strip()
                belo = game.get("BlackElo", "").strip()
                if not welo or not belo:
                    # skip games without ratings
                    continue
                try:
                    welo_i = int(re.sub(r"[^\d-]", "", welo))
                    belo_i = int(re.sub(r"[^\d-]", "", belo))
                except Exception:
                    # skip if elos are malformed
                    continue

                result = game.get("Result", "").strip()
                moves = game.get("moves", "").strip()
                site = game.get("Site", "")
                gameid = extract_gameid(site)

                # ensure non-empty moves and result
                if not moves:
                    continue
                if result not in {"1-0", "0-1", "1/2-1/2"}:
                    # keep only standard results
                    continue

                yield (gameid, welo_i, belo_i, result, moves)


def main():
    parser = argparse.ArgumentParser(description="Import lichess atomic .pgn.zst files into a DuckDB database.")
    parser.add_argument("--input-dir", "-i", required=True, help="Directory containing lichess .pgn.zst files")
    parser.add_argument("--output-db", "-o", default="lichess_atomic.duckdb", help="DuckDB file path to write to")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE, help="Insert batch size")
    args = parser.parse_args()

    files = find_pgn_zst_files(args.input_dir)
    if not files:
        print(f"No files found in {args.input_dir} matching pattern 'lichess_db_atomic_rated_*.pgn.zst'")
        sys.exit(1)

    conn = duckdb.connect(database=args.output_db, read_only=False)
    # Create table (gameid unique)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS games (
            gameid TEXT,
            white_elo INTEGER,
            black_elo INTEGER,
            result TEXT,
            moves TEXT
        )
        """
    )
    # Create unique index to help dedupe inserts (DuckDB doesn't enforce PK easily, but unique index helps)
    # If the index already exists the statement will error, so trap it.
    try:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_gameid ON games(gameid)")
    except Exception:
        # some older duckdb versions may not support IF NOT EXISTS; ignore failures here
        try:
            conn.execute("CREATE UNIQUE INDEX idx_gameid ON games(gameid)")
        except Exception:
            pass

    batch: List[Tuple[str, int, int, str, str]] = []
    total_inserted = 0
    total_seen = 0
    for fpath in files:
        print(f"Processing {fpath} ...")
        for rec in process_file(fpath):
            total_seen += 1
            batch.append(rec)
            if len(batch) >= args.batch_size:
                df = pd.DataFrame(batch, columns=["gameid", "white_elo", "black_elo", "result", "moves"])
                # Use INSERT SELECT via registered temp table to leverage vectorized insertion
                conn.register("temp_batch_df", df)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO games
                    SELECT gameid, white_elo, black_elo, result, moves
                    FROM temp_batch_df
                    WHERE gameid IS NOT NULL AND gameid <> ''
                    """
                )
                # clear registered
                conn.unregister("temp_batch_df")
                inserted = len(batch)
                total_inserted += inserted
                print(f"  inserted batch of {inserted} (total inserted ~ {total_inserted})")
                batch = []

        # after finishing a file, flush remaining batch
        if batch:
            df = pd.DataFrame(batch, columns=["gameid", "white_elo", "black_elo", "result", "moves"])
            conn.register("temp_batch_df", df)
            conn.execute(
                """
                INSERT OR IGNORE INTO games
                SELECT gameid, white_elo, black_elo, result, moves
                FROM temp_batch_df
                WHERE gameid IS NOT NULL AND gameid <> ''
                """
            )
            conn.unregister("temp_batch_df")
            inserted = len(batch)
            total_inserted += inserted
            print(f"  inserted final batch of file ({inserted}) (total inserted ~ {total_inserted})")
            batch = []

    # final flush if any
    if batch:
        df = pd.DataFrame(batch, columns=["gameid", "white_elo", "black_elo", "result", "moves"])
        conn.register("temp_batch_df", df)
        conn.execute(
            """
            INSERT OR IGNORE INTO games
            SELECT gameid, white_elo, black_elo, result, moves
            FROM temp_batch_df
            WHERE gameid IS NOT NULL AND gameid <> ''
            """
        )
        conn.unregister("temp_batch_df")
        total_inserted += len(batch)
        batch = []

    print("Done.")
    print(f"Total games parsed (candidates): {total_seen}")
    cur = conn.execute("SELECT COUNT(*) FROM games").fetchone()
    print(f"Total rows in DB: {cur[0]}")
    conn.close()


if __name__ == "__main__":
    main()