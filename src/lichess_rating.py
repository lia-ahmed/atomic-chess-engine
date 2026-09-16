#!/usr/bin/env python3
"""Fetch and optionally log a bot account's public Lichess Atomic rating."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def fetch_user(username: str, token: str | None = None) -> dict:
    url = f"https://lichess.org/api/user/{username}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "atomic-chess-evaluator/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Lichess HTTP {exc.code}: {body[:500]}") from exc


def make_snapshot(profile: dict) -> dict:
    atomic = dict(profile.get("perfs", {}).get("atomic", {}))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "username": profile.get("username") or profile.get("id"),
        "bot": bool(profile.get("title") == "BOT" or profile.get("bot", False)),
        "atomic": {
            "games": atomic.get("games"),
            "rating": atomic.get("rating"),
            "rd": atomic.get("rd"),
            "progress": atomic.get("prog"),
            "provisional": atomic.get("prov", False),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch a Lichess user's Atomic rating snapshot.")
    p.add_argument("username")
    p.add_argument("--out", default=None, help="optional JSONL log path")
    p.add_argument("--token-env", default="LICHESS_BOT_TOKEN",
                   help="optional environment variable; public request works without it")
    args = p.parse_args()

    token = os.environ.get(args.token_env) or None
    profile = fetch_user(args.username, token=token)
    snapshot = make_snapshot(profile)
    print(json.dumps(snapshot, indent=2))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
