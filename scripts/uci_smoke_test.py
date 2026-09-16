#!/usr/bin/env python3
"""Launch uci_engine.py and verify the minimal UCI handshake/search path."""

from __future__ import annotations

import argparse
import queue
import subprocess
import threading
import time
from pathlib import Path
import sys

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--engine", default="src/uci_engine.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--action-map-path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--movetime-ms", type=int, default=500)
    args = p.parse_args()

    cmd = [
        args.python,
        args.engine,
        "--checkpoint", args.checkpoint,
        "--action-map-path", args.action_map_path,
        "--device", args.device,
        "--max-move-time-ms", str(max(args.movetime_ms, 50)),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    out_q: queue.Queue[str] = queue.Queue()
    err_q: queue.Queue[str] = queue.Queue()

    def pump(stream, q):
        for line in stream:
            q.put(line.rstrip("\r\n"))

    threading.Thread(target=pump, args=(proc.stdout, out_q), daemon=True).start()
    threading.Thread(target=pump, args=(proc.stderr, err_q), daemon=True).start()

    def send(text: str) -> None:
        print(f">> {text}")
        proc.stdin.write(text + "\n")
        proc.stdin.flush()

    def wait_for(prefix: str, timeout: float = 30.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = out_q.get(timeout=0.2)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            print(f"<< {line}")
            if line.startswith(prefix):
                return line
        errors = []
        while not err_q.empty():
            errors.append(err_q.get_nowait())
        raise RuntimeError(f"timeout waiting for {prefix!r}; stderr={' | '.join(errors[-20:])}")

    try:
        send("uci")
        wait_for("uciok", 20)
        send("isready")
        wait_for("readyok", 20)
        send("setoption name UCI_Variant value atomic")
        send("ucinewgame")
        send("position startpos")
        send(f"go movetime {args.movetime_ms}")
        best = wait_for("bestmove ", 60)
        if best == "bestmove 0000":
            raise RuntimeError("engine returned no move from the starting position")
        print("UCI_SMOKE_TEST=PASSED")
    finally:
        if proc.poll() is None:
            try:
                send("quit")
            except Exception:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
