"""Run one local validation process with conservative Linux memory-pressure limits."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def memory():
    return {
        line.split(":")[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
    }


def process_memory(pid):
    try:
        rows = Path(f"/proc/{pid}/status").read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return 0, 0
    values = {line.split(":")[0]: line.split(":", 1)[1].strip() for line in rows}
    return int(values.get("VmRSS", "0 kB").split()[0]), int(values.get("VmSwap", "0 kB").split()[0])


def process_tree_memory(pid):
    """Conservatively sum RSS/swap across the launched process and descendants.

    Shared pages can be counted more than once. This guard favors stopping early
    over undercounting a build worker; host available memory is checked separately.
    """
    pending = [pid]
    seen = set()
    rss = swapped = 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        current_rss, current_swap = process_memory(current)
        rss += current_rss
        swapped += current_swap
        # A worker can be created by any thread, not only the main thread.
        try:
            for task in Path(f"/proc/{current}/task").iterdir():
                try:
                    pending.extend(int(child) for child in (task / "children").read_text().split())
                except (FileNotFoundError, ProcessLookupError):
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rss, swapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rss-mib", type=int, default=4096)
    parser.add_argument("--min-available-mib", type=int, default=3072)
    parser.add_argument("--max-process-swap-mib", type=int, default=256)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required after --")
    if memory()["MemAvailable"] < args.min_available_mib * 1024:
        raise SystemExit("Insufficient available memory before starting; no process launched.")
    peak = 0
    lowest = memory()["MemAvailable"]
    started = time.monotonic()
    next_report = started
    reason = None
    with args.log.open("xb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                rss, swapped = process_tree_memory(process.pid)
                available = memory()["MemAvailable"]
                peak, lowest = max(peak, rss), min(lowest, available)
                if rss > args.max_rss_mib * 1024:
                    reason = "process_rss_limit"
                elif swapped > args.max_process_swap_mib * 1024:
                    reason = "process_swap_limit"
                elif available < args.min_available_mib * 1024:
                    reason = "system_available_memory_floor"
                if reason:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                if time.monotonic() >= next_report:
                    print(
                        json.dumps(
                            {
                                "elapsed_s": round(time.monotonic() - started),
                                "rss_mib": round(rss / 1024),
                                "available_mib": round(available / 1024),
                            }
                        ),
                        flush=True,
                    )
                    next_report = time.monotonic() + 30
                time.sleep(1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    print(
        json.dumps(
            {
                "exit_code": process.returncode,
                "stopped_for": reason,
                "peak_rss_mib": round(peak / 1024),
                "min_available_mib": round(lowest / 1024),
                "elapsed_s": round(time.monotonic() - started),
                "log": str(args.log),
                "memory_scope": "launched_process_and_descendants",
            }
        ),
        flush=True,
    )
    return 3 if reason else process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
