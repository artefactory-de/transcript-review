"""Memory guard limits include build subprocesses, not only the launcher."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc memory guard")
def test_guard_stops_child_memory_exceeding_limit(tmp_path):
    guard = Path(__file__).resolve().parents[1] / "scripts/memory_guard.py"
    child = "import time; payload=bytearray(64*1024*1024); time.sleep(3)"
    parent = (
        "import subprocess,sys; "
        f"subprocess.run([sys.executable, '-c', {child!r}], check=True)"
    )
    result = subprocess.run(
        [sys.executable, str(guard), "--max-rss-mib", "40",
         "--min-available-mib", "128", "--log", str(tmp_path / "guard.log"),
         "--", sys.executable, "-c", parent],
        capture_output=True, text=True, timeout=15, check=False,
    )
    report = json.loads(result.stdout.splitlines()[-1])
    assert result.returncode == 3
    assert report["stopped_for"] == "process_rss_limit"
    assert report["peak_rss_mib"] > 40
    assert report["memory_scope"] == "launched_process_and_descendants"
