"""Cache production detector output for a local, annotated benchmark corpus."""

import argparse
import hashlib
import json
import resource
import socket
import sys
import time
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from transcript_anonymizer.cli import offline_environment
from transcript_anonymizer.detection import Detector, load_policy
from transcript_anonymizer.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--rules-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    offline_environment()

    def blocked(*args, **kwargs):
        raise RuntimeError("Network disabled for benchmark inference")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    documents = json.loads(args.corpus.read_text())
    peak_rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    started = time.monotonic()
    detector = Detector(load_policy(None), args.model, args.rules_only)
    candidates = []

    class RecordingDetector:
        def detect(self, segments):
            findings = detector.detect(segments)
            candidates.append(findings)
            return findings

    metrics = evaluate(documents, RecordingDetector())
    metadata = {
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "detector": detector.metadata,
        "elapsed_s": round(time.monotonic() - started, 3),
        "peak_rss_bytes": max(peak_rss_before, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024),
        "network": "Python socket connect blocked; offline environment enabled",
    }
    for name, data in (("candidates", candidates), ("metrics", metrics), ("metadata", metadata)):
        with (args.output_dir / f"{name}.json").open("x") as handle:
            json.dump(data, handle, indent=2)
    print(json.dumps({"detection": metrics["detection"], "elapsed_s": metadata["elapsed_s"]}))


if __name__ == "__main__":
    main()
