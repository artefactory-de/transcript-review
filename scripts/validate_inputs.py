"""Sequential local-input validation; outputs contain no original passages in logs."""

import argparse
import json
import socket
import time
import zipfile
from collections import Counter
from pathlib import Path

from transcript_anonymizer.cli import offline_environment

offline_environment()

from transcript_anonymizer import workflow
from transcript_anonymizer.detection import Detector, load_policy
from transcript_anonymizer.documents import read_docx, validate_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--block-python-network", action="store_true")
    args = parser.parse_args()
    if args.block_python_network:

        def blocked(*args, **kwargs):
            raise RuntimeError("Network calls are forbidden during local validation")

        socket.socket.connect = blocked
        socket.socket.connect_ex = blocked
        socket.create_connection = blocked
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    policy = load_policy(None)
    detector = Detector(policy, model_path=args.model, rules_only=args.rules_only)

    def progress(completed, total):
        if completed % 25 == 0 or completed == total:
            print(
                json.dumps({"processed_segments": completed, "total_segments": total}), flush=True
            )

    detector.progress = progress
    report = {
        "mode": "rules_only" if args.rules_only else "local_model",
        "inputs": [],
        "quality_claim": "Unlabelled input smoke test, not PII recall or release acceptance",
    }
    for index, source in enumerate(args.inputs, 1):
        start = time.monotonic()
        print(json.dumps({"input_index": index, "event": "prepare_start"}), flush=True)
        result = workflow.prepare(source, args.out / f"run-{index}", policy, detector)
        state, _ = workflow._load(args.out / f"run-{index}")
        candidate = Path(result["candidate"])
        validate_output(candidate)
        reconstructed = read_docx(candidate)["segments"]
        expected = workflow._render(state)
        if [s["text"] for s in reconstructed] != [s["text"] for s in expected]:
            raise RuntimeError("Reconstruction text mismatch")
        with zipfile.ZipFile(candidate) as package:
            media = sum(name.startswith("word/media/") for name in package.namelist())
            parts = len(package.namelist())
        row = {
            "input_index": index,
            "source_sha256": workflow.file_hash(source),
            "segments": result["segments"],
            "source_characters": sum(len(s["text"]) for s in state["source_segments"]),
            "groups": result["groups"],
            "occurrences": result["occurrences"],
            "categories": dict(Counter(f["category"] for f in state["selected"])),
            "input_parts": len(state["inventory"]),
            "output_parts": parts,
            "output_media_parts": media,
            "text_roundtrip": True,
            "signed_off": result["signed_off"],
            "elapsed_s": round(time.monotonic() - start, 2),
            "candidate_bytes": candidate.stat().st_size,
            "workbook_bytes": Path(result["workbook"]).stat().st_size,
        }
        report["inputs"].append(row)
        print(json.dumps(row), flush=True)
        (args.out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
