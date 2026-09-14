"""Evaluate annotated local inputs without exporting their text."""

import argparse
import json
import socket
from pathlib import Path

from transcript_anonymizer.cli import offline_environment
from transcript_anonymizer.errors import AnonymizerError


def main():
    offline_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=Path("examples/evaluation.german.synthetic.json")
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--policy", type=Path, help="Local threshold/category policy to evaluate")
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-python-network", action="store_true")
    args = parser.parse_args()
    if args.block_python_network:

        def blocked(*args, **kwargs):
            raise RuntimeError("Network calls are forbidden during local evaluation")

        socket.socket.connect = blocked
        socket.socket.connect_ex = blocked
        socket.create_connection = blocked
    from transcript_anonymizer.detection import Detector, load_policy
    from transcript_anonymizer.evaluation import evaluate

    try:
        if args.output.exists():
            raise AnonymizerError("Evaluation output exists; choose a new file.")
        corpus = json.loads(args.corpus.read_text())
        detector = Detector(load_policy(args.policy), args.model, args.rules_only)
        result = evaluate(corpus, detector)
        result["detector"] = detector.metadata
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps({"metrics": str(args.output), "detection": result["detection"]}, indent=2))
        return 0
    except AnonymizerError as error:
        print(f"Evaluation failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
