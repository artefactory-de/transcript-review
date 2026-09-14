"""Small operator interface; no uploads or document text in normal output."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .errors import AnonymizerError


def offline_environment():
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "DO_NOT_TRACK",
    ):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def model_location(value: Path | None) -> Path | None:
    if value is not None:
        return value
    bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "model"
    return bundled if bundled.is_dir() else None


class DeferredDetector:
    """Sign-off-only imports never need to allocate a model."""

    def __init__(self, policy: dict, model: Path | None, rules_only: bool):
        self.policy, self.model, self.rules_only = policy, model, rules_only
        self.inner = None

    def _get(self):
        if self.inner is None:
            from .detection import Detector

            self.inner = Detector(self.policy, model_location(self.model), self.rules_only)
            self.inner.progress = self._progress
        return self.inner

    @staticmethod
    def _progress(completed, total):
        if completed % 50 == 0 or completed == total:
            print(f"Processed {completed}/{total} segments", file=sys.stderr, flush=True)

    @property
    def metadata(self):
        return self._get().metadata

    def detect(self, segments):
        return self._get().detect(segments)


def main(argv: list[str] | None = None) -> int:
    offline_environment()
    parser = argparse.ArgumentParser(description="Offline PII review. No automatic transfer.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inventory a DOCX without exposing its text")
    inspect.add_argument("source", type=Path)
    prepare = commands.add_parser(
        "prepare", help="Create a pending candidate and protected workbook"
    )
    prepare.add_argument("source", type=Path)
    prepare.add_argument("--run", type=Path, required=True)
    prepare.add_argument("--policy", type=Path)
    prepare.add_argument(
        "--review-sample-size",
        type=int,
        default=5,
        help="Deterministic low-ranked remainder sample size (0 disables it)",
    )
    review = commands.add_parser("review", help="Import Excel decisions and sign-off")
    reviews = review.add_subparsers(dest="review_command", required=True)
    importing = reviews.add_parser("import")
    importing.add_argument("workbook", type=Path)
    importing.add_argument("--run", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("run", type=Path)
    export = commands.add_parser(
        "export", help="Copy signed-off clean artifacts to a new local folder"
    )
    export.add_argument("run", type=Path)
    export.add_argument("--out", type=Path, required=True)
    health = commands.add_parser("health-check", help="Check local detector and format modules")
    health.add_argument("--policy", type=Path)
    sample = commands.add_parser("sample", help="Create a synthetic, rules-only demonstration run")
    sample.add_argument("--out", type=Path, required=True)
    for command in (prepare, importing, health):
        command.add_argument("--model", type=Path, help="Local model snapshot; no downloads")
        command.add_argument(
            "--rules-only",
            action="store_true",
            help="Explicit limited test mode, not model validation",
        )
    args = parser.parse_args(argv)
    try:
        from . import documents, workflow
        from .detection import load_policy

        if args.command == "inspect":
            document = documents.read_docx(args.source)
            result = {
                "segments": len(document["segments"]),
                "characters": sum(len(s["text"]) for s in document["segments"]),
                "parts": len(document["inventory"]),
                "removed_parts": sum(p["action"] == "removed" for p in document["inventory"]),
            }
        elif args.command == "prepare":
            policy = load_policy(args.policy)
            result = workflow.prepare(
                args.source,
                args.run,
                policy,
                DeferredDetector(policy, args.model, args.rules_only),
                review_sample_size=args.review_sample_size,
            )
        elif args.command == "review":
            state, _ = workflow._load(args.run)
            result = workflow.import_review(
                args.run,
                args.workbook,
                DeferredDetector(state["policy"], args.model, args.rules_only),
            )
        elif args.command == "status":
            result = workflow.status(args.run)
        elif args.command == "export":
            result = workflow.export(args.run, args.out)
        elif args.command == "health-check":
            policy = load_policy(args.policy)
            detector = DeferredDetector(policy, args.model, args.rules_only)
            findings = detector.detect(
                [{"id": "health", "text": "Kontakt: test@example.invalid", "locator": "synthetic"}]
            )
            result = {
                "ok": True,
                "version": __version__,
                "detector": detector.metadata,
                "synthetic_findings": len(findings),
                "network": "offline configuration",
            }
        else:
            if args.out.exists():
                raise AnonymizerError("Sample destination already exists; choose a new directory.")
            args.out.mkdir(parents=True, mode=0o700)
            source = args.out / "synthetic.docx"
            documents.write_docx(
                [
                    {
                        "id": "s000001",
                        "text": "00:00:01 Anna Beispiel: Kontakt anna@example.invalid.",
                        "locator": "p1",
                    },
                    {
                        "id": "s000002",
                        "text": "00:00:05 Max Muster: Ab 100000 Euro erfolgt eine zweite Freigabe.",
                        "locator": "p2",
                    },
                ],
                source,
            )
            policy = load_policy(None)
            policy["aliases"] = [
                {"entity_key": "synthetic-person-1", "aliases": ["Anna Beispiel"]},
                {"entity_key": "synthetic-person-2", "aliases": ["Max Muster"]},
            ]
            result = workflow.prepare(
                source, args.out / "run", policy, DeferredDetector(policy, None, True)
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except AnonymizerError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError):
        print(
            "Error: local input or state could not be processed; check file access and format.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print(
            "Cancelled. No partial revision is approved; use status to inspect committed work.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
