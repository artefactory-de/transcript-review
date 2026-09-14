"""Benchmark the German spaCy PER baseline alongside the production rules.

This runner is deliberately separate from the application detector.  It is an
offline experiment: it never changes the configured production model and it
writes only candidate caches (one finding list per document) and aggregate
resource metrics.  spaCy NER does not expose calibrated confidences, so its
candidate score is a fixed ``1.0`` and must not be read as a probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import socket
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Verified against the official spaCy model metadata.  Keep these URLs in the
# experiment record so a later comparison does not lose the exact artifact.
MODEL_PACKAGE = "de_core_news_sm"
MODEL_VERSION = "3.8.0"
MODEL_LICENSE = "MIT"
MODEL_SOURCE_URL = "https://spacy.io/models/de"
MODEL_METADATA_URL = (
    "https://raw.githubusercontent.com/explosion/spacy-models/master/meta/"
    "de_core_news_sm-3.8.0.json"
)
MODEL_DETECTOR = f"spacy:{MODEL_PACKAGE}@{MODEL_VERSION}"

DEFAULT_CORPORA = (
    Path("examples/evaluation.german.synthetic.json"),
    Path("examples/evaluation.german.challenge.json"),
)
MAX_CORPUS_BYTES = 16 * 1024 * 1024
MAX_DOCUMENTS = 1_000
MAX_SEGMENTS_PER_DOCUMENT = 10_000
MAX_TEXT_CHARS_PER_DOCUMENT = 2_000_000


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def _rss_bytes() -> int:
    # Linux reports KiB.  Keep the conversion here rather than depending on a
    # third-party process monitor in the isolated benchmark environment.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _block_network() -> None:
    """Fail closed if a package tries to fetch an artifact during benchmarking."""

    def blocked(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("network access is disabled for the offline benchmark")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked


def _offline_environment() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"


def _finding_id(segment_id: str, start: int, end: int) -> str:
    digest = hashlib.sha256(f"{segment_id}\x1f{start}\x1f{end}\x1fperson".encode()).hexdigest()
    return f"{MODEL_DETECTOR}:person:{digest[:20]}"


def _segments(document: dict[str, Any]) -> list[dict[str, str]]:
    raw_segments = document.get("segments")
    if not isinstance(document.get("id"), str) or not document["id"]:
        raise ValueError("each benchmark document needs a non-empty id")
    if not isinstance(raw_segments, list) or len(raw_segments) > MAX_SEGMENTS_PER_DOCUMENT:
        raise ValueError("benchmark document has an invalid segment list")
    result: list[dict[str, str]] = []
    total_chars = 0
    seen: set[str] = set()
    for segment in raw_segments:
        if not isinstance(segment, dict):
            raise TypeError("benchmark segments must be objects")
        segment_id, text = segment.get("id"), segment.get("text")
        if not isinstance(segment_id, str) or not segment_id or segment_id in seen:
            raise ValueError("benchmark segment ids must be unique non-empty strings")
        if not isinstance(text, str):
            raise TypeError("benchmark segment text must be a string")
        seen.add(segment_id)
        total_chars += len(text)
        if total_chars > MAX_TEXT_CHARS_PER_DOCUMENT:
            raise ValueError("benchmark document text exceeds the bounded input limit")
        result.append({"id": segment_id, "text": text})
    return result


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Read a bounded annotated corpus without retaining annotation text in logs."""

    try:
        if path.stat().st_size > MAX_CORPUS_BYTES:
            raise ValueError("benchmark corpus exceeds the bounded input limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("could not read benchmark corpus JSON") from exc
    if not isinstance(payload, list) or len(payload) > MAX_DOCUMENTS:
        raise ValueError("benchmark corpus must be a bounded list of documents")
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document in payload:
        if not isinstance(document, dict):
            raise TypeError("benchmark documents must be objects")
        document_id = document.get("id")
        if not isinstance(document_id, str) or document_id in seen:
            raise ValueError("benchmark document ids must be unique strings")
        _segments(document)
        seen.add(document_id)
        documents.append(document)
    return documents


def _spacy_findings(nlp: Any, documents: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Run only the spaCy ``PER`` entity label and return document-grouped rows."""

    rows: list[list[dict[str, Any]]] = []
    for document in documents:
        segments = _segments(document)
        findings: list[dict[str, Any]] = []
        # Segments are independent source units.  Keeping them separate avoids
        # manufacturing offsets for entities that cross a source boundary.
        for segment, parsed in zip(segments, nlp.pipe((item["text"] for item in segments), batch_size=32)):
            for entity in parsed.ents:
                if entity.label_ != "PER":
                    continue
                start, end = int(entity.start_char), int(entity.end_char)
                if not 0 <= start < end <= len(segment["text"]):
                    raise ValueError("spaCy returned an out-of-bounds entity")
                findings.append(
                    {
                        "id": _finding_id(segment["id"], start, end),
                        "segment_id": segment["id"],
                        "start": start,
                        "end": end,
                        "text": segment["text"][start:end],
                        "category": "person",
                        "score": 1.0,
                        "detector": MODEL_DETECTOR,
                    }
                )
        rows.append(findings)
    return rows


def _rules_findings(detector: Any, documents: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for document in documents:
        findings = detector.detect(_segments(document))
        rows.append(findings)
    return rows


def _word_count(documents: Iterable[dict[str, Any]]) -> int:
    return sum(len(segment["text"].split()) for document in documents for segment in _segments(document))


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run(corpora: list[Path], output_dir: Path, model_name: str = MODEL_PACKAGE) -> dict[str, Any]:
    """Run the benchmark.  Loading the spaCy model happens only when called."""

    if model_name != MODEL_PACKAGE:
        raise ValueError(f"this benchmark is pinned to {MODEL_PACKAGE} {MODEL_VERSION}")
    _offline_environment()
    _block_network()
    if output_dir.exists():
        raise ValueError("benchmark output directory already exists; choose a fresh cache path")
    documents: list[dict[str, Any]] = []
    corpus_records: list[dict[str, Any]] = []
    corpus_lengths: list[int] = []
    all_ids: set[str] = set()
    for corpus_path in corpora:
        corpus_documents = load_corpus(corpus_path)
        # IDs are namespaced in output, but duplicate IDs within one benchmark
        # would make later union/evaluation ambiguous.
        for document in corpus_documents:
            key = f"{corpus_path.resolve()}\x1f{document['id']}"
            if key in all_ids:
                raise ValueError("duplicate document id in benchmark corpus")
            all_ids.add(key)
            documents.append(document)
        corpus_records.append(
            {
                "path": str(corpus_path),
                "sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                "documents": len(corpus_documents),
                "words": _word_count(corpus_documents),
            }
        )
        corpus_lengths.append(len(corpus_documents))

    output_dir.mkdir(parents=True)
    sys.path.insert(0, str(_repo_src()))
    from importlib.metadata import PackageNotFoundError, version

    import spacy  # type: ignore

    from transcript_anonymizer.detection import Detector, load_policy

    try:
        installed_version = version(MODEL_PACKAGE)
    except PackageNotFoundError as exc:
        raise RuntimeError("the pinned spaCy model package is not installed") from exc
    if installed_version != MODEL_VERSION:
        raise RuntimeError(
            f"installed {MODEL_PACKAGE} version {installed_version} does not match {MODEL_VERSION}"
        )

    before = _rss_bytes()
    load_started = time.perf_counter()
    nlp = spacy.load(model_name)
    loaded_version = nlp.meta.get("version")
    if loaded_version != MODEL_VERSION:
        raise RuntimeError("loaded spaCy pipeline metadata does not match the pinned model")
    model_load_seconds = time.perf_counter() - load_started
    inference_started = time.perf_counter()
    spacy_rows = _spacy_findings(nlp, documents)
    spacy_seconds = time.perf_counter() - inference_started
    spacy_peak = max(before, _rss_bytes())

    rules_detector = Detector(load_policy(None), rules_only=True)
    rules_started = time.perf_counter()
    rules_rows = _rules_findings(rules_detector, documents)
    rules_seconds = time.perf_counter() - rules_started

    _write_json(output_dir / "spacy_candidates.json", spacy_rows)
    _write_json(output_dir / "rules_candidates.json", rules_rows)
    cursor = 0
    for record, length, corpus_path in zip(corpus_records, corpus_lengths, corpora, strict=True):
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", corpus_path.stem).strip("._") or "corpus"
        spacy_cache = f"spacy_candidates.{safe_stem}.json"
        rules_cache = f"rules_candidates.{safe_stem}.json"
        _write_json(output_dir / spacy_cache, spacy_rows[cursor : cursor + length])
        _write_json(output_dir / rules_cache, rules_rows[cursor : cursor + length])
        record["spacy_cache"] = spacy_cache
        record["rules_cache"] = rules_cache
        cursor += length
    metrics = {
        "schema_version": 1,
        "model": {
            "package": MODEL_PACKAGE,
            "version": MODEL_VERSION,
            "runtime_version": getattr(spacy, "__version__", None),
            "license": MODEL_LICENSE,
            "source_url": MODEL_SOURCE_URL,
            "metadata_url": MODEL_METADATA_URL,
            "detector": MODEL_DETECTOR,
            "score_semantics": "fixed 1.0; spaCy NER does not expose calibrated confidence",
        },
        "corpora": corpus_records,
        "documents": len(documents),
        "spacy": {
            "candidate_count": sum(len(row) for row in spacy_rows),
            "model_load_seconds": model_load_seconds,
            "inference_seconds": spacy_seconds,
            "peak_rss_bytes": spacy_peak,
        },
        "production_rules": {
            "candidate_count": sum(len(row) for row in rules_rows),
            "seconds": rules_seconds,
            "peak_rss_bytes": _rss_bytes(),
        },
        "caveat": "Synthetic German text is a smoke comparison, not representative calibration or acceptance evidence.",
    }
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", help="JSON corpus (repeatable; defaults to both German fixtures)")
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh directory for candidate caches and metrics")
    parser.add_argument("--spacy-model", default=MODEL_PACKAGE, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        metrics = run(args.corpus or list(DEFAULT_CORPORA), args.output_dir, args.spacy_model)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"spaCy benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir), "documents": metrics["documents"], "spacy_candidates": metrics["spacy"]["candidate_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
