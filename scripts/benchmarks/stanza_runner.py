#!/usr/bin/env python3
"""Compare Stanza German NER packages with production deterministic rules offline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

MODEL_REPOSITORY = "stanfordnlp/stanza-de"
MODEL_REVISION = "47008d32ed8ae28bd955e74984230617eccfd4c3"
MODEL_LICENSE = "Apache-2.0"
MODEL_SOURCE_URL = f"https://huggingface.co/{MODEL_REPOSITORY}/tree/{MODEL_REVISION}"
MODEL_RESOURCES_VERSION = "1.14.0"
MAX_CORPUS_BYTES = 16 * 1024 * 1024
MAX_DOCUMENTS = 1_000
MAX_SEGMENTS_PER_DOCUMENT = 10_000
MAX_TEXT_CHARS_PER_DOCUMENT = 2_000_000
PINNED_ASSET_SHA256 = {
    "resources.json": "4e41c1df152146fa26ed0c006a08feea7a60bb3414bb6d57dbda24ad2e3cb99c",
    "de/tokenize/gsd.pt": "828a0a738ec5828d68f3b4926ff4a5ef469d84d9d43cbc1ccb4e504b8cd65a53",
    "de/mwt/gsd.pt": "7b93f089f1df31150c1489fbff45e7809d22148d5ce1be95480d457efd063f03",
    "de/ner/conll03.pt": "1773447e10ce4ae7525cd52bfba5e90eb177a5f34ef97c84c3b8ed427853b9aa",
    "de/ner/germeval2014.pt": "7c819f5d28c7546a2cb50c49e4ef1cdeb635f6547a2547414eaec5b3d95ff164",
    "de/pretrain/fasttextwiki.pt": "c5a3cce2b51a8dbb38522a6b0365a193839a19f0647cbdecc62aac715c3cdecf",
    "de/forward_charlm/newswiki.pt": "ba0ebfb852329cee8f13d9a0d0a30b4bdab7d95587029e78bd90c2a0dfe22248",
    "de/backward_charlm/newswiki.pt": "447e419d87d0623f6ae5a84021472ea8024d6a4b6803566aa8a4f5a80ef598e4",
}


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _offline_environment() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"


def _block_network() -> None:
    def blocked(*_: Any, **__: Any) -> None:
        raise RuntimeError("network access is disabled for the offline benchmark")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked


def _load_corpus(path: Path) -> list[dict[str, Any]]:
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
        if not isinstance(document, dict) or not isinstance(document.get("id"), str) or document["id"] in seen:
            raise ValueError("benchmark document ids must be unique strings")
        segments = document.get("segments")
        if not isinstance(segments, list) or len(segments) > MAX_SEGMENTS_PER_DOCUMENT:
            raise ValueError("benchmark document has an invalid segment list")
        total_chars = 0
        normalized: list[dict[str, str]] = []
        segment_ids: set[str] = set()
        for segment in segments:
            if not isinstance(segment, dict) or not isinstance(segment.get("id"), str) or segment["id"] in segment_ids:
                raise ValueError("benchmark segment ids must be unique strings")
            if not isinstance(segment.get("text"), str):
                raise TypeError("benchmark segment text must be a string")
            segment_ids.add(segment["id"])
            total_chars += len(segment["text"])
            if total_chars > MAX_TEXT_CHARS_PER_DOCUMENT:
                raise ValueError("benchmark document text exceeds the bounded input limit")
            normalized.append({"id": segment["id"], "text": segment["text"]})
        seen.add(document["id"])
        documents.append({"id": document["id"], "segments": normalized})
    return documents


def _asset_manifest(model_root: Path) -> list[dict[str, Any]]:
    if not model_root.is_dir():
        raise ValueError("Stanza model root does not exist")
    entries: list[dict[str, Any]] = []
    for path in sorted(model_root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                size += len(block)
                digest.update(block)
        entries.append({"path": path.relative_to(model_root).as_posix(), "size": size, "sha256": digest.hexdigest()})
    required = {
        "resources.json",
        # Stanza's loader resolves paths as model_root/de/{processor}/...;
        # the downloaded Hugging Face tree is retained under de/models and
        # the benchmark setup supplies symlinks at these runtime paths.
        "de/tokenize/gsd.pt",
        "de/mwt/gsd.pt",
        "de/pretrain/fasttextwiki.pt",
        "de/forward_charlm/newswiki.pt",
        "de/backward_charlm/newswiki.pt",
        "de/ner/conll03.pt",
        "de/ner/germeval2014.pt",
    }
    if not required.issubset({entry["path"] for entry in entries}):
        raise ValueError("Stanza model root lacks one or more pinned benchmark assets")
    return entries


def _verify_pinned_assets(entries: list[dict[str, Any]]) -> None:
    actual = {entry["path"]: entry["sha256"] for entry in entries}
    mismatches = [path for path, digest in PINNED_ASSET_SHA256.items() if actual.get(path) != digest]
    if mismatches:
        raise ValueError("Stanza model root does not contain the pinned Stanford assets: " + ", ".join(mismatches))


@contextmanager
def _trusted_numpy_checkpoint_globals(torch: Any) -> Any:
    """Allow only NumPy classes required by the pinned Stanza files."""
    import numpy as np
    from numpy.core.multiarray import _reconstruct

    dtype_classes = [type(np.dtype(name)) for name in ("float32", "float64", "int64")]
    with torch.serialization.safe_globals([
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        *dtype_classes,
    ]):
        yield


def _finding(segment: dict[str, str], start: int, end: int, detector: str) -> dict[str, Any]:
    text = segment["text"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("alternative model returned an invalid source span")
    return {
        "id": f"{segment['id']}:{start}:{end}:person:{detector}",
        "segment_id": segment["id"],
        "start": start,
        "end": end,
        "text": text[start:end],
        "category": "person",
        "score": 1.0,
        "detector": detector,
    }


def _stanza_findings(pipeline: Any, documents: list[dict[str, Any]], package: str) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for document in documents:
        findings: list[dict[str, Any]] = []
        for segment in document["segments"]:
            parsed = pipeline(segment["text"])
            for entity in parsed.entities:
                if entity.type != "PER":
                    continue
                start = entity.start_char
                end = entity.end_char
                # Do not coerce model output: coercion can silently turn malformed
                # float/bool offsets into plausible source spans.  Stanza's spans
                # are integer character offsets and its text must map exactly to
                # the segment supplied to the pipeline.
                if type(start) is not int or type(end) is not int:
                    raise ValueError("alternative model returned non-integer source offsets")
                entity_text = getattr(entity, "text", None)
                if not isinstance(entity_text, str) or entity_text != segment["text"][start:end]:
                    raise ValueError("alternative model entity text does not match its source span")
                findings.append(_finding(segment, start, end, f"stanza:{package}"))
        rows.append(findings)
    return rows


def _run_rules(documents: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], float]:
    from transcript_anonymizer.detection import Detector, load_policy

    detector = Detector(load_policy(None), rules_only=True)
    started = time.perf_counter()
    rows = [detector.detect(document["segments"]) for document in documents]
    return rows, time.perf_counter() - started


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run(corpora: list[Path], output_dir: Path, model_root: Path, package: str) -> dict[str, Any]:
    _offline_environment()
    _block_network()
    if package not in {"conll03", "germeval2014"}:
        raise ValueError("Stanza package is not pinned for this benchmark")
    if output_dir.exists():
        raise ValueError("benchmark output directory already exists; choose a fresh cache path")
    documents: list[dict[str, Any]] = []
    corpus_records: list[dict[str, Any]] = []
    lengths: list[int] = []
    for corpus_path in corpora:
        corpus_documents = _load_corpus(corpus_path)
        documents.extend(corpus_documents)
        lengths.append(len(corpus_documents))
        corpus_records.append({"path": str(corpus_path), "sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(), "documents": len(corpus_documents)})
    output_dir.mkdir(parents=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    try:
        import stanza  # type: ignore
        import torch  # type: ignore
        from stanza.pipeline.core import DownloadMethod  # type: ignore
    except ImportError as exc:
        raise RuntimeError("the isolated Stanza benchmark environment is incomplete") from exc
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    assets = _asset_manifest(model_root)
    _verify_pinned_assets(assets)
    started = time.perf_counter()
    with _trusted_numpy_checkpoint_globals(torch):
        pipeline = stanza.Pipeline(lang="de", processors={"tokenize": "gsd", "ner": package}, package=None, dir=str(model_root), download_method=DownloadMethod.NONE, use_gpu=False, device="cpu", verbose=False)
    load_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    stanza_rows = _stanza_findings(pipeline, documents, package)
    inference_seconds = time.perf_counter() - inference_started
    rules_rows, rules_seconds = _run_rules(documents)
    _write_json(output_dir / "stanza_candidates.json", stanza_rows)
    _write_json(output_dir / "rules_candidates.json", rules_rows)
    cursor = 0
    for corpus_path, length in zip(corpora, lengths, strict=True):
        stem = corpus_path.stem.replace("/", "_")
        _write_json(output_dir / f"stanza_candidates.{stem}.json", stanza_rows[cursor : cursor + length])
        _write_json(output_dir / f"rules_candidates.{stem}.json", rules_rows[cursor : cursor + length])
        cursor += length
    runtime = {"python": sys.version.split()[0], "stanza": getattr(stanza, "__version__", None), "torch": getattr(torch, "__version__", None)}
    for package_name in ("stanza", "torch"):
        try:
            runtime[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            pass
    resources_manifest = next(entry for entry in assets if entry["path"] == "resources.json")
    metrics = {
        "schema_version": 1,
        "model": {"repository": MODEL_REPOSITORY, "weights_revision": MODEL_REVISION, "license": MODEL_LICENSE, "source_url": MODEL_SOURCE_URL, "package": package, "detector": f"stanza:{package}", "score_semantics": "fixed 1.0 for PER spans; Stanza does not expose calibrated confidence", "resources_manifest_version": MODEL_RESOURCES_VERSION, "resources_manifest_sha256": resources_manifest["sha256"], "pipeline_config": {"lang": "de", "processors": {"tokenize": "gsd", "ner": package}, "download_method": "NONE", "device": "cpu", "torch_threads": 2}},
        "assets": assets,
        "runtime": runtime,
        "corpora": corpus_records,
        "documents": len(documents),
        "stanza": {"candidate_count": sum(len(row) for row in stanza_rows), "load_seconds": load_seconds, "inference_seconds": inference_seconds},
        "production_rules": {"candidate_count": sum(len(row) for row in rules_rows), "seconds": rules_seconds},
        "peak_rss_bytes": _rss_bytes(),
        "network": "socket connect/connect_ex/create_connection blocked; offline environment enforced",
        "caveat": "Synthetic German text is a smoke comparison, not representative calibration or acceptance evidence.",
    }
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--package", choices=("conll03", "germeval2014"), required=True)
    args = parser.parse_args(argv)
    try:
        _offline_environment()
        _block_network()
        metrics = run(args.corpus, args.output_dir, args.model_root, args.package)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Stanza benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir), "documents": metrics["documents"], "candidates": metrics["stanza"]["candidate_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
