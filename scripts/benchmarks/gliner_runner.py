"""Benchmark the original multilingual GLiNER PII checkpoint offline.

This runner is deliberately separate from the production detector.  It combines
the production deterministic rules with the alternative model's PII labels and
emits the same finding shape used by the cached comparison tooling.  It never
downloads a model, trusts remote code, or prints source text.  Run it under
``scripts/memory_guard.py`` and only when the single model-process slot is free.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # ``resource`` is POSIX-only; the runner also supports Windows setup.
    import resource
except ImportError:  # pragma: no cover - exercised on Windows
    resource = None  # type: ignore[assignment]

MODEL_REPOSITORY = "urchade/gliner_multi_pii-v1"
MODEL_REVISION = "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d"
BASE_MODEL_REPOSITORY = "microsoft/mdeberta-v3-base"
BASE_MODEL_REVISION = "a0484667b22365f84929a935b5e50a51f71f159d"
MODEL_CONFIG = "gliner_config.json"
MODEL_WEIGHT = "pytorch_model.bin"
MODEL_LICENSE = "Apache-2.0"

# The runner is a comparator for one immutable snapshot, not a generic model
# launcher.  Keep the expected checksums here so a caller cannot point
# ``--model`` at a different directory while the metadata still claims this
# revision.  README and cache files are intentionally not part of this gate.
PINNED_ASSETS = {
    "config.json": {
        "size": 579,
        "sha256": "bcffcd343dc5efa5ef2d5a58d2b405eed108f01cc45b48d0a907b333ec41801f",
    },
    MODEL_CONFIG: {
        "size": 478,
        "sha256": "2088bee85f50849dea2ab4cf411ab5cf3aaaf7c8a45c23412035978b2bc02801",
    },
    MODEL_WEIGHT: {
        "size": 1155900362,
        "sha256": "3003753fba99e40645cf088c7367a2c6211fc174897dc64f1f9c147c29d18d2d",
    },
    "spm.model": {
        "size": 4305025,
        "sha256": "13c8d666d62a7bc4ac8f040aab68e942c861f93303156cc28f5c7e885d86d6e3",
    },
    "tokenizer_config.json": {
        "size": 52,
        "sha256": "3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d",
    },
}

# The labels below are all direct PII classes from the model card.  Generic
# organization, location, date, amount, product, and process labels are omitted
# so this experiment cannot silently become a confidentiality detector.
PII_LABELS = [
    "person",
    "first name",
    "last name",
    "email",
    "email address",
    "phone number",
    "mobile phone number",
    "landline phone number",
    "address",
    "full address",
    "street address",
    "iban",
    "bank account number",
    "account number",
    "date of birth",
    "username",
    "ip address",
    "tax identification number",
    "national id number",
    "identity document number",
    "passport number",
    "drivers license number",
]


def _label_key(label: str) -> str:
    return " ".join(label.casefold().replace("_", " ").split())


_CATEGORY_BY_LABEL = {
    "person": "person",
    "first name": "person",
    "last name": "person",
    "email": "email",
    "email address": "email",
    "phone number": "phone",
    "mobile phone number": "phone",
    "landline phone number": "phone",
    "address": "address",
    "full address": "address",
    "street address": "address",
    "iban": "iban",
    "bank account number": "account_id",
    "account number": "account_id",
    "date of birth": "date_of_birth",
    "username": "username",
    "ip address": "ip_address",
    "tax identification number": "government_id",
    "national id number": "government_id",
    "identity document number": "government_id",
    "passport number": "government_id",
    "drivers license number": "government_id",
}


def file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def model_manifest(model_path: Path) -> list[dict[str, Any]]:
    """Return deterministic local asset metadata without loading weights."""

    if not model_path.is_dir():
        raise ValueError("Model directory does not exist")
    entries = []
    for path in sorted(model_path.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(model_path).as_posix()
        if relative.startswith(".cache/") or "/.cache/" in relative:
            continue
        size, sha256 = file_sha256(path)
        entries.append({"path": relative, "size": size, "sha256": sha256})
    required = {MODEL_CONFIG, MODEL_WEIGHT, "spm.model", "tokenizer_config.json"}
    if not required.issubset({item["path"] for item in entries}):
        raise ValueError("Model directory lacks the checkpoint or tokenizer assets")
    return entries


def verify_pinned_assets(entries: list[dict[str, Any]]) -> None:
    """Reject a local checkpoint whose required bytes differ from the pin."""

    by_path = {entry.get("path"): entry for entry in entries}
    mismatches = []
    for path, expected in PINNED_ASSETS.items():
        actual = by_path.get(path)
        if (
            actual is None
            or actual.get("size") != expected["size"]
            or actual.get("sha256") != expected["sha256"]
        ):
            mismatches.append(path)
    if mismatches:
        raise ValueError(f"Model assets do not match pinned revision: {', '.join(mismatches)}")


def process_peak_rss_mib() -> int | None:
    """Return this process's RSS high-water mark where the platform exposes it."""

    if resource is None:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def map_entities(
    segment: dict[str, Any],
    entities: list[dict[str, Any]],
    *,
    score_threshold: float,
    chunk_start: int = 0,
    chunk_text: str | None = None,
) -> list[dict[str, Any]]:
    """Map GLiNER output to production categories with checked source offsets."""

    segment_id = segment.get("id")
    text = segment.get("text")
    if not isinstance(segment_id, str) or not isinstance(text, str):
        raise TypeError("Benchmark segment must contain string id and text")
    if chunk_text is None:
        chunk_text = text
    if not isinstance(chunk_text, str) or not 0 <= chunk_start <= len(text):
        raise ValueError("Alternative model chunk is invalid")
    output: list[dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise TypeError("Alternative model returned an invalid entity")
        raw_label = entity.get("label")
        if not isinstance(raw_label, str):
            raise TypeError("Alternative model entity lacks a label")
        label = _label_key(raw_label)
        category = _CATEGORY_BY_LABEL.get(label)
        if category is None:
            # The model can return labels outside the requested set on an API
            # mismatch.  Ignore them rather than widening PII scope silently.
            continue
        score = entity.get("score", entity.get("confidence"))
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("Alternative model entity has an invalid score")
        score = float(score)
        if not 0 <= score <= 1:
            raise ValueError("Alternative model entity score is out of range")
        if score < score_threshold:
            continue
        start = _integer(entity.get("start", entity.get("start_idx")))
        end = _integer(entity.get("end", entity.get("end_idx")))
        if start is None or end is None or not 0 <= start < end <= len(chunk_text):
            raise ValueError("Alternative model entity has an invalid source span")
        value = entity.get("text", entity.get("span", entity.get("value")))
        absolute_start = chunk_start + start
        absolute_end = chunk_start + end
        if absolute_end > len(text):
            raise ValueError("Alternative model entity has an invalid source span")
        if value is not None and (not isinstance(value, str) or value != chunk_text[start:end]):
            raise ValueError("Alternative model entity span does not match source text")
        output.append(
            {
                "id": f"{segment_id}:{absolute_start}:{absolute_end}:{category}:model:gliner:{label}",
                "segment_id": segment_id,
                "start": absolute_start,
                "end": absolute_end,
                "text": text[absolute_start:absolute_end],
                "category": category,
                "score": round(score, 6),
                "detector": f"model:gliner:{label}",
            }
        )
    output.sort(key=lambda item: (item["start"], item["end"], item["category"], item["detector"]))
    return output


def _offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"


def _block_network() -> None:
    def blocked(*_: Any, **__: Any) -> None:
        raise RuntimeError("Network disabled for alternative model benchmark")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked


def _load_model(model_path: Path) -> Any:
    # Imports are intentionally delayed so metadata and unit tests do not load
    # torch or allocate the checkpoint.
    try:
        from gliner import GLiNER  # type: ignore
        from gliner.model import BaseGLiNER  # type: ignore
    except ImportError as exc:
        raise RuntimeError("The isolated gliner benchmark environment is incomplete") from exc
    import torch  # type: ignore

    # The alternative checkpoint is a legacy ``pytorch_model.bin``.  Refuse to
    # load if the installed GLiNER release does not explicitly use the safe
    # weights-only torch loader.  This check avoids relying on pickle behavior.
    loader_source = inspect.getsource(BaseGLiNER._load_state_dict)
    if "weights_only=True" not in loader_source:
        raise RuntimeError("Installed GLiNER loader does not guarantee weights_only loading")

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    model = GLiNER.from_pretrained(
        str(model_path),
        local_files_only=True,
        map_location="cpu",
        strict=True,
        load_tokenizer=True,
    )
    model.eval()
    return model


def _runtime_model(model_path: Path) -> tuple[Path, str | None]:
    """Create a temporary local config adapter when the legacy card omits encoder_config.

    The original checkpoint remains untouched and its hashes are recorded.  The
    adapter adds only the pinned base encoder's JSON config, never its weights;
    the resulting directory is removed by ``run`` after the model is loaded.
    """

    source = json.loads((model_path / MODEL_CONFIG).read_text(encoding="utf-8"))
    if "encoder_config" in source:
        return model_path, None
    base_config_path = model_path / "config.json"
    if not base_config_path.is_file():
        raise RuntimeError("Legacy checkpoint lacks its local base encoder config")
    source["encoder_config"] = json.loads(base_config_path.read_text(encoding="utf-8"))
    # Keep large temporary model assets in an explicitly selected directory.
    runtime_root = Path(os.environ.get("TMPDIR", str(Path.home() / "tmp")))
    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_path = Path(tempfile.mkdtemp(prefix="gliner-runtime-", dir=runtime_root))
    try:
        for path in model_path.iterdir():
            if path.name == ".cache":
                continue
            destination = runtime_path / path.name
            if path.name == MODEL_CONFIG:
                destination.write_text(json.dumps(source, indent=2), encoding="utf-8")
            elif path.is_file():
                try:
                    destination.symlink_to(path)
                except OSError:
                    shutil.copy2(path, destination)
        runtime_config = (runtime_path / MODEL_CONFIG).read_bytes()
        return runtime_path, hashlib.sha256(runtime_config).hexdigest()
    except Exception:
        shutil.rmtree(runtime_path, ignore_errors=True)
        raise


def _chunk_ranges(model: Any, text: str, labels: list[str]) -> list[tuple[int, int]]:
    """Build overlapping character ranges within GLiNER's actual word/token limits."""

    processor = getattr(model, "data_processor", None)
    splitter = getattr(processor, "words_splitter", None)
    tokenizer = getattr(processor, "transformer_tokenizer", None)
    config = getattr(model, "config", None)
    if splitter is None or tokenizer is None or config is None:
        raise RuntimeError("Alternative GLiNER model does not expose tokenizer configuration")
    words = list(splitter(text))
    if not words:
        return []
    max_words = getattr(config, "max_len", None)
    if not isinstance(max_words, int) or max_words <= 0:
        raise RuntimeError("Alternative GLiNER model has no positive max_len")
    limits = [max_words]
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10_000_000:
        limits.append(tokenizer_limit)
    encoder = getattr(getattr(model, "model", None), "text_encoder", None)
    encoder_limit = getattr(getattr(encoder, "config", None), "max_position_embeddings", None)
    if isinstance(encoder_limit, int) and encoder_limit > 0:
        limits.append(encoder_limit)
    token_limit = min(limits)

    def encoded_length(tokens: list[str]) -> int:
        prepared, _ = processor.prepare_inputs([tokens], labels)
        encoded = tokenizer(
            prepared,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        # Hugging Face tokenizers return BatchEncoding, a Mapping rather than a
        # concrete dict.  Treating it as a dict would reject the real tokenizer
        # after the model has loaded, while still accepting lightweight mocks.
        ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if not isinstance(ids, list) or not ids or not isinstance(ids[0], list):
            raise RuntimeError("Alternative tokenizer returned invalid input IDs")
        return len(ids[0])

    ranges: list[tuple[int, int]] = []
    start_index = 0
    overlap_words = min(64, max_words - 1) if max_words > 1 else 0
    while start_index < len(words):
        end_index = min(len(words), start_index + max_words)
        while (
            end_index > start_index
            and encoded_length([item[0] for item in words[start_index:end_index]]) > token_limit
        ):
            end_index -= 1
        if end_index == start_index:
            raise RuntimeError("Alternative tokenizer cannot fit one source token")
        start = words[start_index][1]
        end = words[end_index - 1][2]
        if end_index == len(words):
            end = len(text)
        ranges.append((start, end))
        if end_index == len(words):
            break
        next_start = max(start_index + 1, end_index - overlap_words)
        start_index = next_start
    return ranges


def _model_findings(model: Any, segment: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    text = segment["text"]
    findings: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _chunk_ranges(model, text, PII_LABELS):
        chunk = text[chunk_start:chunk_end]
        # The checkpoint card documents a default threshold of 0.5.  Keep a single
        # explicit threshold for this comparator and record it in metadata; scores
        # are not calibrated across models.
        entities = model.predict_entities(chunk, PII_LABELS, threshold=threshold)
        if not isinstance(entities, list):
            raise TypeError("Alternative model returned an invalid entity collection")
        findings.extend(
            map_entities(
                segment,
                entities,
                score_threshold=threshold,
                chunk_start=chunk_start,
                chunk_text=chunk,
            )
        )
    unique: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for finding in findings:
        key = (finding["start"], finding["end"], finding["category"], finding["detector"])
        if key not in unique or finding["score"] > unique[key]["score"]:
            unique[key] = finding
    return sorted(
        unique.values(),
        key=lambda item: (item["start"], item["end"], item["category"], item["detector"]),
    )


def run(corpus_path: Path, model_path: Path, output_dir: Path, threshold: float) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("Output directory already exists")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    corpus_bytes = corpus_path.read_bytes()
    documents = json.loads(corpus_bytes)
    if not isinstance(documents, list) or not documents:
        raise ValueError("Corpus must be a non-empty JSON list")
    assets = model_manifest(model_path)
    verify_pinned_assets(assets)
    _offline_environment()
    _block_network()
    started = time.monotonic()
    runtime_path, runtime_config_sha256 = _runtime_model(model_path)
    try:
        model = _load_model(runtime_path)
    finally:
        if runtime_path != model_path:
            shutil.rmtree(runtime_path, ignore_errors=True)
    # The isolated benchmark environment intentionally does not install the
    # project package.  Resolve the checked-out source explicitly so the rules
    # comparator uses exactly the production implementation under test.
    source_root = Path(__file__).resolve().parents[2] / "src"
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from transcript_anonymizer.detection import Detector, load_policy

    rules = Detector(load_policy(None), rules_only=True)
    candidates: list[list[dict[str, Any]]] = []
    for document in documents:
        if not isinstance(document, dict) or not isinstance(document.get("segments"), list):
            raise TypeError("Corpus document has invalid segments")
        document_findings = rules.detect(document["segments"])
        for segment in document["segments"]:
            document_findings.extend(_model_findings(model, segment, threshold))
        document_findings.sort(
            key=lambda item: (
                item["segment_id"],
                item["start"],
                item["end"],
                item["category"],
                item["detector"],
            )
        )
        candidates.append(document_findings)
    output_dir.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "base_model_repository": BASE_MODEL_REPOSITORY,
        "base_model_revision": BASE_MODEL_REVISION,
        "model_path": str(model_path),
        "runtime_config_sha256": runtime_config_sha256,
        "model_assets": assets,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "corpus_documents": len(documents),
        "labels": PII_LABELS,
        "threshold": threshold,
        "method": "production rules once per document plus original GLiNER PII labels; overlapping tokenizer-bounded per-segment chunks with tail coverage; CPU; no neighboring context windows",
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("gliner", "torch", "transformers", "tokenizers")
        },
        "elapsed_s": round(time.monotonic() - started, 3),
        # Linux reports ru_maxrss in KiB.  This is the child process's own
        # high-water mark and should be compared with, but does not replace,
        # the outer memory_guard's system measurements.
        "process_peak_rss_mib": process_peak_rss_mib(),
        "network": "socket connect blocked; offline environment enabled",
    }
    (output_dir / "candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "documents": len(documents),
        "findings": sum(len(rows) for rows in candidates),
        "elapsed_s": metadata["elapsed_s"],
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.corpus, args.model, args.output_dir, args.threshold)))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"benchmark failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
