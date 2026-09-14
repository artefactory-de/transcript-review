from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import UserDict
from pathlib import Path

import pytest

_RUNNER_PATH = Path(__file__).parents[1] / "scripts" / "benchmarks" / "gliner_runner.py"
_SPEC = importlib.util.spec_from_file_location("gliner_runner", _RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)
map_entities = _RUNNER.map_entities
model_manifest = _RUNNER.model_manifest
verify_pinned_assets = _RUNNER.verify_pinned_assets
chunk_ranges = _RUNNER._chunk_ranges
runtime_model = _RUNNER._runtime_model


def test_map_entities_preserves_checked_offsets_and_scope() -> None:
    segment = {"id": "s1", "text": "Kontakt anna@example.invalid bei 10.000 EUR."}
    start = segment["text"].index("anna@example.invalid")
    rows = map_entities(
        segment,
        [
            {
                "label": "email",
                "text": "anna@example.invalid",
                "start": start,
                "end": start + 20,
                "score": 0.8,
            },
            {
                "label": "organization",
                "text": "EUR",
                "start": segment["text"].index("EUR"),
                "end": len(segment["text"]),
                "score": 0.99,
            },
        ],
        score_threshold=0.5,
    )
    assert len(rows) == 1
    assert rows[0]["category"] == "email"
    assert segment["text"][rows[0]["start"] : rows[0]["end"]] == rows[0]["text"]


def test_map_entities_rejects_bad_span_and_thresholds() -> None:
    segment = {"id": "s1", "text": "Anna"}
    with pytest.raises(ValueError, match="invalid source span"):
        map_entities(
            segment, [{"label": "person", "start": 1, "end": 99, "score": 0.9}], score_threshold=0.5
        )
    assert (
        map_entities(
            segment, [{"label": "person", "start": 0, "end": 4, "score": 0.4}], score_threshold=0.5
        )
        == []
    )


def test_model_manifest_is_stream_hashed_and_requires_assets(tmp_path: Path) -> None:
    payloads = {
        "gliner_config.json": b"{}",
        "pytorch_model.bin": b"weights",
        "spm.model": b"tokenizer",
        "tokenizer_config.json": b"{}",
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
    entries = model_manifest(tmp_path)
    by_path = {entry["path"]: entry for entry in entries}
    assert by_path["pytorch_model.bin"]["size"] == len(payloads["pytorch_model.bin"])
    assert (
        by_path["pytorch_model.bin"]["sha256"]
        == hashlib.sha256(payloads["pytorch_model.bin"]).hexdigest()
    )


def test_verify_pinned_assets_rejects_changed_required_bytes() -> None:
    entries = [{"path": path, **expected} for path, expected in _RUNNER.PINNED_ASSETS.items()]
    verify_pinned_assets(entries)
    entries[0] = {**entries[0], "size": entries[0]["size"] + 1}
    with pytest.raises(ValueError, match="pinned revision"):
        verify_pinned_assets(entries)


def test_runtime_model_adds_local_encoder_config_without_mutating_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "gliner_config.json").write_text('{"max_len": 384}', encoding="utf-8")
    (model_path / "config.json").write_text('{"model_type": "deberta-v2"}', encoding="utf-8")
    (model_path / "pytorch_model.bin").write_bytes(b"weights")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))

    runtime_path, runtime_sha256 = runtime_model(model_path)
    assert runtime_path != model_path
    assert runtime_path.parent == tmp_path / "scratch"
    runtime_config = json.loads((runtime_path / "gliner_config.json").read_text(encoding="utf-8"))
    assert runtime_config["encoder_config"]["model_type"] == "deberta-v2"
    assert "encoder_config" not in json.loads(
        (model_path / "gliner_config.json").read_text(encoding="utf-8")
    )
    assert (
        runtime_sha256
        == hashlib.sha256((runtime_path / "gliner_config.json").read_bytes()).hexdigest()
    )


def test_chunk_ranges_overlap_and_cover_long_tail() -> None:
    class Splitter:
        def __call__(self, text: str):
            import re

            for match in re.finditer(r"\w+", text):
                yield match.group(), match.start(), match.end()

    class Tokenizer:
        model_max_length = 12

        def __call__(self, prepared, **_: object):
            # One token per source word plus a fixed prompt budget.
            return UserDict({"input_ids": [list(range(len(prepared[0]) + 2))]})

    class Processor:
        words_splitter = Splitter()
        transformer_tokenizer = Tokenizer()

        @staticmethod
        def prepare_inputs(texts, labels):
            return [[*labels, *texts[0]]], [len(labels)]

    class Config:
        max_len = 8

    class Model:
        data_processor = Processor()
        config = Config()
        model = object()

    text = " ".join(f"word{i}" for i in range(30))
    ranges = chunk_ranges(Model(), text, ["person", "email"])
    assert len(ranges) > 1
    assert ranges[-1][1] == len(text)
    covered = set()
    for start, end in ranges:
        covered.update(range(start, end))
    assert all(index in covered for index, char in enumerate(text) if not char.isspace())
