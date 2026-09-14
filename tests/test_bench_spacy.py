from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("bench_spacy", ROOT / "scripts" / "bench_spacy.py")
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class _Entity:
    def __init__(self, label: str, start: int, end: int):
        self.label_ = label
        self.start_char = start
        self.end_char = end


class _Doc:
    def __init__(self, entities):
        self.ents = entities


class _Nlp:
    def pipe(self, texts, batch_size=32):
        del batch_size
        return (_Doc([_Entity("PER", 0, 4), _Entity("ORG", 5, 9)]) for _ in texts)


def test_spacy_cache_is_one_finding_list_per_document_and_per_only():
    documents = [{"id": "synthetic", "segments": [{"id": "s1", "text": "Anna AG"}]}]

    cache = bench._spacy_findings(_Nlp(), documents)

    assert isinstance(cache, list)
    assert len(cache) == 1
    assert len(cache[0]) == 1
    finding = cache[0][0]
    assert set(finding) == {
        "id",
        "segment_id",
        "start",
        "end",
        "text",
        "category",
        "score",
        "detector",
    }
    assert finding["text"] == "Anna"
    assert finding["category"] == "person"
    assert finding["score"] == 1.0
    assert finding["detector"] == "spacy:de_core_news_sm@3.8.0"
    assert "Anna" not in finding["id"]


def test_checked_in_corpora_have_bounded_document_order():
    documents = bench.load_corpus(ROOT / "examples" / "evaluation.german.synthetic.json")

    assert documents
    assert all(document["id"] for document in documents)
    assert all(document["segments"] for document in documents)


def test_model_record_is_pinned_to_primary_metadata():
    assert bench.MODEL_PACKAGE == "de_core_news_sm"
    assert bench.MODEL_VERSION == "3.8.0"
    assert bench.MODEL_LICENSE == "MIT"
    assert bench.MODEL_METADATA_URL.endswith("de_core_news_sm-3.8.0.json")


def test_direct_run_installs_offline_environment_and_network_blocker(monkeypatch, tmp_path):
    import os
    import socket

    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
    monkeypatch.setattr(
        bench, "_offline_environment", lambda: os.environ.__setitem__("SPACY_TEST_OFFLINE", "1")
    )
    monkeypatch.setattr(
        bench, "_block_network", lambda: os.environ.__setitem__("SPACY_TEST_BLOCKED", "1")
    )
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="output directory"):
        bench.run([], tmp_path)
    assert os.environ["SPACY_TEST_OFFLINE"] == "1"
    assert os.environ["SPACY_TEST_BLOCKED"] == "1"
    assert (socket.socket.connect, socket.socket.connect_ex, socket.create_connection) == originals
