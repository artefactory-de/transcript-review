from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import transcript_anonymizer.detection as detection_module
from transcript_anonymizer.detection import Detector, load_policy, rank_passages
from transcript_anonymizer.errors import AnonymizerError


def _policy(*, aliases: list[dict] | None = None) -> dict:
    policy = load_policy(None)
    if aliases is not None:
        policy["aliases"] = aliases
    return policy


def test_split_model_assets_reassemble_and_remove_parts(tmp_path, monkeypatch):
    model = tmp_path / 'model'
    model.mkdir()
    payload = b'0123456789abcdef'
    parts = [payload[:8], payload[8:]]
    parts_root = tmp_path / 'model-parts'
    parts_root.mkdir()
    records = []
    for index, part in enumerate(parts, 1):
        name = f'model.safetensors.part{index:03d}'
        (parts_root / name).write_bytes(part)
        records.append({'name': name, 'bytes': len(part), 'sha256': hashlib.sha256(part).hexdigest()})
    (tmp_path / 'model-parts.json').write_text(json.dumps({
        'format': 1, 'target': 'model.safetensors', 'bytes': len(payload),
        'sha256': hashlib.sha256(payload).hexdigest(), 'parts': records,
    }), encoding='utf-8')
    detection_module._assemble_model_parts(model)
    assert (model / 'model.safetensors').read_bytes() == payload
    assert not parts_root.exists()
    assert not (tmp_path / 'model-parts.json').exists()


def test_split_model_assets_accept_legacy_release_root_parts(tmp_path):
    """v0.2.4 published its part folders beside, rather than inside, _internal."""
    internal = tmp_path / "_internal"
    model = internal / "model"
    model.mkdir(parents=True)
    payload = b"0123456789abcdef"
    part = tmp_path / "model-parts" / "model.safetensors.part001"
    part.parent.mkdir()
    part.write_bytes(payload)
    (internal / "model-parts.json").write_text(json.dumps({
        "format": 1, "target": "model.safetensors", "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "parts": [{"name": part.name, "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest()}],
    }), encoding="utf-8")
    detection_module._assemble_model_parts(model)
    assert (model / "model.safetensors").read_bytes() == payload
    assert not part.parent.exists()


def test_rules_only_detects_direct_pii_and_preserves_alias_key() -> None:
    policy = _policy(aliases=[{"entity_key": "employee:anna", "aliases": ["Anna Müller", "A. Müller"]}])
    detector = Detector(policy, rules_only=True)
    text = (
        "Anna Müller ist unter anna.mueller@example.de oder +49 30 1234567 erreichbar. "
        "Kundennummer: K-123456, Depot-Nr. D98765, IBAN DE89370400440532013000. "
        "Termin 2026-09-13, Budget 42.000 EUR, System Atlas."
    )

    findings = detector.detect([{"id": "s1", "text": text}])
    assert all(set(item) <= {"id", "segment_id", "start", "end", "text", "category", "score", "detector", "entity_key"} for item in findings)
    assert {item["category"] for item in findings} >= {"person", "email", "phone", "customer_id", "depot_id", "iban"}
    assert not any(item["text"] == "2026-09-13" for item in findings)
    assert not any(item["text"] in {"Atlas", "42.000 EUR"} for item in findings)
    alias = next(item for item in findings if item["detector"] == "rule:alias")
    assert alias["entity_key"] == "employee:anna"
    for item in findings:
        assert text[item["start"] : item["end"]] == item["text"]
    assert detector.metadata["model"]["status"] == "disabled"


def test_contextual_identifiers_require_identifier_like_values() -> None:
    detector = Detector(_policy(), rules_only=True)
    text = "Depot wird eröffnet. Depot prüfen. Kundennummer wird erklärt. Konto bleibt leer."
    assert detector.detect([{"id": "s1", "text": text}]) == []


def test_large_business_numbers_are_not_phone_candidates() -> None:
    detector = Detector(_policy(), rules_only=True)
    text = "Umsatz 10000000 Euro, Volumen 100000000, Zeit 2026-09-13 10:30:00."
    assert detector.detect([{"id": "s1", "text": text}]) == []


@pytest.mark.parametrize("punctuation", [".", ",", ";", ")"])
def test_email_rule_stops_before_sentence_punctuation(punctuation: str) -> None:
    text = f"Kontakt anna@example.invalid{punctuation}"
    findings = Detector(_policy(), rules_only=True).detect([{"id": "s1", "text": text}])
    assert [(item["text"], item["category"]) for item in findings] == [("anna@example.invalid", "email")]


def test_default_policy_is_independent_and_policy_json_is_validated(tmp_path: Path) -> None:
    first = load_policy(None)
    first["aliases"].append({"entity_key": "x", "aliases": ["X"]})
    assert load_policy(None)["aliases"] == []

    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps({"schema_version": 1, "aliases": {"employee:one": ["Erika Beispiel"]}}),
        encoding="utf-8",
    )
    loaded = load_policy(path)
    assert loaded["aliases"] == [{"entity_key": "employee:one", "aliases": ["Erika Beispiel"]}]

    path.write_text(json.dumps({"schema_version": 1, "model": {"labels": ["company"]}}), encoding="utf-8")
    with pytest.raises(AnonymizerError):
        load_policy(path)

    path.write_text(json.dumps({"schema_version": 1, "modle": {}}), encoding="utf-8")
    with pytest.raises(AnonymizerError):
        load_policy(path)


def test_non_rules_mode_is_lazy_but_never_silently_falls_back() -> None:
    detector = Detector(_policy(), model_path=None)
    assert detector.metadata["model"]["status"] == "configured"
    with pytest.raises(AnonymizerError, match="model_path"):
        detector.detect([{"id": "s1", "text": "Email a@example.de"}])


class _CharTokenizer:
    model_max_length = 128

    def __call__(self, text: str, **_: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


class _MockModel:
    tokenizer = _CharTokenizer()

    def extract_entities(self, text: str, labels: list[str], **_: object) -> dict:
        del labels
        if "TAILPII" in text:
            start = text.index("TAILPII")
            return {"entities": {"person": [{"text": "TAILPII", "start": start, "end": start + 7, "score": 0.41}]}}
        return {"entities": {}}


def test_model_adapter_chunks_to_token_budget_and_covers_tail(tmp_path: Path) -> None:
    detector = Detector(_policy(), model_path=tmp_path, rules_only=False)
    detector._model = _MockModel()
    metadata = copy.deepcopy(detector.metadata)
    text = "prefix " * 9 + "TAILPII"
    findings = detector.detect([{"id": "s1", "text": text}])
    assert [(item["text"], item["start"], item["end"]) for item in findings if item["detector"].startswith("model:")] == [
        ("TAILPII", text.index("TAILPII"), text.index("TAILPII") + 7)
    ]
    assert detector.metadata == metadata


def test_model_chunk_budget_reserves_schema_tokens_and_encoder_limit(tmp_path: Path) -> None:
    class Tokenizer(_CharTokenizer):
        model_max_length = 1000

        def __call__(self, text: str, **kwargs: object) -> dict:
            if text in _policy()["model"]["labels"]:
                return {"input_ids": [1, 2, 3]}
            return super().__call__(text, **kwargs)

    class Encoder:
        class Config:
            max_position_embeddings = 80

        config = Config()

    class Model(_MockModel):
        tokenizer = Tokenizer()
        encoder = Encoder()

    detector = Detector(_policy(), model_path=tmp_path)
    detector._model = Model()
    chunks = detector._chunks("x " * 200)
    assert max(end - start for start, end in chunks) < len("x " * 200)
    assert detector._schema_token_reserve(detector._model.tokenizer) >= 32


def test_local_model_assets_are_verified_before_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"config"
    item = {"path": "config.json", "size": len(payload), "sha256": __import__("hashlib").sha256(payload).hexdigest()}
    manifest = {"schema_version": 1, "files": [item]}
    monkeypatch.setattr(detection_module, "_asset_manifest", lambda: (manifest, "fingerprint"))
    (tmp_path / "config.json").write_bytes(payload)
    detection_module._verify_model_assets(tmp_path)
    (tmp_path / "config.json").write_bytes(b"changed")
    with pytest.raises(AnonymizerError):
        detection_module._verify_model_assets(tmp_path)


class _FailingModel:
    tokenizer = _CharTokenizer()

    def extract_entities(self, *_: object, **__: object) -> dict:
        raise RuntimeError("model failed")


def test_model_failure_is_not_converted_to_rules_only(tmp_path: Path) -> None:
    detector = Detector(_policy(), model_path=tmp_path)
    detector._model = _FailingModel()
    with pytest.raises(AnonymizerError, match="inference failed"):
        detector.detect([{"id": "s1", "text": "person@example.de"}])


def test_identifier_does_not_consume_sentence_punctuation() -> None:
    detector = Detector(load_policy(None), rules_only=True)
    findings = detector.detect([{"id": "s1", "text": "Depotnummer: D9876543."}])
    assert [(finding["text"], finding["category"]) for finding in findings] == [("D9876543", "depot_id")]


def test_rank_passages_covers_clean_and_partially_detected_segments() -> None:
    segments = [
        {"id": "s1", "text": "Kunde nennt keinen direkten Wert, aber seine Telefonnummer."},
        {"id": "s2", "text": "Budget 50.000 EUR und System Atlas bleiben erhalten."},
        {"id": "s3", "text": "Kunde: email anna@example.de"},
    ]
    findings = [
        {"segment_id": "s3", "start": 14, "end": 31, "score": 0.55},
        {"segment_id": "s3", "start": 14, "end": 31, "score": 0.8},
    ]
    ranked = rank_passages(segments, findings)
    assert {item["id"] for item in ranked} == {"s1", "s2", "s3"}
    assert ranked[0]["id"] == "s3"
    assert "overlapping_candidates" in ranked[0]["reasons"]
    assert any(item["id"] == "s2" and item["reasons"] == ["no_candidate_signal"] for item in ranked)
    assert all(0 <= item["score"] <= 1 for item in ranked)
