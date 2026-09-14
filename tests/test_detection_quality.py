from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcript_anonymizer.detection import Detector, load_policy
from transcript_anonymizer.errors import AnonymizerError


class _Tokenizer:
    model_max_length = 256

    def __call__(self, text: str, **_: object) -> dict[str, list[tuple[int, int]]]:
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


class _BoundaryModel:
    tokenizer = _Tokenizer()

    def extract_entities(self, text: str, labels: list[str], **kwargs: object) -> dict:
        del labels
        assert kwargs["threshold"] == pytest.approx(0.2)
        if "ALICE" in text:
            start = text.index("ALICE")
            return {"entities": {"person": [{"start": start, "end": start + 5, "score": 0.4}]}}
        return {"entities": {}}


def test_review_threshold_keeps_borderline_model_evidence(tmp_path: Path) -> None:
    policy = load_policy(None)
    policy["model"]["proposal_threshold"] = 0.8
    policy["model"]["review_threshold"] = 0.2
    detector = Detector(policy, model_path=tmp_path)
    detector._model = _BoundaryModel()
    findings = detector.detect([{"id": "s1", "text": "prefix"}, {"id": "s2", "text": "ALICE"}])
    candidate = next(item for item in findings if item["detector"].startswith("model:"))
    assert candidate["segment_id"] == "s2"
    assert candidate["text"] == "ALICE"
    assert candidate["review_only"] is True


def test_structural_speaker_name_propagation_is_document_scoped() -> None:
    policy = load_policy(None)
    detector = Detector(policy, rules_only=True)
    segments = [
        {"id": "s1", "text": "[00:01:02] Anna Müller: Hallo"},
        {"id": "s2", "text": "Anna Müller übernimmt. Müller prüft den Vorgang."},
    ]
    findings = detector.detect(segments)
    full = [item for item in findings if item["text"].casefold() == "anna müller"]
    assert {item["entity_key"] for item in full} and len({item["entity_key"] for item in full}) == 1
    assert all(not item.get("review_only", False) for item in full)
    surname = next(
        item
        for item in findings
        if item["text"] == "Müller" and item["detector"].endswith("suggestion")
    )
    assert surname["review_only"] is True
    other = Detector(policy, rules_only=True).detect([{"id": "s1", "text": "Anna Müller: Hallo"}])
    assert (
        next(item for item in other if item["text"] == "Anna Müller")["entity_key"]
        == next(item for item in full if item["text"] == "Anna Müller")["entity_key"]
    )


def test_structural_extraction_does_not_treat_headers_as_speakers() -> None:
    detector = Detector(load_policy(None), rules_only=True)
    findings = detector.detect(
        [{"id": "s1", "text": "Die IBAN lautet: DE89370400440532013000\nDeutsche Bank: Prozess"}]
    )
    assert not any(item["category"] == "person" for item in findings)


def test_timestamp_after_speaker_identifier_is_a_structural_name() -> None:
    detector = Detector(load_policy(None), rules_only=True)
    findings = detector.detect(
        [
            {"id": "s1", "text": "\nAnna_Muster   0:03\nHallo."},
            {"id": "s2", "text": "\nSprecher_1   0:05\nWeiter."},
        ]
    )
    speaker = next(item for item in findings if item["detector"] == "rule:speaker")
    assert speaker["text"] == "Anna_Muster"
    assert not any(item["text"] == "Sprecher_1" for item in findings)


@pytest.mark.parametrize(
    "model, message",
    [
        ({"review_threshold": 1.1}, "between 0 and 1"),
        ({"proposal_threshold": 0.1, "review_threshold": 0.2}, "cannot exceed"),
        ({"thresholds": {"not_a_label": {"proposal": 0.8, "review": 0.2}}}, "unsupported label"),
    ],
)
def test_threshold_configuration_is_validated(tmp_path: Path, model: dict, message: str) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"schema_version": 1, "model": model}), encoding="utf-8")
    with pytest.raises(AnonymizerError, match=message):
        load_policy(path)


def test_legacy_explicit_threshold_keeps_single_threshold_semantics(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps({"schema_version": 1, "model": {"threshold": 0.6}}), encoding="utf-8"
    )
    policy = load_policy(path)
    assert policy["model"]["proposal_threshold"] == 0.6
    assert policy["model"]["review_threshold"] == 0.6


def test_partial_policy_keeps_new_default_review_threshold(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"schema_version": 1, "aliases": []}), encoding="utf-8")
    assert load_policy(path)["model"]["review_threshold"] == 0.2


def test_new_threshold_fields_have_explicit_precedence(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {
                    "threshold": 0.6,
                    "proposal_threshold": 0.4,
                    "review_threshold": 0.2,
                },
            }
        )
    )
    detector = Detector(load_policy(path), rules_only=True)
    assert detector.policy["model"]["proposal_threshold"] == 0.4
    assert detector.policy["model"]["review_threshold"] == 0.2


def test_threshold_for_unqueried_label_is_rejected(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {
                    "labels": ["person"],
                    "thresholds": {"email": 0.5},
                },
            }
        )
    )
    with pytest.raises(AnonymizerError, match="unsupported label"):
        load_policy(path)


def test_spanless_model_output_is_rejected_instead_of_matching_all_occurrences():
    detector = Detector(load_policy(None), rules_only=True)
    with pytest.raises(AnonymizerError, match="exact offsets"):
        detector._parse_model_result(
            "s1",
            "Anna met Anna",
            0,
            "Anna met Anna",
            {
                "entities": {"person": [{"text": "Anna", "score": 0.9}]},
            },
        )


def test_direct_model_loader_forces_cpu_offline_environment(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import transcript_anonymizer.detection as module

    monkeypatch.setattr(module, "_verify_model_assets", lambda path: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")

    def load(path, **kwargs):
        assert kwargs == {"local_files_only": True, "map_location": "cpu"}
        return "loaded"

    monkeypatch.setitem(
        sys.modules, "gliner2", SimpleNamespace(AutoExtractor=SimpleNamespace(from_pretrained=load))
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None),
    )
    assert module.Detector._load_model(tmp_path) == "loaded"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
