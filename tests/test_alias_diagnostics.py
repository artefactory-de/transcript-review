import json
from pathlib import Path

from transcript_anonymizer.detection import Detector, load_policy

CORPUS_PATH = Path(__file__).parents[1] / "examples" / "evaluation.german.aliases.json"


def _document() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))[0]


def test_rules_only_diagnostics_keep_structural_and_ambiguous_cases_distinct() -> None:
    document = _document()
    findings = Detector(load_policy(None), rules_only=True).detect(document["segments"])

    structural = [item for item in findings if item["detector"] == "rule:speaker"]
    assert {item["text"] for item in structural} == {"Anna Beispiel", "Max_Muster", "Dr. Eva Keller"}
    assert not any(item["text"] == "Anni" for item in findings)

    suggestions = [item for item in findings if item["detector"] == "rule:speaker-alias-suggestion"]
    assert any(item["text"] == "Beispiel" and item.get("review_only") is True for item in suggestions)
    assert not any(item.get("entity_key") for item in suggestions)
    assert not any(item["text"] in {"Müller", "MÜLLER", "müller"} for item in findings)


def test_explicit_aliases_cover_initial_nickname_and_case_without_inference() -> None:
    document = _document()
    policy = load_policy(None)
    policy["aliases"] = [{"entity_key": "synthetic:anna", "aliases": ["Anna Beispiel", "A. Beispiel", "Anni"]}]
    findings = Detector(policy, rules_only=True).detect(document["segments"])

    for value in ("Anna Beispiel", "A. Beispiel", "Anni"):
        matches = [item for item in findings if item["detector"] == "rule:alias" and item["text"].casefold() == value.casefold()]
        assert matches
        assert all(item["entity_key"] == "synthetic:anna" for item in matches)
    assert not any(item["text"] == "Anni" for item in Detector(load_policy(None), rules_only=True).detect(document["segments"]))


def test_alias_diagnostics_preserve_unicode_source_offsets() -> None:
    document = _document()
    policy = load_policy(None)
    policy["aliases"] = [{"entity_key": "synthetic:muller", "aliases": ["Müller"]}]
    findings = Detector(policy, rules_only=True).detect(document["segments"])
    segments = {segment["id"]: segment["text"] for segment in document["segments"]}
    for finding in findings:
        assert segments[finding["segment_id"]][finding["start"] : finding["end"]] == finding["text"]


def test_alias_matching_links_underscore_names_without_inventing_unicode_equivalence() -> None:
    policy = load_policy(None)
    policy["aliases"] = [{"entity_key": "synthetic:max", "aliases": ["Max_Muster", "Muller"]}]
    segments = [{"id": "s1", "text": "Max_Muster sprach. Muster antwortet. Müller bleibt unklar."}]
    findings = Detector(policy, rules_only=True).detect(segments)
    assert any(item["text"] == "Max_Muster" and item["entity_key"] == "synthetic:max" for item in findings)
    assert any(item["text"] == "Muster" and item.get("entity_key") == "synthetic:max" for item in findings)
    assert not any(item["text"] == "Müller" and item.get("entity_key") == "synthetic:max" for item in findings)


def test_ambiguous_shared_surname_suggestion_has_no_identity_key() -> None:
    segments = [{"id": "s1", "text": "Ada Beispiel: Hallo\nBen Beispiel: Guten Tag\nBeispiel prüft."}]
    findings = Detector(load_policy(None), rules_only=True).detect(segments)
    suggestions = [item for item in findings if item["detector"] == "rule:speaker-alias-suggestion"]
    assert suggestions
    assert all(item["text"] == "Beispiel" and item.get("review_only") is True and "entity_key" not in item for item in suggestions)
