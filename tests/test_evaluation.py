import json
from pathlib import Path

import pytest

from transcript_anonymizer.errors import AnonymizerError
from transcript_anonymizer.evaluation import evaluate


class FixedDetector:
    def __init__(self, findings):
        self.findings = findings
        self.calls = []

    def detect(self, segments):
        self.calls.append([segment["id"] for segment in segments])
        text_by_id = {segment["id"]: segment["text"] for segment in segments}
        output = []
        for finding in self.findings:
            if finding["segment_id"] not in self.calls[-1]:
                continue
            row = dict(finding)
            row.setdefault("detector", "fake")
            row.setdefault("text", text_by_id[row["segment_id"]][row["start"] : row["end"]])
            output.append(row)
        return output


def corpus():
    return [
        {
            "id": "synthetic-1",
            "segments": [
                {"id": "s1", "text": "Alice met Bob."},
                {"id": "s2", "text": "Call 123456789."},
                {"id": "s3", "text": "Contact Carol."},
            ],
            "annotations": [
                {"segment_id": "s1", "start": 0, "end": 5, "category": "person"},
                {"segment_id": "s1", "start": 10, "end": 13, "category": "person"},
                {"segment_id": "s2", "start": 5, "end": 14, "category": "phone"},
                {"segment_id": "s3", "start": 8, "end": 13, "category": "person"},
            ],
        }
    ]


def test_exact_partial_residual_and_false_positive_metrics():
    detector = FixedDetector(
        [
            {
                "id": "f1",
                "segment_id": "s1",
                "start": 0,
                "end": 5,
                "category": "person",
                "score": 0.9,
            },
            {
                "id": "f2",
                "segment_id": "s1",
                "start": 6,
                "end": 9,
                "category": "email",
                "score": 0.8,
            },
            # A partial overlap with the phone gold span, not an exact match.
            {
                "id": "f3",
                "segment_id": "s2",
                "start": 5,
                "end": 10,
                "category": "phone",
                "score": 0.7,
            },
        ]
    )
    result = evaluate(corpus(), detector)
    detection = result["detection"]
    assert detection["gold"] == 4
    assert detection["predictions"] == 3
    assert detection["tp"] == 1
    assert detection["fp"] == 2
    assert detection["fn"] == 3
    assert detection["precision"] == pytest.approx(1 / 3, abs=1e-6)
    assert detection["recall"] == pytest.approx(1 / 4, abs=1e-6)
    assert detection["partial_overlap_predictions"] == 1
    assert detection["by_category"]["person"]["tp"] == 1
    assert detection["by_category"]["person"]["fn"] == 2
    assert detection["by_category"]["email"]["fp"] == 1
    assert detection["by_category"]["phone"]["fn"] == 1
    assert detection["document_residual_misses"] == [
        {
            "document_id": "synthetic-1",
            "gold_annotations": 4,
            "detected_exact": 1,
            "candidate_exact_misses": 3,
            "residual_misses": 3,
            "residual_categories": {"person": 2, "phone": 1},
        }
    ]
    assert result["corpus"]["scope"].startswith("synthetic")
    assert any("not representative" in caveat for caveat in result["caveats"])


def test_duplicate_prediction_counts_as_false_positive():
    detector = FixedDetector(
        [
            {"id": "f1", "segment_id": "s1", "start": 0, "end": 5, "category": "person"},
            {"id": "f2", "segment_id": "s1", "start": 0, "end": 5, "category": "person"},
        ]
    )
    result = evaluate(corpus(), detector)
    assert result["detection"]["tp"] == 1
    assert result["detection"]["fp"] == 1
    assert result["detection"]["by_category"]["person"]["fp"] == 1


def test_fully_masked_wrong_category_is_exact_fn_but_not_residual_disclosure():
    result = evaluate(
        [
            {
                "id": "wrong-category",
                "segments": [{"id": "s", "text": "Alice"}],
                "annotations": [{"segment_id": "s", "start": 0, "end": 5, "category": "person"}],
            }
        ],
        FixedDetector([{"id": "f", "segment_id": "s", "start": 0, "end": 5, "category": "email"}]),
    )
    assert result["detection"]["fn"] == 1
    assert result["detection"]["residual_disclosures"] == 0
    assert result["ranking"]["undetected_annotations"] == 0


def test_fully_masked_pii_does_not_retain_identifier_queue_signal():
    result = evaluate(
        [
            {
                "id": "masked",
                "segments": [{"id": "s", "text": "Alice"}],
                "annotations": [{"segment_id": "s", "start": 0, "end": 5, "category": "person"}],
            }
        ],
        FixedDetector([{"id": "f", "segment_id": "s", "start": 0, "end": 5, "category": "person"}]),
    )
    ranked = result["ranking"]["ranked"]["curve"][0]
    assert result["detection"]["residual_disclosures"] == 0
    assert ranked[-1]["residual_undetected_annotations"] == 0


def test_partial_masking_remains_a_residual_disclosure():
    result = evaluate(
        [
            {
                "id": "partial",
                "segments": [{"id": "s", "text": "Alice"}],
                "annotations": [{"segment_id": "s", "start": 0, "end": 5, "category": "person"}],
            }
        ],
        FixedDetector([{"id": "f", "segment_id": "s", "start": 0, "end": 3, "category": "person"}]),
    )
    assert result["detection"]["residual_disclosures"] == 1
    assert result["ranking"]["undetected_annotations"] == 1


def test_connected_overlap_union_masks_full_gold_span():
    result = evaluate(
        [
            {
                "id": "overlap",
                "segments": [{"id": "s", "text": "Anna Beispiel."}],
                "annotations": [{"segment_id": "s", "start": 0, "end": 13, "category": "person"}],
            }
        ],
        FixedDetector(
            [
                {
                    "id": "short",
                    "segment_id": "s",
                    "start": 0,
                    "end": 4,
                    "category": "person",
                    "score": 0.95,
                },
                {
                    "id": "full",
                    "segment_id": "s",
                    "start": 0,
                    "end": 13,
                    "category": "person",
                    "score": 0.60,
                },
            ]
        ),
    )
    assert result["detection"]["residual_disclosures"] == 0
    assert result["replacement_quality"]["proposed_occurrences"] == 1


def test_ranked_curve_beats_source_order_for_contextual_miss():
    ranking_corpus = [
        {
            "id": "ranking-only",
            "segments": [
                {"id": "r1", "text": "Ordinary process note."},
                {"id": "r2", "text": "Telefon Carol."},
                {"id": "r3", "text": "Another ordinary note."},
            ],
            "annotations": [{"segment_id": "r2", "start": 8, "end": 13, "category": "person"}],
        }
    ]
    result = evaluate(ranking_corpus, FixedDetector([]))
    ranked = result["ranking"]["ranked"]["curve"][0]
    baseline = result["ranking"]["unranked_source_order"]["curve"][0]
    # The missed "Contact Carol" segment receives a personal-context signal,
    # so it is recovered before the final source-order passage.
    assert ranked[0]["recovered_undetected_annotations"] == 0
    assert baseline[0]["recovered_undetected_annotations"] == 0
    assert ranked[1]["recovered_undetected_annotations"] == 1
    assert baseline[1]["recovered_undetected_annotations"] == 0
    assert ranked[-1]["residual_undetected_annotations"] == 0
    assert baseline[-1]["residual_undetected_annotations"] == 0


def test_evaluation_does_not_load_a_model_and_calls_detector_once():
    detector = FixedDetector([])
    result = evaluate(corpus(), detector)
    assert len(detector.calls) == 1
    assert result["schema_version"] == 1


def test_detector_failure_is_safe_and_does_not_echo_source():
    class FailingDetector:
        def detect(self, segments):
            raise RuntimeError("secret source text")

    with pytest.raises(AnonymizerError, match="detector evaluation failed") as exc:
        evaluate(corpus(), FailingDetector())
    assert "secret source text" not in str(exc.value)


@pytest.mark.parametrize(
    "bad",
    [
        [
            {
                "id": "d",
                "segments": [{"id": "s", "text": "x"}],
                "annotations": [{"segment_id": "s", "start": 0, "end": 2, "category": "person"}],
            }
        ],
        [
            {
                "id": "d",
                "segments": [{"id": "s", "text": "x"}],
                "annotations": [
                    {"segment_id": "missing", "start": 0, "end": 1, "category": "person"}
                ],
            }
        ],
        [
            {
                "id": "d",
                "segments": [{"id": "s", "text": "x"}],
                "annotations": [
                    {"segment_id": "s", "start": 0, "end": 1, "category": "person"},
                    {"segment_id": "s", "start": 0, "end": 1, "category": "person"},
                ],
            }
        ],
    ],
)
def test_invalid_gold_is_rejected_without_source_echo(bad):
    with pytest.raises(AnonymizerError) as exc:
        evaluate(bad, FixedDetector([]))
    assert "x" not in str(exc.value)


def test_public_synthetic_corpus_is_json_compatible():
    path = Path(__file__).parents[1] / "examples" / "evaluation.synthetic.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert all(
        "id" in document and "segments" in document and "annotations" in document
        for document in payload
    )
