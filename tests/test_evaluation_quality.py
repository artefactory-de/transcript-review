from transcript_anonymizer.evaluation import evaluate


class Detector:
    def __init__(self, findings):
        self.findings = findings

    def detect(self, segments):
        return self.findings


def finding(identifier, start, end, text, **extra):
    return dict(
        id=identifier,
        segment_id="s1",
        start=start,
        end=end,
        text=text,
        category="person",
        detector="fixture",
        score=0.8,
        **extra,
    )


def test_review_candidates_are_evidence_but_not_masking():
    corpus = [
        {
            "id": "example",
            "segments": [{"id": "s1", "text": "Anna spricht."}],
            "annotations": [{"segment_id": "s1", "start": 0, "end": 4, "category": "person"}],
        }
    ]
    result = evaluate(corpus, Detector([finding("f1", 0, 4, "Anna", review_only=True)]))
    assert result["detection"]["recall"] == 1
    assert result["detection"]["residual_disclosures"] == 1
    assert result["review_only_candidates"] == 1
    assert result["replacement_quality"]["proposed_occurrences"] == 0


def test_replacement_metrics_do_not_double_count_duplicate_candidates():
    corpus = [
        {
            "id": "example",
            "segments": [{"id": "s1", "text": "Anna nutzt SAP."}],
            "annotations": [{"segment_id": "s1", "start": 0, "end": 4, "category": "person"}],
        }
    ]
    result = evaluate(
        corpus,
        Detector(
            [
                finding("f1", 0, 4, "Anna"),
                finding("f2", 0, 4, "Anna"),
                finding("f3", 11, 14, "SAP"),
            ]
        ),
    )
    assert result["detection"]["fp"] == 2
    assert result["replacement_quality"] == {
        "proposed_occurrences": 2,
        "wholly_false_positive_occurrences": 1,
        "masked_nonwhitespace_characters": 7,
        "non_pii_masked_characters": 3,
    }
    assert (
        result["ranking"]["ranked"]["effort_by_document"][0]["words_to_recover_all_residuals"] == 0
    )


def test_partial_masking_counts_non_pii_characters_and_remaining_miss_separately():
    corpus = [
        {
            "id": "example",
            "segments": [{"id": "s1", "text": "Anna sagt ja."}],
            "annotations": [{"segment_id": "s1", "start": 0, "end": 4, "category": "person"}],
        }
    ]
    result = evaluate(corpus, Detector([finding("f1", 2, 9, "na sagt")]))
    assert result["detection"]["residual_disclosures"] == 1
    assert result["replacement_quality"]["non_pii_masked_characters"] == 4
    assert result["replacement_quality"]["wholly_false_positive_occurrences"] == 0
    effort = result["ranking"]["ranked"]["effort_by_document"][0]
    assert effort["passages_to_recover_all_residuals"] == 1
    assert effort["residuals_after_passage_budget"]["1"] == 0
