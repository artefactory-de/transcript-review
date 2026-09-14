import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_compare", Path(__file__).parents[1] / "scripts/benchmarks/compare.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def finding(detector="rule:email", **kwargs):
    return {
        "id": "a", "segment_id": "s1", "start": 0, "end": 4,
        "text": "Anna", "category": "person", "score": 0.8,
        "detector": detector, **kwargs,
    }


def test_common_rules_are_not_duplicated_and_inputs_unchanged():
    first = [[finding()]]
    second = [[finding(), finding("model:secondary")]]
    merged = module.combine(first, second, review_only=True)
    assert len(merged[0]) == 2
    assert "review_only" not in merged[0][0]
    assert merged[0][1]["review_only"] is True
    assert "review_only" not in second[0][1]


def test_cache_length_and_segment_order_checked():
    documents = [{"segments": [{"id": "s1", "text": "Anna"}]}]
    with pytest.raises(ValueError, match="count"):
        module.CachedDetector(documents, [])
    detector = module.CachedDetector(documents, [[finding()]])
    with pytest.raises(ValueError, match="order"):
        detector.detect([{"id": "s2", "text": "Anna"}])
    with pytest.raises(ValueError, match="counts"):
        module.combine([[]], [])


def _document(text="abcdefghij"):
    return [{"id": "doc", "segments": [{"id": "s1", "text": text}]}]


def _span(start, end, *, category="person", detector="detector", review_only=False, text=None):
    row = {
        "id": f"{detector}-{start}-{end}",
        "segment_id": "s1",
        "start": start,
        "end": end,
        "text": text if text is not None else "abcdefghij"[start:end],
        "category": category,
        "score": 0.8,
        "detector": detector,
    }
    if review_only:
        row["review_only"] = True
    return row


def test_coverage_union_preserves_all_overlapping_source_characters_and_copies_inputs():
    primary = [[_span(0, 6, detector="rules")]]
    secondary = [[_span(4, 10, detector="spacy")]]
    before = (copy.deepcopy(primary), copy.deepcopy(secondary))

    merged = module.combine_coverage_union(primary, secondary, _document())

    assert [(row["start"], row["end"], row["text"]) for row in merged[0]] == [(0, 10, "abcdefghij")]
    assert merged[0][0]["detector"] == "experimental:coverage-union"
    assert merged[0][0]["id"].startswith("o")
    assert primary == before[0]
    assert secondary == before[1]


def test_coverage_union_keeps_disjoint_spans_separate_and_preserves_whitespace_boundaries():
    documents = _document("ab  cd ef")
    cache = [[_span(0, 2, detector="one", text="ab"), _span(7, 9, detector="two", text="ef")]]

    merged = module.coverage_union(documents, cache)

    assert [(row["start"], row["end"], row["text"]) for row in merged[0]] == [
        (0, 2, "ab"),
        (7, 9, "ef"),
    ]


def test_coverage_union_merges_overlap_chains_but_review_only_is_evidence_not_mask():
    documents = _document("abcdefghij")
    primary = [[_span(0, 3, detector="one"), _span(4, 7, detector="three")]]
    secondary = [[_span(2, 5, detector="two"), _span(7, 10, detector="review", review_only=True)]]

    merged = module.combine_coverage_union(primary, secondary, documents)
    assert [(row["start"], row["end"]) for row in merged[0] if not row.get("review_only")] == [(0, 7)]
    assert [(row["start"], row["end"]) for row in merged[0] if row.get("review_only")] == [(7, 10)]

    reviewed = module.combine_coverage_union(primary, secondary, documents, review_only=True)
    assert [(row["start"], row["end"]) for row in reviewed[0] if not row.get("review_only")] == [(0, 3), (4, 7)]
    assert all(row.get("review_only") for row in reviewed[0] if row["start"] == 2)


def test_coverage_union_uses_deterministic_category_priority_for_mixed_overlap():
    documents = _document("abcdefgh")
    cache = [[_span(0, 5, category="email", detector="email"), _span(3, 8, category="person", detector="person")]]

    merged = module.coverage_union(documents, cache)

    assert merged[0][0]["category"] == "person"
    assert merged[0][0]["text"] == "abcdefgh"


def test_coverage_union_rejects_stale_text_and_boolean_offsets():
    documents = _document("abcdefgh")
    stale = [[_span(0, 4, text="WRONG")]]
    with pytest.raises(ValueError, match="span"):
        module.coverage_union(documents, stale)
    boolean_offset = [[_span(0, 4)]]
    boolean_offset[0][0]["start"] = True
    with pytest.raises(ValueError, match="span"):
        module.coverage_union(documents, boolean_offset)
