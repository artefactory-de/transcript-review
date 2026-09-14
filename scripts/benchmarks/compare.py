"""Compare cached detector findings without loading multiple models into RAM.

Caches are arrays of finding arrays, in corpus document order. Only use caches
from the exact corpus bytes recorded by the associated runner metadata. The
comparator does not read those metadata files and cached findings do not encode
the complete source corpus, so callers must verify the recorded corpus hashes
themselves. Annotation-only corpus edits are permissible only after confirming
that every document's segment IDs, order, and text are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transcript_anonymizer.evaluation import evaluate
from transcript_anonymizer.ranking import canonical_occurrence_id


class CachedDetector:
    def __init__(self, documents, findings):
        if len(documents) != len(findings):
            raise ValueError("Cache document count does not match corpus")
        self.documents = documents
        self.findings = findings
        self.index = 0

    def detect(self, segments):
        expected = self.documents[self.index]["segments"]
        if [(s["id"], s["text"]) for s in segments] != [
            (s["id"], s["text"]) for s in expected
        ]:
            raise ValueError("Cache segment order does not match corpus")
        result = self.findings[self.index]
        self.index += 1
        return result


def combine(primary, secondary, *, review_only=False):
    if len(primary) != len(secondary):
        raise ValueError("Cache document counts differ")
    result = []
    for first, second in zip(primary, secondary, strict=True):
        # Keep original scores. Cross-model scores are not calibrated; this
        # experiment measures the existing resolver, not an optimized ensemble.
        rows = list(first)
        keys = {
            (f["segment_id"], f["start"], f["end"], f["category"], f["detector"])
            for f in first
        }
        for finding in second:
            key = (
                finding["segment_id"], finding["start"], finding["end"],
                finding["category"], finding["detector"],
            )
            if key not in keys:
                rows.append({**finding, "review_only": True} if review_only else dict(finding))
                keys.add(key)
        result.append(rows)
    return result


_CATEGORY_PRIORITY = {
    "person": 0,
    "email": 1,
    "phone": 2,
    "address": 3,
    "iban": 4,
    "account_id": 5,
    "customer_id": 6,
    "depot_id": 7,
}


def _source_segments(documents: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
    """Normalize source segments while retaining document boundaries by index."""

    result = []
    for document in documents:
        segments = document.get("segments")
        if not isinstance(segments, list):
            raise TypeError("Coverage union source segments are invalid")
        rows = []
        seen = set()
        for segment in segments:
            if not isinstance(segment, dict):
                raise TypeError("Coverage union source segments are invalid")
            segment_id, text = segment.get("id"), segment.get("text")
            if not isinstance(segment_id, str) or segment_id in seen or not isinstance(text, str):
                raise ValueError("Coverage union source segments are invalid")
            seen.add(segment_id)
            rows.append({"id": segment_id, "text": text})
        result.append(rows)
    return result


def _make_union_finding(component: list[dict[str, Any]], segment_id: str, source: dict[str, str]):
    start = min(row["start"] for row in component)
    end = max(row["end"] for row in component)
    category = min(
        (row["category"] for row in component),
        key=lambda value: (_CATEGORY_PRIORITY.get(value, 100), value),
    )
    finding = {
        "id": "",
        "segment_id": segment_id,
        "start": start,
        "end": end,
        "text": source[segment_id][start:end],
        "category": category,
        "score": max(float(row["score"]) for row in component),
        "detector": "experimental:coverage-union",
    }
    finding["id"] = canonical_occurrence_id(finding)
    return finding


def _copy_rows(first, second, *, review_only: bool) -> list[list[dict[str, Any]]]:
    if len(first) != len(second):
        raise ValueError("Cache document counts differ")
    result = []
    for first_rows, second_rows in zip(first, second, strict=True):
        rows = [dict(row) for row in first_rows]
        keys = {
            (row["segment_id"], row["start"], row["end"], row["category"], row["detector"])
            for row in first_rows
        }
        for finding in second_rows:
            key = (
                finding["segment_id"], finding["start"], finding["end"],
                finding["category"], finding["detector"],
            )
            if key in keys:
                continue
            row = dict(finding)
            if review_only:
                row["review_only"] = True
            rows.append(row)
            keys.add(key)
        result.append(rows)
    return result


def _coverage_union_rows(rows_by_document, documents):
    """Apply the experimental union to already-combined document rows."""

    source_by_document = _source_segments(documents)
    if len(rows_by_document) != len(source_by_document):
        raise ValueError("Coverage union document counts differ")
    output = []
    for rows, source_segments in zip(rows_by_document, source_by_document, strict=True):
        source = {segment["id"]: segment["text"] for segment in source_segments}
        for row in rows:
            segment_id, start, end = row.get("segment_id"), row.get("start"), row.get("end")
            if (
                segment_id not in source
                or not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not 0 <= start < end <= len(source[segment_id])
                or row.get("text") != source[segment_id][start:end]
            ):
                raise ValueError("Coverage union finding span is invalid")

        proposals = [row for row in rows if not row.get("review_only", False)]
        evidence = [row for row in rows if row.get("review_only", False)]
        by_segment: dict[str, list[dict[str, Any]]] = {}
        for row in proposals:
            by_segment.setdefault(row["segment_id"], []).append(row)
        merged: list[dict[str, Any]] = []
        for segment_id, segment_rows in by_segment.items():
            ordered = sorted(segment_rows, key=lambda row: (row["start"], row["end"], row["id"]))
            component = []
            component_end = -1

            for row in ordered:
                if component and row["start"] >= component_end:
                    merged.append(_make_union_finding(component, segment_id, source))
                    component = []
                component.append(row)
                component_end = max(component_end, row["end"])
            if component:
                merged.append(_make_union_finding(component, segment_id, source))
        output.append(
            sorted(merged + [dict(row) for row in evidence], key=lambda row: (
                row["segment_id"], row["start"], row["end"], row.get("review_only", False), row["id"]
            ))
        )
    return output


def combine_coverage_union(primary, secondary, documents, *, review_only=False):
    """Union overlapping proposal coverage for an experimental comparison.

    Non-review findings are merged per source segment into connected overlap
    components.  The merged row uses the exact source slice, the highest input
    score, a deterministic category priority, and a stable canonical ID.  Rows
    marked ``review_only`` are never part of a mask component and are retained
    as evidence for the ranker.  Inputs are copied and never modified.
    """

    return _coverage_union_rows(
        _copy_rows(primary, secondary, review_only=review_only), documents
    )


def coverage_union(documents, cache):
    """Apply coverage union to one cached detector (including current rules)."""

    if len(documents) != len(cache):
        raise ValueError("Cache document counts differ")
    return _coverage_union_rows([[dict(row) for row in rows] for rows in cache], documents)


def summary(result):
    effort = result["ranking"]["ranked"]["effort_by_document"]
    return {
        "gold_identifiers": result["detection"]["gold"],
        "residual_disclosures": result["detection"]["residual_disclosures"],
        **result["replacement_quality"],
        "review_only_candidates": result["review_only_candidates"],
        "passages_to_recover_all_residuals": sum(
            e["passages_to_recover_all_residuals"] for e in effort
        ),
        "words_to_recover_all_residuals": sum(
            e["words_to_recover_all_residuals"] for e in effort
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cache", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--primary", help="Also compare each other cache as a complement")
    parser.add_argument(
        "--coverage-union",
        action="store_true",
        help="Add experimental variants that preserve all overlapping proposal coverage",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    documents = json.loads(args.corpus.read_text())
    caches = {}
    hashes = {}
    for item in args.cache:
        name, path = item.split("=", 1)
        if not name or name in caches:
            parser.error("Cache names must be nonempty and unique")
        raw = Path(path).read_bytes()
        caches[name] = json.loads(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    variants = dict(caches)
    if args.coverage_union:
        for name, cache in caches.items():
            variants[f"{name}:coverage-union"] = coverage_union(documents, cache)
    if args.primary:
        if args.primary not in caches:
            parser.error("Primary must name one supplied cache")
        for name, cache in caches.items():
            if name == args.primary:
                continue
            for review_only in (False, True):
                suffix = "review" if review_only else "union"
                variants[f"{args.primary}+{name}:{suffix}"] = combine(
                    caches[args.primary], cache, review_only=review_only
                )
                if args.coverage_union:
                    suffix = "coverage-review" if review_only else "coverage-union"
                    variants[f"{args.primary}+{name}:{suffix}"] = combine_coverage_union(
                        caches[args.primary], cache, documents, review_only=review_only
                    )
    results = {}
    for name, findings in variants.items():
        metrics = evaluate(documents, CachedDetector(documents, findings))
        results[name] = {"summary": summary(metrics), "metrics": metrics}
    output = {
        "scope": "Synthetic diagnostic comparison; not held-out acceptance",
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "cache_sha256": hashes,
        "documents": len(documents),
        "variants": results,
    }
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({name: value["summary"] for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
