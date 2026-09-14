"""Review queue ranking over source evidence and the rendered candidate."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any

from .detection import rank_passages
from .errors import AnonymizerError


def _error(message: str) -> AnonymizerError:
    return AnonymizerError(message)


def _masked(interval: tuple[int, int], intervals: list[tuple[int, int]], source_text: str) -> bool:
    cursor = interval[0]
    for start, end in sorted(intervals):
        if end <= cursor:
            continue
        if start > cursor and not source_text[cursor:start].isspace():
            return False
        cursor = max(cursor, end)
        if cursor >= interval[1]:
            return True
    return not source_text[cursor : interval[1]].strip()


def _overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


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


def canonical_occurrence_id(finding: dict[str, Any]) -> str:
    """Return the workflow's stable ID for a source finding."""
    fields = [finding[key] for key in ("segment_id", "start", "end", "category", "text")]
    raw = json.dumps(fields, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    return "o" + hashlib.sha256(raw).hexdigest()[:20]


def _legacy_resolve_findings(
    findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve with the pre-coverage-union priority projection."""
    findings = [finding for finding in findings if not finding.get("review_only", False)]
    selected: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen: set[str] = set()
    for finding in sorted(
        findings, key=lambda item: (-item["score"], -(item["end"] - item["start"]), item["id"])
    ):
        if finding["id"] in seen:
            continue
        seen.add(finding["id"])
        span = (finding["start"], finding["end"])
        if any(_overlap(span, existing) for existing in occupied[finding["segment_id"]]):
            suppressed.append({**finding, "reason": "overlap_priority"})
            continue
        occupied[finding["segment_id"]].append(span)
        selected.append(finding)
    return selected, suppressed


def _merged_text(component: list[dict[str, Any]], start: int, end: int) -> str:
    """Stitch a connected component from checked source slices."""
    characters: list[str | None] = [None] * (end - start)
    for finding in component:
        finding_start, finding_end = finding["start"], finding["end"]
        value = finding.get("text")
        if not isinstance(value, str) or len(value) != finding_end - finding_start:
            raise _error("Finding text does not match its source span")
        for offset, character in enumerate(value, finding_start - start):
            prior = characters[offset]
            if prior is not None and prior != character:
                raise _error("Overlapping findings disagree about source text")
            characters[offset] = character
    if any(character is None for character in characters):
        raise _error("Overlapping findings leave a source-text gap")
    return "".join(characters)  # type: ignore[arg-type]


def _merge_component(
    component: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start = min(finding["start"] for finding in component)
    end = max(finding["end"] for finding in component)
    category = min(
        (finding["category"] for finding in component),
        key=lambda value: (_CATEGORY_PRIORITY.get(value, 100), value),
    )
    merged_text = _merged_text(component, start, end)
    keys = [finding.get("entity_key") for finding in component]
    explicit_keys = {key for key in keys if isinstance(key, str) and key}
    entity_key = None
    if len(explicit_keys) == 1:
        key = next(iter(explicit_keys))
        # Unkeyed model evidence must not erase an explicit full-span alias.
        # A partial identity claim alone cannot identify an expanded union.
        if all(value == key for value in keys) or any(
            finding.get("entity_key") == key
            and finding["start"] == start and finding["end"] == end
            for finding in component
        ):
            entity_key = key
    merged: dict[str, Any] = {
        "id": "",
        "segment_id": component[0]["segment_id"],
        "start": start,
        "end": end,
        "text": merged_text,
        "category": category,
        "score": max(float(finding["score"]) for finding in component),
        "detector": "resolver:coverage_union",
        "provenance": {
            "categories": sorted({finding["category"] for finding in component}),
            "detectors": sorted({finding["detector"] for finding in component}),
            "finding_ids": sorted(finding["id"] for finding in component),
        },
    }
    if entity_key is not None:
        merged["entity_key"] = entity_key
    merged["id"] = canonical_occurrence_id(merged)
    suppressed = [
        {**finding, "reason": "overlap_merged", "merged_into": merged["id"]}
        for finding in component
    ]
    return merged, suppressed


def resolve_findings(
    findings: list[dict[str, Any]],
    *,
    preserve_coverage: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return replacement findings and evidence without losing overlap coverage.

    New runs merge transitively connected overlapping proposal spans into one
    exact source interval.  ``review_only`` findings are never part of a mask.
    ``preserve_coverage=False`` is reserved for replaying legacy run projections
    whose selected/suppressed state was produced by the old greedy resolver.
    """
    if not preserve_coverage:
        return _legacy_resolve_findings(findings)
    proposals = [finding for finding in findings if not finding.get("review_only", False)]
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in proposals:
        by_segment[finding["segment_id"]].append(finding)
    selected: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for segment_id in sorted(by_segment):
        ordered = sorted(
            by_segment[segment_id],
            key=lambda finding: (finding["start"], finding["end"], finding["id"]),
        )
        component: list[dict[str, Any]] = []
        component_end = -1
        for finding in ordered:
            if component and finding["start"] >= component_end:
                if len(component) == 1:
                    selected.append(component[0])
                else:
                    merged, component_suppressed = _merge_component(component)
                    selected.append(merged)
                    suppressed.extend(component_suppressed)
                component = []
            component.append(finding)
            component_end = max(component_end, finding["end"])
        if component:
            if len(component) == 1:
                selected.append(component[0])
            else:
                merged, component_suppressed = _merge_component(component)
                selected.append(merged)
                suppressed.extend(component_suppressed)
    selected.sort(
        key=lambda finding: (finding["segment_id"], finding["start"], finding["end"], finding["id"])
    )
    suppressed.sort(
        key=lambda finding: (finding["segment_id"], finding["start"], finding["end"], finding["id"])
    )
    return selected, suppressed


def rank_retained_passages(
    source_segments: list[dict[str, Any]],
    rendered_segments: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    masked_spans: list[dict[str, Any]] | None = None,
    *,
    sample_size: int = 5,
    sample_seed: str = "review-remainder-v1",
) -> list[dict[str, Any]]:
    """Rank visible candidate passages with source-offset evidence.

    ``findings`` use immutable source offsets.  They are never used to index
    ``rendered_segments``; only ``masked_spans`` can suppress their
    review signal.  Scores remain heuristic and are not probabilities.
    """
    if not isinstance(source_segments, list) or not isinstance(rendered_segments, list):
        raise _error("Review ranking segments are invalid")
    if (
        not isinstance(findings, list)
        or not isinstance(sample_size, int)
        or sample_size < 0
        or not isinstance(sample_seed, str)
    ):
        raise _error("Review ranking inputs are invalid")
    source = {
        item.get("id"): item.get("text") for item in source_segments if isinstance(item, dict)
    }
    rendered = {
        item.get("id"): item.get("text") for item in rendered_segments if isinstance(item, dict)
    }
    if (
        len(source) != len(source_segments)
        or len(rendered) != len(rendered_segments)
        or set(source) != set(rendered)
        or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in source.items()
        )
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in rendered.items()
        )
    ):
        raise _error("Review ranking segments do not match")
    masks: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for mask in masked_spans or []:
        if not isinstance(mask, dict):
            raise _error("Review ranking masks are invalid")
        segment_id = mask.get("segment_id")
        interval = (mask.get("start"), mask.get("end"))
        if (
            segment_id not in source
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in interval)
            or not 0 <= interval[0] < interval[1] <= len(source[segment_id])
        ):
            raise _error("Review ranking masks are invalid")
        masks[segment_id].append((interval[0], interval[1]))

    valid_findings: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, dict):
            raise _error("Review ranking findings are invalid")
        segment_id = finding.get("segment_id")
        start, end = finding.get("start"), finding.get("end")
        if (
            not isinstance(segment_id, str)
            or segment_id not in source
            or not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(source[segment_id])
        ):
            raise _error("Review ranking findings are invalid")
        row = dict(finding)
        score = finding.get("score", 0.0)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise _error("Review ranking findings are invalid")
        row["score"] = float(score)
        valid_findings.append(row)

    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in valid_findings:
        by_segment[finding["segment_id"]].append(finding)
    evidence: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for finding in valid_findings:
        segment_id = finding["segment_id"]
        interval = (finding["start"], finding["end"])
        if _masked(interval, masks[segment_id], source[segment_id]):
            continue
        reasons: list[str] = []
        if finding.get("review_only", False):
            reasons.append("review_only_candidate")
        else:
            reasons.append("unresolved_candidate")
        if finding["score"] < 0.75:
            reasons.append("low_confidence_candidate")
        if any(
            other["segment_id"] == segment_id
            and _overlap(interval, (other["start"], other["end"]))
            and other is not finding
            and (other["start"], other["end"], other.get("category"))
            != (finding["start"], finding["end"], finding.get("category"))
            and other["score"] >= finding["score"]
            for other in by_segment[segment_id]
        ):
            reasons.append("overlap_suppressed_candidate")
        reason_boosts = {reasons[0]: (0.30 if finding.get("review_only", False) else 0.10)}
        if "low_confidence_candidate" in reasons:
            reason_boosts["low_confidence_candidate"] = 0.20
        if "overlap_suppressed_candidate" in reasons:
            reason_boosts["overlap_suppressed_candidate"] = 0.18
        for reason, reason_boost in reason_boosts.items():
            evidence[segment_id].append((reason_boost, reason))

    ranked = rank_passages(rendered_segments, [])
    for item in ranked:
        signals = evidence.get(item["id"], [])
        strongest = {}
        for boost, reason in signals:
            strongest[reason] = max(strongest.get(reason, 0.0), boost)
        item["score"] = round(min(1.0, item["score"] + sum(strongest.values())), 6)
        reasons = list(item.get("reasons", []))
        for _, reason in signals:
            if reason not in reasons:
                reasons.append(reason)
        item["reasons"] = reasons
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    if sample_size:
        candidates = [item for item in ranked if not evidence.get(item["id"])]
        candidates = candidates[len(candidates) // 2 :]
        for item in sorted(
            candidates,
            key=lambda value: hashlib.sha256(f"{sample_seed}:{value['id']}".encode()).hexdigest(),
        )[:sample_size]:
            if "remainder_sample" not in item["reasons"]:
                item["reasons"].append("remainder_sample")
    return ranked


# Kept for callers that used the initial design-note name.
rank_review_passages = rank_retained_passages
