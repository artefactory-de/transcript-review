"""Small, offline evaluation harness for detection and passage ranking.

This module intentionally evaluates spans and queue concentration only.  It does
not claim that a synthetic corpus represents real-world material and does not
invent a process-meaning score.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from .errors import AnonymizerError
from .ranking import canonical_occurrence_id, rank_retained_passages, resolve_findings

_WORD_RE = re.compile(r"\S+")


def _error(message: str) -> AnonymizerError:
    return AnonymizerError(message)


def _text(value: Any, field: str, *, blank: bool = False) -> str:
    if not isinstance(value, str) or (not blank and not value):
        raise _error(f"invalid evaluation {field}")
    return value


def _number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise _error(f"invalid evaluation {field}")
    return float(value)


def _span(value: Any, field: str, text: str) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise _error(f"invalid evaluation {field}")
    start, end = value.get("start"), value.get("end")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not 0 <= start < end <= len(text)
    ):
        raise _error(f"invalid evaluation {field} span")
    return start, end


def _validate_documents(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise _error("evaluation documents must be a non-empty list")
    documents: list[dict[str, Any]] = []
    doc_ids: set[str] = set()
    for document in raw:
        if not isinstance(document, dict):
            raise _error("invalid evaluation document")
        doc_id = _text(document.get("id"), "document ID")
        if doc_id in doc_ids:
            raise _error("duplicate evaluation document ID")
        doc_ids.add(doc_id)
        segments = document.get("segments")
        annotations = document.get("annotations")
        if not isinstance(segments, list) or not isinstance(annotations, list):
            raise _error("evaluation document needs segments and annotations")
        segment_map: dict[str, dict[str, Any]] = {}
        ordered_segments: list[dict[str, str]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                raise _error("invalid evaluation segment")
            segment_id = _text(segment.get("id"), "segment ID")
            text = _text(segment.get("text"), "segment text", blank=True)
            if segment_id in segment_map:
                raise _error("duplicate evaluation segment ID")
            segment_map[segment_id] = {"id": segment_id, "text": text}
            ordered_segments.append({"id": segment_id, "text": text})
        gold: list[dict[str, Any]] = []
        gold_keys: set[tuple[str, int, int, str]] = set()
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise _error("invalid evaluation annotation")
            segment_id = _text(annotation.get("segment_id"), "annotation segment ID")
            segment = segment_map.get(segment_id)
            if segment is None:
                raise _error("annotation references an unknown segment")
            start, end = _span(annotation, "annotation", segment["text"])
            category = _text(annotation.get("category"), "annotation category")
            key = (segment_id, start, end, category)
            if key in gold_keys:
                raise _error("duplicate evaluation annotation")
            gold_keys.add(key)
            gold.append(
                {"segment_id": segment_id, "start": start, "end": end, "category": category}
            )
        documents.append({"id": doc_id, "segments": ordered_segments, "annotations": gold})
    return documents


def _validate_findings(raw: Any, segments: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise _error("detector returned invalid evaluation findings")
    segment_map = {segment["id"]: segment["text"] for segment in segments}
    findings: list[dict[str, Any]] = []
    for finding in raw:
        if not isinstance(finding, dict):
            raise _error("detector returned invalid evaluation finding")
        _text(finding.get("id"), "finding ID")
        _text(finding.get("detector"), "finding detector")
        segment_id = _text(finding.get("segment_id"), "finding segment ID")
        text = segment_map.get(segment_id)
        if text is None:
            raise _error("detector returned a finding for an unknown segment")
        start, end = _span(finding, "finding", text)
        category = _text(finding.get("category"), "finding category")
        finding_text = _text(finding.get("text"), "finding text", blank=True)
        if finding_text != text[start:end]:
            raise _error("detector returned a finding with inconsistent text")
        normalized = dict(finding)
        if "review_only" in normalized and not isinstance(normalized["review_only"], bool):
            raise _error("detector returned an invalid review-only flag")
        normalized.update(
            {"segment_id": segment_id, "start": start, "end": end, "category": category}
        )
        if "score" in normalized:
            normalized["score"] = _number(normalized["score"], "finding score")
        else:
            normalized["score"] = 0.0
        if not 0 <= normalized["score"] <= 1:
            raise _error("detector returned an invalid finding score")
        findings.append(normalized)
    return findings


def _words(segments: list[dict[str, str]]) -> int:
    return sum(len(_WORD_RE.findall(segment["text"])) for segment in segments)


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _metric(value: int, denominator: int) -> float:
    return round(value / denominator, 6) if denominator else 0.0


def _curve(order: list[dict[str, str]], undetected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_segment: dict[str, int] = Counter(item["segment_id"] for item in undetected)
    total = len(undetected)
    words = 0
    recovered = 0
    curve = [
        {
            "inspected_passages": 0,
            "inspected_words": 0,
            "recovered_undetected_annotations": 0,
            "residual_undetected_annotations": total,
        }
    ]
    for index, segment in enumerate(order, 1):
        words += len(_WORD_RE.findall(segment["text"]))
        recovered += by_segment.get(segment["id"], 0)
        curve.append(
            {
                "inspected_passages": index,
                "inspected_words": words,
                "recovered_undetected_annotations": recovered,
                "residual_undetected_annotations": total - recovered,
            }
        )
    return curve


def _selected_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve overlaps with the workflow's coverage-preserving projection."""
    return resolve_findings(
        [
            {**f, "id": canonical_occurrence_id(f)}
            for f in findings
            if not f.get("review_only", False)
        ]
    )[0]


def _render_selected(
    segments: list[dict[str, str]], selected: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Render a candidate after selected findings have been replaced."""
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in selected:
        by_segment[finding["segment_id"]].append(finding)
    rendered: list[dict[str, str]] = []
    for segment in segments:
        text = segment["text"]
        for finding in sorted(by_segment.get(segment["id"], []), key=lambda item: -item["start"]):
            text = text[: finding["start"]] + "<REDACTED>" + text[finding["end"] :]
        rendered.append({"id": segment["id"], "text": text})
    return rendered


def _residual_annotations(
    gold: list[dict[str, Any]], selected: list[dict[str, Any]], segments: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Keep gold spans with at least one uncovered, non-whitespace code point."""
    text_by_segment = {segment["id"]: segment["text"] for segment in segments}
    residual: list[dict[str, Any]] = []
    for annotation in gold:
        covered = [False] * (annotation["end"] - annotation["start"])
        for finding in selected:
            if finding["segment_id"] != annotation["segment_id"]:
                continue
            start = max(annotation["start"], finding["start"])
            end = min(annotation["end"], finding["end"])
            for offset in range(start, end):
                covered[offset - annotation["start"]] = True
        source_text = text_by_segment[annotation["segment_id"]][
            annotation["start"] : annotation["end"]
        ]
        if any(
            not covered[index] and not character.isspace()
            for index, character in enumerate(source_text)
        ):
            residual.append(annotation)
    return residual


def _replacement_metrics(segments, gold, selected):
    """Measure actual masking, independently of duplicate/category candidates."""
    gold_positions = defaultdict(set)
    texts = {s["id"]: s["text"] for s in segments}
    for item in gold:
        gold_positions[item["segment_id"]].update(range(item["start"], item["end"]))
    masked = unnecessary = wholly_false = 0
    for finding in selected:
        sid = finding["segment_id"]
        positions = {
            i for i in range(finding["start"], finding["end"]) if not texts[sid][i].isspace()
        }
        masked += len(positions)
        unnecessary += len(positions - gold_positions[sid])
        wholly_false += bool(positions and not positions.intersection(gold_positions[sid]))
    return {
        "proposed_occurrences": len(selected),
        "wholly_false_positive_occurrences": wholly_false,
        "masked_nonwhitespace_characters": masked,
        "non_pii_masked_characters": unnecessary,
    }


def _effort(curve):
    """Summarize oracle-labelled effort; it is not an operational stop rule."""
    completion = next(row for row in curve if row["residual_undetected_annotations"] == 0)
    return {
        "passages_to_recover_all_residuals": completion["inspected_passages"],
        "words_to_recover_all_residuals": completion["inspected_words"],
        "residuals_after_passage_budget": {
            str(budget): curve[min(budget, len(curve) - 1)]["residual_undetected_annotations"]
            for budget in (1, 3, 5, 10)
        },
    }


def evaluate(annotated_documents: list[dict], detector: Any, *, ranker=None) -> dict:
    """Evaluate a detector and its uncertainty queue on annotated documents.

    Gold source text is consumed locally and never copied into the returned
    metrics.  ``detector`` must expose ``detect(segments)``; no model is loaded
    by this function itself.
    """
    documents = _validate_documents(annotated_documents)
    if detector is None or not callable(getattr(detector, "detect", None)):
        raise _error("evaluation detector must provide detect()")
    selected_ranker = rank_retained_passages if ranker is None else ranker
    if not callable(selected_ranker):
        raise _error("evaluation ranker must be callable")

    total_gold = total_predictions = total_tp = total_partial = 0
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    document_residuals: list[dict[str, Any]] = []
    ranked_curves: list[list[dict[str, Any]]] = []
    baseline_curves: list[list[dict[str, Any]]] = []
    total_words = 0
    all_residual: list[dict[str, Any]] = []
    replacement_totals = Counter()
    review_candidates = 0

    for document in documents:
        segments = document["segments"]
        gold = document["annotations"]
        try:
            raw_findings = detector.detect(segments)
        except AnonymizerError:
            raise
        except Exception as exc:
            raise _error("detector evaluation failed") from exc
        findings = _validate_findings(raw_findings, segments)
        gold_keys = {
            (item["segment_id"], item["start"], item["end"], item["category"]) for item in gold
        }
        matched: set[tuple[str, int, int, str]] = set()
        prediction_is_tp: list[bool] = []
        exact_predictions = 0
        partial_predictions = 0
        for finding in findings:
            key = (finding["segment_id"], finding["start"], finding["end"], finding["category"])
            if key in gold_keys and key not in matched:
                matched.add(key)
                exact_predictions += 1
                prediction_is_tp.append(True)
            else:
                prediction_is_tp.append(False)
                span = (finding["start"], finding["end"])
                if any(
                    finding["segment_id"] == item["segment_id"]
                    and _overlap(span, (item["start"], item["end"]))
                    and key != (item["segment_id"], item["start"], item["end"], item["category"])
                    for item in gold
                ):
                    partial_predictions += 1
        undetected_exact = [
            item
            for item in gold
            if (item["segment_id"], item["start"], item["end"], item["category"]) not in matched
        ]
        for item in gold:
            category_counts[item["category"]]["gold"] += 1
            if (item["segment_id"], item["start"], item["end"], item["category"]) in matched:
                category_counts[item["category"]]["tp"] += 1
            else:
                category_counts[item["category"]]["fn"] += 1
        for index, finding in enumerate(findings):
            key = (finding["segment_id"], finding["start"], finding["end"], finding["category"])
            if not prediction_is_tp[index]:
                category_counts[finding["category"]]["fp"] += 1
        total_gold += len(gold)
        total_predictions += len(findings)
        total_tp += exact_predictions
        total_partial += partial_predictions
        total_words += _words(segments)
        document_residuals.append(
            {
                "document_id": document["id"],
                "gold_annotations": len(gold),
                "detected_exact": exact_predictions,
                "candidate_exact_misses": len(undetected_exact),
            }
        )

        selected = _selected_findings(findings)
        replacement_totals.update(_replacement_metrics(segments, gold, selected))
        review_candidates += sum(bool(f.get("review_only", False)) for f in findings)
        residual = _residual_annotations(gold, selected, segments)
        rendered = _render_selected(segments, selected)
        ranked = selected_ranker(segments, rendered, findings, selected, sample_size=0)
        ranked_by_id = {item["id"]: item for item in ranked}
        if set(ranked_by_id) != {segment["id"] for segment in segments}:
            raise _error("ranker did not return every evaluation segment")
        ranked_order = sorted(
            segments, key=lambda segment: (-ranked_by_id[segment["id"]]["score"], segment["id"])
        )
        ranked_curves.append(_curve(ranked_order, residual))
        baseline_curves.append(_curve(segments, residual))
        all_residual.extend({**item, "document_id": document["id"]} for item in residual)
        document_residuals[-1]["residual_misses"] = len(residual)
        document_residuals[-1]["residual_categories"] = dict(
            Counter(item["category"] for item in residual)
        )

    category_metrics = {}
    for category, counts in sorted(category_counts.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        category_metrics[category] = {
            "gold": counts["gold"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _metric(tp, tp + fp),
            "recall": _metric(tp, tp + fn),
        }
    fp = total_predictions - total_tp
    detection = {
        "gold": total_gold,
        "predictions": total_predictions,
        "tp": total_tp,
        "fp": fp,
        "fn": total_gold - total_tp,
        "precision": _metric(total_tp, total_tp + fp),
        "recall": _metric(total_tp, total_gold),
        "partial_overlap_predictions": total_partial,
        "residual_disclosures": len(all_residual),
        "false_positives_per_1000_words": round(fp * 1000 / total_words, 6) if total_words else 0.0,
        "by_category": category_metrics,
        "document_residual_misses": document_residuals,
    }
    return {
        "schema_version": 1,
        "corpus": {
            "documents": len(documents),
            "segments": sum(len(document["segments"]) for document in documents),
            "words": total_words,
            "scope": "synthetic evaluation corpus; not representative calibration data",
        },
        "detection": detection,
        "replacement_quality": dict(replacement_totals),
        "review_only_candidates": review_candidates,
        "ranking": {
            "undetected_annotations": len(all_residual),
            "ranked": {
                "curve": ranked_curves,
                "effort_by_document": [_effort(curve) for curve in ranked_curves],
                "method": "retained-evidence heuristic; not a calibrated probability"
                if ranker is None
                else "supplied comparison ranker",
            },
            "unranked_source_order": {
                "curve": baseline_curves,
                "effort_by_document": [_effort(curve) for curve in baseline_curves],
                "method": "original segment order baseline",
            },
        },
        "caveats": [
            "Synthetic examples are not representative of real-world data and do not set acceptance thresholds.",
            "No process-meaning or business-utility score is inferred from span metrics.",
            "Ranking scores and early-stopping curves are evaluation signals, not safety or anonymity proof.",
        ],
    }
