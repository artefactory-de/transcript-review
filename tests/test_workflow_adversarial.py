"""Adversarial workflow checks for review import and release boundaries."""

from pathlib import Path
from unittest.mock import patch

import pytest
from openpyxl import load_workbook

from transcript_anonymizer import workflow
from transcript_anonymizer.detection import Detector, load_policy
from transcript_anonymizer.documents import read_docx, write_docx
from transcript_anonymizer.errors import AnonymizerError


def _prepared(tmp_path: Path):
    policy = load_policy(None)
    policy["aliases"] = [{"entity_key": "person-a", "aliases": ["Anna Beispiel"]}]
    detector = Detector(policy, rules_only=True)
    source = tmp_path / "input.docx"
    write_docx(
        [
            {
                "id": "s000001",
                "text": "Anna Beispiel: anna@example.invalid",
                "locator": {"kind": "paragraph", "index": 0},
            },
            {
                "id": "s000002",
                "text": "The approval threshold is 100000 EUR.",
                "locator": {"kind": "paragraph", "index": 1},
            },
            {
                "id": "s000003",
                "text": "No identifying content here.",
                "locator": {"kind": "paragraph", "index": 2},
            },
        ],
        source,
    )
    run = tmp_path / "run"
    return run, detector, workflow.prepare(source, run, policy, detector)


def _edit(
    result: dict,
    tmp_path: Path,
    *,
    sheet: str | None = None,
    record_id: str | None = None,
    edits: dict | None = None,
    signoff: str | None = None,
) -> Path:
    book = load_workbook(result["workbook"])
    if signoff is not None:
        ws = book["Replacements"]
        cell = next(cell for row in ws for cell in row if cell.value == "Signed off by")
        ws.cell(cell.row, cell.column + 1).value = signoff
    if sheet is not None:
        ws = book[sheet]
        header = next(row for row in ws if row[0].value == "id")
        columns = {cell.value: cell.column for cell in header}
        row = next(row for row in ws if row[0].value == record_id)
        for key, value in (edits or {}).items():
            ws.cell(row[0].row, columns[key]).value = value
    path = tmp_path / f"edited-{len(list(tmp_path.glob('edited-*.xlsx')))}.xlsx"
    book.save(path)
    book.close()
    return path


def test_conflicting_entity_and_passage_edits_are_atomic(tmp_path: Path):
    run, detector, result = _prepared(tmp_path)
    state, _ = workflow._load(run)
    group_id = state["view"]["groups"][0]["id"]
    passage_id = state["view"]["passages"][0]["id"]
    workbook = _edit(
        result,
        tmp_path,
        sheet="Replacements",
        record_id=group_id,
        edits={"action": "replace", "replacement": "PERSON_A"},
    )
    # Apply the passage edit to the same workbook. The import must reject the
    # conflict before publishing a new revision.
    book = load_workbook(workbook)
    ws = book["Ranked passages"]
    header = {cell.value: cell.column for cell in next(row for row in ws if row[0].value == "id")}
    row = next(row for row in ws if row[0].value == passage_id)[0].row
    ws.cell(row, header["action"]).value = "replace"
    ws.cell(row, header["replacement"]).value = "manual replacement"
    ws.cell(row, header["outcome"]).value = "miss"
    book.save(workbook)
    book.close()
    pointer = (run / "current.json").read_bytes()
    with pytest.raises(AnonymizerError, match="Conflicting passage"):
        workflow.import_review(run, workbook, detector)
    assert (run / "current.json").read_bytes() == pointer
    assert not (run / ".lock").exists()


def test_replacement_text_is_scanned_before_commit(tmp_path: Path):
    run, detector, result = _prepared(tmp_path)
    state, _ = workflow._load(run)
    group_id = state["view"]["groups"][0]["id"]
    workbook = _edit(
        result,
        tmp_path,
        sheet="Replacements",
        record_id=group_id,
        edits={"action": "replace", "replacement": "contact me at leaked@example.invalid"},
        signoff="Reviewer",
    )
    pointer = (run / "current.json").read_bytes()
    with pytest.raises(AnonymizerError, match="replacement contains detected PII"):
        workflow.import_review(run, workbook, detector)
    assert (run / "current.json").read_bytes() == pointer
    assert not (run / ".lock").exists()


def test_changed_alias_policy_cannot_bypass_rescan_binding(tmp_path):
    run, detector, result = _prepared(tmp_path)
    state, _ = workflow._load(run)
    workbook = _edit(
        result,
        tmp_path,
        sheet="Replacements",
        record_id=state["view"]["groups"][0]["id"],
        edits={"action": "retain"},
    )
    altered = load_policy(None)
    altered["aliases"] = []
    other = Detector(altered, rules_only=True)
    assert other.metadata == detector.metadata  # Metadata alone is insufficient.
    before = (run / "current.json").read_bytes()
    with pytest.raises(AnonymizerError, match="policy differs"):
        workflow.import_review(run, workbook, other)
    assert (run / "current.json").read_bytes() == before


def test_clean_review_state_survives_unrelated_candidate_change(tmp_path: Path):
    run, detector, result = _prepared(tmp_path)
    state, _ = workflow._load(run)
    # Inspect the unmodified third segment, then change the entity group in
    # the first segment. These candidate bytes are intentionally disjoint.
    first_passage = state["view"]["passages"][-1]["id"]
    clean = _edit(
        result,
        tmp_path,
        sheet="Ranked passages",
        record_id=first_passage,
        edits={"outcome": "clean"},
        signoff="Reviewer",
    )
    signed = workflow.import_review(run, clean, detector)
    signed_state, _ = workflow._load(run)
    assert signed_state["view"]["passages"][-1]["outcome"] == "clean"

    # A later edit changes a different source segment. The inspected passage
    # is still byte-identical and its clean state must remain recorded.
    group_id = signed_state["view"]["groups"][0]["id"]
    changed = _edit(
        signed,
        tmp_path,
        sheet="Replacements",
        record_id=group_id,
        edits={"action": "replace", "replacement": "PERSON_A"},
    )
    workflow.import_review(run, changed, detector)
    state_after, _ = workflow._load(run)
    assert state_after["view"]["passages"][-1]["outcome"] == "clean"


def test_passage_correction_resets_only_changed_review_state(tmp_path: Path):
    run, detector, result = _prepared(tmp_path)
    state, _ = workflow._load(run)
    passage_ids = [row["id"] for row in state["view"]["passages"]]
    clean = _edit(
        result,
        tmp_path,
        sheet="Ranked passages",
        record_id=passage_ids[1],
        edits={"outcome": "clean"},
        signoff="Reviewer",
    )
    signed = workflow.import_review(run, clean, detector)
    corrected = _edit(
        signed,
        tmp_path,
        sheet="Ranked passages",
        record_id=passage_ids[0],
        edits={"outcome": "miss", "action": "replace", "replacement": "redacted"},
    )
    workflow.import_review(run, corrected, detector)
    after, _ = workflow._load(run)
    by_id = {row["id"]: row for row in after["view"]["passages"]}
    assert by_id[passage_ids[0]]["outcome"] == "uninspected"
    assert by_id[passage_ids[1]]["outcome"] == "clean"


def test_cli_signoff_only_import_does_not_initialize_detector(tmp_path: Path, capsys):
    run, _, result = _prepared(tmp_path)
    workbook = _edit(result, tmp_path, signoff="CLI reviewer")
    with patch(
        "transcript_anonymizer.cli.DeferredDetector._get",
        side_effect=AssertionError("model loaded"),
    ):
        from transcript_anonymizer.cli import main

        assert main(["review", "import", str(workbook), "--run", str(run)]) == 0
    assert "import_result" in capsys.readouterr().out


def test_export_remains_allowlisted_after_reviewer_free_text(tmp_path: Path):
    run, detector, result = _prepared(tmp_path)
    state, _ = workflow._load(run)
    group_id = state["view"]["groups"][0]["id"]
    workbook = _edit(
        result,
        tmp_path,
        sheet="Replacements",
        record_id=group_id,
        edits={"note": "Sensitive reviewer rationale with Anna Beispiel"},
        signoff="Reviewer Name",
    )
    workflow.import_review(run, workbook, detector)
    out = tmp_path / "export"
    workflow.export(run, out)
    assert {path.name for path in out.iterdir()} == {"transcript.docx", "manifest.json"}
    manifest = (out / "manifest.json").read_text()
    assert "Reviewer Name" not in manifest
    assert "Anna Beispiel" not in manifest
    assert "Sensitive reviewer rationale" not in manifest
    assert "Anna Beispiel" not in " ".join(
        segment["text"] for segment in read_docx(out / "transcript.docx")["segments"]
    )
