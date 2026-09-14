"""Bounded, protected XLSX review workbook.

The workbook is deliberately a transport format, not an integrity boundary.  The
writer makes the common Excel editing path safe and :func:`read_workbook` checks
the complete file against the trusted local view before returning any decisions.
"""

from __future__ import annotations

import math
import re
import textwrap
import zipfile
from copy import copy
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import Cell
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill, Protection
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet

from .errors import AnonymizerError

SCHEMA_VERSION = 1
MAX_CELL_CHARS = 32_767
MAX_XLSX_BYTES = 50 * 1024 * 1024
MAX_ZIP_MEMBERS = 2_000
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_MEMBER_BYTES = 50 * 1024 * 1024
MAX_EXPANSION_RATIO = 100
MAX_SHEET_ROWS = 100_000
MAX_SHEET_COLUMNS = 64

_HEADER_ROW = 9
_META_LABELS = {
    1: "Workbook metadata",
    2: "Schema version",
    3: "Run ID",
    4: "Revision",
    5: "Candidate SHA-256",
}
_PROTECTED_PASSWORD = "review"
_FORMULA_RE = re.compile(r"^\s*=", re.ASCII)
_CELL_REF_RE = re.compile(rb"<c\b[^>]*\br=[\"']([A-Z]+[0-9]+)[\"']")
_DIMENSION_REF_RE = re.compile(rb"<dimension\b[^>]*\bref=[\"']([^\"']+)[\"']")

_GROUP_COLUMNS = (
    "id", "source", "aliases", "category", "token", "count", "context",
    "action", "replacement", "note",
)
_PASSAGE_COLUMNS = (
    "id", "source", "text", "score", "reasons", "outcome", "action", "replacement", "note",
)
_OCCURRENCE_COLUMNS = (
    "id", "group_id", "segment_id", "source", "context", "action", "replacement", "note",
)
_GROUP_ACTIONS = ("accept", "retain", "replace", "remove")
_PASSAGE_OUTCOMES = ("uninspected", "clean", "miss")
_PASSAGE_ACTIONS = ("keep", "replace", "remove")
_OCCURRENCE_ACTIONS = ("inherit", "retain", "replace", "remove")
_EDITABLE = {
    "Replacements": {"action", "replacement", "note"},
    "Ranked passages": {"outcome", "action", "replacement", "note"},
    "Occurrences": {"action", "replacement", "note"},
}

_DECISION_HELP = {
    ("Replacements", "action"): {
        "accept": "Use the proposed token for the group's occurrences (default).",
        "retain": "Keep the original identifying text visible. Use only for a false positive or deliberate retention.",
        "replace": "Use your neutral text from replacement instead of the proposed token for this group.",
        "remove": "Delete the matched text for this group; do not delete the surrounding passage.",
    },
    ("Ranked passages", "outcome"): {
        "uninspected": "You have not reviewed this passage (default). It may remain unread at sign-off.",
        "clean": "You reviewed the candidate text and found no remaining PII. This records your review, not a guarantee.",
        "miss": "You found remaining PII. Also choose action replace or remove and supply the correction.",
    },
    ("Ranked passages", "action"): {
        "keep": "Make no whole-passage correction (default). Existing redactions remain; this does not restore the original.",
        "replace": "Copy the candidate from text into replacement. Edit every missed identifier there; keep all other wording and existing tokens. This replaces the ENTIRE passage, not a substring. Import rescans it.",
        "remove": "Delete the ENTIRE passage, including its non-PII content.",
    },
    ("Occurrences", "action"): {
        "inherit": "Follow the Replacements group decision for this occurrence (default).",
        "retain": "Override the group for this occurrence only: keep its original text visible.",
        "replace": "Override the group for this occurrence only: use the neutral text in replacement.",
        "remove": "Override the group for this occurrence only: delete the matched text.",
    },
}
_PROMPTS = {
    ("Replacements", "H"): "accept: proposed token; retain: original text visible; replace: your replacement text; remove: delete matched text. Occurrence overrides take precedence. See Start here.",
    ("Ranked passages", "F"): "uninspected: not reviewed; clean: reviewed, no remaining PII found; miss: PII found, so also set action to replace or remove. See Start here.",
    ("Ranked passages", "G"): "replace: copy text into replacement, edit all missed PII, keep the rest and existing tokens. Set outcome=miss. This replaces the ENTIRE passage. keep: unchanged; remove: delete ENTIRE passage. See Start here.",
    ("Occurrences", "F"): "inherit: follow group; retain: original text visible here; replace: your text here; remove: delete this match. Overrides affect only this occurrence. See Start here.",
}
_SHEET_TASKS = {
    "Replacements": "1. Review groups here. accept uses token; retain exposes the original; replace uses replacement; remove deletes matches. Occurrences can override a group. Leave replacement blank unless action is replace.",
    "Ranked passages": "2. For missed PII: copy text into replacement, edit every missed item, keep the rest and existing tokens. Set outcome=miss and action=replace. Import the ENTIRE corrected passage. source is the original; text is the candidate. See Start here.",
    "Occurrences": "Optional: inspect individual matches or override a group for one occurrence. inherit follows the group; other actions override it. No need to review every occurrence.",
}
_FIELD_HELP = {
    "replacement": "Enter plain text only when action is replace; otherwise leave blank. In Ranked passages: copy text (not source) here, edit all missed identifiers and preserve the rest plus existing tokens. Supply the entire passage, not only a name. Save, close Excel and import; corrections are rescanned and clear sign-off.",
    "note": "Optional explanation for the reviewer record. Notes do not change transcript text or approve the candidate.",
    "source": "Source text before redaction. This can contain PII and must stay in the local review workspace.",
    "text": "Current candidate passage after redaction. Inspect this for remaining PII. Do not edit this locked field; use action and replacement.",
    "token": "Proposed pseudonym used by accept. To change it, choose replace and enter neutral text in replacement.",
    "count": "Number of matched occurrences in this group. Individual occurrence overrides may change how each is treated.",
    "score": "Heuristic priority for reviewing retained text, highest first. It is not a probability or proof that low-ranked text is safe.",
    "reasons": "Signals behind the review priority. remainder_sample marks an optional sample from lower-ranked passages.",
    "aliases": "Linked variants for this group. Use Occurrences for exceptions; do not edit this generated list.",
}


def _write_instructions(wb: Workbook, view: dict) -> None:
    ws = wb.create_sheet("Start here", 0)
    rows = [
        ("Review this transcript", "Instructions and decision reference"),
        ("Input file", view.get("source_filename") or "See the workbook filename."),
        ("What to redact", "PII means personal information that can identify someone. Review for PII only; preserve ordinary process facts, business thresholds and organization/system names unless they identify a person."),
        ("Keep this workbook local", "It contains original PII and mappings. Only the approved DOCX and minimal export manifest are intended for transfer."),
        ("1. Review Replacements", "Start with aggregated groups. Correct false positives or proposed replacements. Defaults already apply; you do not need to click accept on every row."),
        ("2. Review Ranked passages", "Inspect candidate text in descending score order for missed PII, including passages already partly redacted. Mark inspected rows clean or miss. The score is not a calibrated probability."),
        ("When to stop", "You can stop when you consistently find no further misses. Leave unread rows uninspected. No fixed clean-streak threshold is enforced; a clean streak does not certify the unread remainder."),
        ("3. Make corrections", "Edit yellow cells only. Choose action from its dropdown. Fill replacement only for replace; otherwise clear it. Use neutral text without PII. Notes are optional and do not change the transcript."),
        ("Passage columns: source / text", "source is the original passage. text is the current candidate after redactions. They are identical when no change was proposed or applied in that passage; this does not mean it is free of PII. Review text for remaining PII."),
        ("Correct one or more missed items", "On Ranked passages, copy the entire text cell (Ctrl+C). Select replacement on the SAME row and paste as values (Home > Paste > Values), preserving the editable cell formatting. Edit that copied passage: replace every missed identifier with a neutral label such as [PERSON] or [PHONE]. Keep all other wording and existing pseudonym tokens unchanged. Do not copy source, which may restore already-redacted PII."),
        ("Finish the passage correction", "Set outcome to miss and action to replace on that row. replacement must contain the ENTIRE corrected passage, not only the names or replacement tokens. You can correct several items in this one cell. Save the XLSX, close Excel and import it in the application. Import rescans the correction and clears sign-off; open the updated workbook before continuing."),
        ("Example: two missed items", "If text says: 'Ask Ada Example to call 030 123456; [PERSON_1] will join.', replacement could be: 'Ask [PERSON] to call [PHONE]; [PERSON_1] will join.' Both missed items are replaced, and the other wording and existing [PERSON_1] token stay unchanged. Set outcome=miss and action=replace."),
        ("Optional Occurrences", "Use this tab, when present, for context or an exception affecting just one match. Its explicit decisions override the group. Reviewing every occurrence is not required."),
        ("Avoid conflicting edits", "Do not change a group/occurrence and replace or remove the same passage in one import. Import one set of corrections first, then open the updated workbook."),
        ("4. Save and import", "Save as XLSX locally and close Excel. In the application choose Import saved XLSX. Changing this workbook does not change the candidate until import succeeds. Word edits are not imported."),
        ("5. Reopen after corrections", "Import creates a new revision. Open its workbook through the application; do not keep editing an older copy. Content changes clear sign-off, even if you filled it in with the corrections."),
        ("6. Sign off and export", "When finished, enter your name in Replacements!B6 (Signed off by), save, close Excel and import. Sign-off accepts the current decisions, including defaults and any unread passages. Then export in the application."),
        ("Editable vs generated", "Yellow cells are editable; all other cells and sheet structure are protected. You can resize columns by dragging a column-header boundary without unprotecting the sheet. Do not add/delete rows, rename tabs or paste formulas. Invalid or stale edits are rejected without partial import."),
        ("Decision values", "The same word can have different scope on different tabs. Use the definitions below."),
    ]
    for (sheet, field), choices in _DECISION_HELP.items():
        rows.append((f"{sheet}: {field}", "Applies only to this tab and field."))
        rows.extend((value, meaning) for value, meaning in choices.items())
    rows.extend((field, meaning) for field, meaning in _FIELD_HELP.items())
    for index, values in enumerate(rows, 1):
        for column, value in enumerate(values, 1):
            cell = ws.cell(index, column)
            _set_literal(cell, value)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.protection = Protection(locked=True)
        ws.cell(index, 1).font = Font(bold=True)
        lines = max(len(textwrap.wrap(str(value), width=width)) for value, width in zip(values, (28, 92)))
        ws.row_dimensions[index].height = max(30, 15 * lines + 12)
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.font = Font(bold=True, color="FFFFFF")
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 100
    ws.freeze_panes = "A3"
    ws.sheet_view.showGridLines = False
    ws.protection.sheet = True
    ws.protection.set_password(_PROTECTED_PASSWORD)
    wb.active = 0


def _error(message: str) -> AnonymizerError:
    """Create an error whose message contains no source-derived values."""
    return AnonymizerError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _text(value: Any, *, field: str, allow_blank: bool = True) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise _error(f"invalid text in {field}")
    if not allow_blank and not value:
        raise _error(f"missing value in {field}")
    if len(value.encode("utf-16-le")) // 2 > MAX_CELL_CHARS:
        raise _error(f"text exceeds Excel cell limit in {field}")
    return value


def _safe_key(value: Any, field: str) -> str:
    value = _text(value, field=field, allow_blank=False)
    if value.startswith("="):
        raise _error(f"formula-like value in {field}")
    return value


def _row_schema(row: Any, columns: tuple[str, ...], *, sheet: str, index: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise _error(f"invalid row structure in {sheet} row {index}")
    extras = set(row) - set(columns)
    if extras:
        raise _error(f"unexpected field in {sheet} row {index}")
    result = dict(row)
    for name in columns:
        if name not in result:
            if name in {"action", "outcome", "replacement", "note"}:
                result[name] = "" if name in {"replacement", "note"} else (
                    "uninspected" if name == "outcome" else (
                        "accept" if sheet == "Replacements" else
                        "keep" if sheet == "Ranked passages" else "inherit"
                    )
                )
            else:
                raise _error(f"missing field in {sheet} row {index}")
    return result


def _validate_view(view: Any) -> dict[str, Any]:
    if not isinstance(view, dict):
        raise _error("workbook view must be an object")
    for key in ("schema_version", "run_id", "revision", "candidate_sha256", "groups", "passages"):
        if key not in view:
            raise _error(f"missing workbook field {key}")
    if view["schema_version"] != SCHEMA_VERSION:
        raise _error("unsupported workbook schema")
    run_id = _safe_key(view["run_id"], "run_id")
    revision = view["revision"]
    if not _is_int(revision) or revision < 0:
        raise _error("invalid workbook revision")
    candidate = _safe_key(view["candidate_sha256"], "candidate_sha256")
    signed = _text(view.get("signed_off_by", ""), field="signed_off_by")
    groups = view["groups"]
    passages = view["passages"]
    occurrences = view.get("occurrences", [])
    if not isinstance(groups, list) or not isinstance(passages, list) or not isinstance(occurrences, list):
        raise _error("workbook rows must be lists")

    normalized: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "revision": revision,
        "candidate_sha256": candidate,
        "signed_off_by": signed,
        "groups": [], "passages": [], "occurrences": [],
    }
    if "source_filename" in view:
        normalized["source_filename"] = _text(view["source_filename"], field="source_filename")
    for i, row in enumerate(groups, 1):
        r = _row_schema(row, _GROUP_COLUMNS, sheet="Replacements", index=i)
        r["id"] = _safe_key(r["id"], f"Replacements row {i} id")
        for field in ("source", "aliases", "category", "token", "context", "replacement", "note"):
            r[field] = _text(r[field], field=f"Replacements row {i} {field}")
        if not _is_int(r["count"]) or r["count"] < 0:
            raise _error(f"invalid count in Replacements row {i}")
        if r["action"] not in _GROUP_ACTIONS:
            raise _error(f"invalid action in Replacements row {i}")
        if r["action"] == "replace" and not r["replacement"]:
            raise _error(f"replacement required in Replacements row {i}")
        if r["action"] != "replace" and r["replacement"]:
            raise _error(f"unexpected replacement in Replacements row {i}")
        normalized["groups"].append(r)
    for i, row in enumerate(passages, 1):
        r = _row_schema(row, _PASSAGE_COLUMNS, sheet="Ranked passages", index=i)
        r["id"] = _safe_key(r["id"], f"Ranked passages row {i} id")
        for field in ("source", "text", "reasons", "replacement", "note"):
            r[field] = _text(r[field], field=f"Ranked passages row {i} {field}")
        if not _is_number(r["score"]):
            raise _error(f"invalid score in Ranked passages row {i}")
        if r["outcome"] not in _PASSAGE_OUTCOMES:
            raise _error(f"invalid outcome in Ranked passages row {i}")
        if r["action"] not in _PASSAGE_ACTIONS:
            raise _error(f"invalid action in Ranked passages row {i}")
        if r["action"] == "replace" and not r["replacement"]:
            raise _error(f"replacement required in Ranked passages row {i}")
        if r["action"] != "replace" and r["replacement"]:
            raise _error(f"unexpected replacement in Ranked passages row {i}")
        if r["outcome"] == "miss" and r["action"] == "keep":
            raise _error(f"miss requires correction in Ranked passages row {i}")
        normalized["passages"].append(r)
    group_ids = {r["id"] for r in normalized["groups"]}
    if len(group_ids) != len(normalized["groups"]):
        raise _error("duplicate replacement group ID")
    passage_ids = {r["id"] for r in normalized["passages"]}
    if len(passage_ids) != len(normalized["passages"]):
        raise _error("duplicate passage ID")
    for i, row in enumerate(occurrences, 1):
        r = _row_schema(row, _OCCURRENCE_COLUMNS, sheet="Occurrences", index=i)
        r["id"] = _safe_key(r["id"], f"Occurrences row {i} id")
        r["group_id"] = _safe_key(r["group_id"], f"Occurrences row {i} group_id")
        r["segment_id"] = _safe_key(r["segment_id"], f"Occurrences row {i} segment_id")
        for field in ("source", "context", "replacement", "note"):
            r[field] = _text(r[field], field=f"Occurrences row {i} {field}")
        if r["action"] not in _OCCURRENCE_ACTIONS:
            raise _error(f"invalid action in Occurrences row {i}")
        if r["action"] == "replace" and not r["replacement"]:
            raise _error(f"replacement required in Occurrences row {i}")
        if r["action"] != "replace" and r["replacement"]:
            raise _error(f"unexpected replacement in Occurrences row {i}")
        if r["group_id"] not in group_ids:
            raise _error(f"foreign group ID in Occurrences row {i}")
        normalized["occurrences"].append(r)
    occurrence_ids = {r["id"] for r in normalized["occurrences"]}
    if len(occurrence_ids) != len(normalized["occurrences"]):
        raise _error("duplicate occurrence ID")
    return normalized


def _set_literal(cell: Cell, value: Any) -> None:
    """Write a value without allowing Excel/openpyxl to infer a formula."""
    if isinstance(value, str):
        _text(value, field="cell")
        cell.value = value
        cell.data_type = "s"
        cell.hyperlink = None
    else:
        cell.value = value


def _cell_text(cell: Cell, *, sheet: str, ref: str) -> str:
    if cell.data_type == "f":
        raise _error(f"formula not allowed at {sheet}!{ref}")
    if cell.value is None:
        return ""
    if isinstance(cell.value, str):
        return _text(cell.value, field=f"{sheet}!{ref}")
    raise _error(f"text required at {sheet}!{ref}")


def _cell_any(cell: Cell, *, sheet: str, ref: str) -> Any:
    if cell.data_type == "f":
        raise _error(f"formula not allowed at {sheet}!{ref}")
    if cell.value is None:
        return ""
    if isinstance(cell.value, str):
        return _text(cell.value, field=f"{sheet}!{ref}")
    return cell.value


def _write_metadata(ws: Worksheet, view: dict[str, Any], *, signoff: bool) -> None:
    for row, label in _META_LABELS.items():
        _set_literal(ws.cell(row, 1), label)
        ws.cell(row, 1).font = Font(bold=True)
        ws.cell(row, 1).protection = Protection(locked=True)
    values = {2: view["schema_version"], 3: view["run_id"], 4: view["revision"],
              5: view["candidate_sha256"]}
    for row, value in values.items():
        _set_literal(ws.cell(row, 2), value)
        ws.cell(row, 2).protection = Protection(locked=True)
    if signoff:
        _set_literal(ws.cell(6, 1), "Signed off by")
        ws.cell(6, 1).font = Font(bold=True)
        _set_literal(ws.cell(6, 2), view["signed_off_by"])
        ws.cell(6, 2).protection = Protection(locked=False)
        ws.cell(6, 2).fill = PatternFill("solid", fgColor="FFF2CC")
    ws.cell(7, 1).value = "Read Start here first. Edit yellow cells only. Save as XLSX, close Excel, then import in the application. After import, open the updated workbook."
    ws.cell(7, 1).font = Font(italic=True, color="666666")
    ws.merge_cells(start_row=7, start_column=1, end_row=7, end_column=10)
    for c in ws[7]:
        c.protection = Protection(locked=True)
    ws.cell(7, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[7].height = 30
    columns = {"Replacements": _GROUP_COLUMNS, "Ranked passages": _PASSAGE_COLUMNS,
               "Occurrences": _OCCURRENCE_COLUMNS}[ws.title]
    ws.merge_cells(start_row=8, start_column=1, end_row=8, end_column=len(columns))
    ws.cell(8, 1).value = _SHEET_TASKS[ws.title]
    ws.cell(8, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[8].height = 36
    if signoff:
        ws.cell(6, 2).comment = Comment("Enter your name only after review is complete. Save, close Excel and import. Content-changing corrections clear sign-off and require review of the updated workbook.", "Review guide")


def _setup_sheet(ws: Worksheet, columns: tuple[str, ...], rows: list[dict[str, Any]], *, view: dict[str, Any], signoff: bool) -> None:
    _write_metadata(ws, view, signoff=signoff)
    for col, name in enumerate(columns, 1):
        cell = ws.cell(_HEADER_ROW, col)
        _set_literal(cell, name)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.protection = Protection(locked=True)
        choices = _DECISION_HELP.get((ws.title, name))
        help_text = "\n".join(f"{key}: {value}" for key, value in choices.items()) if choices else _FIELD_HELP.get(name)
        if help_text:
            cell.comment = Comment(help_text, "Review guide")
            cell.comment.width = 440
            cell.comment.height = 180
    for offset, row in enumerate(rows, _HEADER_ROW + 1):
        for col, name in enumerate(columns, 1):
            value = row[name]
            cell = ws.cell(offset, col)
            _set_literal(cell, value)
            editable = name in _EDITABLE[ws.title]
            cell.protection = Protection(locked=not editable)
            if editable:
                cell.fill = PatternFill("solid", fgColor="FFF2CC")
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            elif name in {"source", "text", "context"}:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    last = max(_HEADER_ROW, _HEADER_ROW + len(rows))
    ws.auto_filter.ref = f"A{_HEADER_ROW}:{_col_letter(len(columns))}{last}"
    ws.freeze_panes = f"A{_HEADER_ROW + 1}"
    ws.sheet_view.showGridLines = False
    widths = [18, 34, 28, 18, 24, 12, 45, 16, 28, 34]
    for col, width in enumerate(widths[:len(columns)], 1):
        ws.column_dimensions[_col_letter(col)].width = width
    ws.protection.sheet = True
    ws.protection.set_password(_PROTECTED_PASSWORD)
    ws.protection.autoFilter = True
    ws.protection.sort = True


def _col_letter(n: int) -> str:
    result = ""
    while n:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


def _cell_ref_bounds(reference: bytes) -> tuple[int, int] | None:
    match = re.fullmatch(rb"([A-Z]+)([0-9]+)", reference)
    if not match:
        return None
    column = 0
    for character in match.group(1):
        column = column * 26 + character - 64
    row = int(match.group(2))
    return row, column


def _check_sheet_xml_bounds(data: bytes) -> None:
    """Reject sparse far-away cells before openpyxl can materialize dimensions."""
    references = list(_CELL_REF_RE.findall(data))
    references.extend(
        endpoint
        for dimension in _DIMENSION_REF_RE.findall(data)
        for endpoint in dimension.split(b":")
    )
    for reference in references:
        bounds = _cell_ref_bounds(reference)
        if bounds is None or bounds[0] < 1 or bounds[0] > MAX_SHEET_ROWS or bounds[1] < 1 or bounds[1] > MAX_SHEET_COLUMNS:
            raise _error("workbook worksheet dimensions exceed the supported limit")


def _add_validation(ws: Worksheet, column: str, start: int, end: int, choices: tuple[str, ...]) -> None:
    if end < start:
        return
    dv = DataValidation(type="list", formula1='"' + ",".join(choices) + '"', allow_blank=False)
    dv.errorTitle = "Invalid choice"
    dv.error = "Choose one of the listed values."
    dv.promptTitle = "Review decision"
    dv.prompt = _PROMPTS[(ws.title, column)]
    dv.showErrorMessage = True
    dv.showInputMessage = True
    ws.add_data_validation(dv)
    dv.add(f"{column}{start}:{column}{end}")


def write_workbook(view: dict, path: Path) -> None:
    """Write a bounded, protected review workbook at *path*."""
    normalized = _validate_view(view)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    replacements = wb.active
    replacements.title = "Replacements"
    ranked = wb.create_sheet("Ranked passages")
    _setup_sheet(replacements, _GROUP_COLUMNS, normalized["groups"], view=normalized, signoff=True)
    _setup_sheet(ranked, _PASSAGE_COLUMNS, normalized["passages"], view=normalized, signoff=False)
    _add_validation(replacements, "H", _HEADER_ROW + 1, _HEADER_ROW + len(normalized["groups"]), _GROUP_ACTIONS)
    _add_validation(ranked, "F", _HEADER_ROW + 1, _HEADER_ROW + len(normalized["passages"]), _PASSAGE_OUTCOMES)
    _add_validation(ranked, "G", _HEADER_ROW + 1, _HEADER_ROW + len(normalized["passages"]), _PASSAGE_ACTIONS)
    if normalized["occurrences"]:
        occurrences = wb.create_sheet("Occurrences")
        _setup_sheet(occurrences, _OCCURRENCE_COLUMNS, normalized["occurrences"], view=normalized, signoff=False)
        _add_validation(occurrences, "F", _HEADER_ROW + 1, _HEADER_ROW + len(normalized["occurrences"]), _OCCURRENCE_ACTIONS)
    _write_instructions(wb, normalized)
    wb.security.lockStructure = True
    wb.security.set_workbook_password(_PROTECTED_PASSWORD)
    wb.calculation.fullCalcOnLoad = False
    wb.calculation.forceFullCalc = False
    wb.calculation.calcMode = "manual"
    try:
        # Preserve wrapping and horizontal alignment, including metadata, numeric
        # cells and blank editable fields. Merged placeholders have no own content.
        for sheet in wb:
            # False permits column formatting (including width changes) while
            # sheet protection and each cell's locked/editable state stay intact.
            sheet.protection.formatColumns = False
            for row in sheet:
                for cell in row:
                    if isinstance(cell, Cell):
                        alignment = copy(cell.alignment)
                        alignment.vertical = "top"
                        cell.alignment = alignment
        wb.save(path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise _error("unable to write workbook") from exc


def _preflight(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _error("unable to read workbook") from exc
    if size <= 0 or size > MAX_XLSX_BYTES:
        raise _error("workbook file is outside the supported size limit")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ZIP_MEMBERS:
                raise _error("workbook contains too many package parts")
            total = 0
            for info in infos:
                name = info.filename.replace("\\", "/")
                if name.startswith("/") or ".." in name.split("/"):
                    raise _error("workbook contains an unsafe package path")
                if info.file_size > MAX_MEMBER_BYTES:
                    raise _error("workbook package part is too large")
                total += info.file_size
                if total > MAX_UNCOMPRESSED_BYTES:
                    raise _error("workbook expands beyond the supported size limit")
                if info.compress_size and info.file_size > info.compress_size * MAX_EXPANSION_RATIO:
                    raise _error("workbook package compression ratio is unsupported")
                lowered = name.lower()
                if lowered.endswith("vbaproject.bin") or "externallink" in lowered or "connections.xml" in lowered:
                    raise _error("workbook active content or external links are unsupported")
                if lowered.startswith("xl/worksheets/") and lowered.endswith(".xml"):
                    _check_sheet_xml_bounds(archive.read(info))
                if info.file_size and info.compress_size == 0:
                    raise _error("invalid workbook package part")
            for info in infos:
                if info.filename.endswith(".rels"):
                    data = archive.read(info).lower()
                    if b'targetmode="external"' in data or b"targetmode='external'" in data:
                        raise _error("workbook external links are unsupported")
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, AnonymizerError):
            raise
        raise _error("invalid workbook package") from exc


def _used_rows(ws: Worksheet, columns: tuple[str, ...]) -> list[int]:
    result = []
    for row in range(_HEADER_ROW + 1, ws.max_row + 1):
        values = [ws.cell(row, col).value for col in range(1, len(columns) + 1)]
        outside = [ws.cell(row, col).value for col in range(len(columns) + 1, ws.max_column + 1)]
        if any(value is not None for value in outside):
            raise _error(f"unexpected content at {ws.title} row {row}")
        if any(value is not None for value in values):
            result.append(row)
    return result


def _validate_metadata(ws: Worksheet, expected: dict[str, Any], *, signoff: bool) -> str:
    for row, label in _META_LABELS.items():
        if _cell_text(ws.cell(row, 1), sheet=ws.title, ref=f"A{row}") != label:
            raise _error(f"invalid metadata label at {ws.title}!A{row}")
    checks = ((2, expected["schema_version"]), (3, expected["run_id"]),
              (4, expected["revision"]), (5, expected["candidate_sha256"]))
    for row, value in checks:
        actual = _cell_any(ws.cell(row, 2), sheet=ws.title, ref=f"B{row}")
        if actual != value:
            raise _error(f"stale or changed metadata at {ws.title}!B{row}")
    if signoff:
        if _cell_text(ws.cell(6, 1), sheet=ws.title, ref="A6") != "Signed off by":
            raise _error(f"invalid sign-off label at {ws.title}!A6")
        signoff_cell = ws.cell(6, 2)
        value = _cell_text(signoff_cell, sheet=ws.title, ref="B6")
        return value
    if _cell_text(ws.cell(6, 2), sheet=ws.title, ref="B6") != "":
        raise _error(f"unexpected sign-off at {ws.title}!B6")
    return ""


def _validate_headers(ws: Worksheet, columns: tuple[str, ...]) -> None:
    actual = [_cell_text(ws.cell(_HEADER_ROW, col), sheet=ws.title, ref=f"{_col_letter(col)}{_HEADER_ROW}")
              for col in range(1, ws.max_column + 1)]
    # Empty trailing cells are normal, but every used header must be an exact known column.
    while actual and actual[-1] == "":
        actual.pop()
    if tuple(actual) != columns:
        raise _error(f"invalid columns in {ws.title}")
    for cell in ws._cells.values():  # openpyxl's populated-cell map avoids sparse expansion
        if cell.column > len(columns) and cell.value is not None:
            raise _error(f"unexpected content at {ws.title}!{cell.coordinate}")


def _read_rows(ws: Worksheet, columns: tuple[str, ...], expected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _validate_headers(ws, columns)
    row_numbers = _used_rows(ws, columns)
    if len(row_numbers) != len(expected_rows):
        raise _error(f"missing or extra rows in {ws.title}")
    expected_by_id = {row["id"]: row for row in expected_rows}
    if len(expected_by_id) != len(expected_rows):
        raise _error(f"duplicate trusted ID in {ws.title}")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row_number in row_numbers:
        values: dict[str, Any] = {}
        for col, name in enumerate(columns, 1):
            ref = f"{_col_letter(col)}{row_number}"
            values[name] = _cell_any(ws.cell(row_number, col), sheet=ws.title, ref=ref)
        row_id = values["id"]
        if not isinstance(row_id, str) or not row_id:
            raise _error(f"invalid ID at {ws.title}!A{row_number}")
        if row_id in seen:
            raise _error(f"duplicate ID at {ws.title}!A{row_number}")
        if row_id not in expected_by_id:
            raise _error(f"foreign ID at {ws.title}!A{row_number}")
        seen.add(row_id)
        trusted = expected_by_id[row_id]
        for name in columns:
            if name in _EDITABLE[ws.title]:
                continue
            if values[name] != trusted[name]:
                raise _error(f"immutable field changed at {ws.title}!{_col_letter(columns.index(name) + 1)}{row_number}")
        # Normalize blank editable cells and validate against the same schema as writing.
        result.append(values)
    if seen != set(expected_by_id):
        raise _error(f"missing ID in {ws.title}")
    return result


def read_workbook(path: Path, expected: dict) -> dict:
    """Validate and import an edited workbook against the trusted local view.

    The workbook is read completely before any result is returned.  Invalid files
    raise :class:`AnonymizerError`; callers can therefore preserve the previous run
    state without needing rollback writes here.
    """
    trusted = _validate_view(expected)
    path = Path(path)
    _preflight(path)
    try:
        wb = load_workbook(path, read_only=False, data_only=False, keep_links=False, keep_vba=False)
    except Exception as exc:
        raise _error("unable to parse workbook") from exc
    try:
        allowed = {"Start here", "Replacements", "Ranked passages", "Occurrences"}
        if set(wb.sheetnames) - allowed or "Replacements" not in wb.sheetnames or "Ranked passages" not in wb.sheetnames:
            raise _error("invalid workbook sheets")
        for ws in wb.worksheets:
            if ws.max_row > MAX_SHEET_ROWS or ws.max_column > MAX_SHEET_COLUMNS:
                raise _error("workbook worksheet dimensions exceed the supported limit")
            for cell in ws._cells.values():  # avoid iterating every coordinate in a sparse sheet
                if cell.data_type == "f":
                    raise _error(f"formula not allowed at {ws.title}!{cell.coordinate}")
        if "Start here" in wb.sheetnames and "source_filename" in trusted:
            source_label = _cell_text(wb["Start here"]["B2"], sheet="Start here", ref="B2")
            if source_label != (trusted["source_filename"] or "See the workbook filename."):
                raise _error("immutable input filename changed at Start here!B2")
        signed = _validate_metadata(wb["Replacements"], trusted, signoff=True)
        _validate_metadata(wb["Ranked passages"], trusted, signoff=False)
        groups = _read_rows(wb["Replacements"], _GROUP_COLUMNS, trusted["groups"])
        passages = _read_rows(wb["Ranked passages"], _PASSAGE_COLUMNS, trusted["passages"])
        if "Occurrences" in wb.sheetnames:
            occurrences = _read_rows(wb["Occurrences"], _OCCURRENCE_COLUMNS, trusted["occurrences"])
        elif trusted["occurrences"]:
            raise _error("missing Occurrences sheet")
        else:
            occurrences = []
        imported = {
            "schema_version": trusted["schema_version"],
            "run_id": trusted["run_id"],
            "revision": trusted["revision"],
            "candidate_sha256": trusted["candidate_sha256"],
            "signed_off_by": signed,
            "groups": groups,
            "passages": passages,
            "occurrences": occurrences,
        }
        # This checks types, allowed choices, replacements and miss corrections after
        # unprotecting or pasting values into Excel.
        imported = _validate_view(imported)
        # The reviewer may sort/filter rows. Return decisions in trusted source
        # order so downstream regeneration remains deterministic.
        by_group = {row["id"]: row for row in imported["groups"]}
        by_passage = {row["id"]: row for row in imported["passages"]}
        by_occurrence = {row["id"]: row for row in imported["occurrences"]}
        imported["groups"] = [by_group[row["id"]] for row in trusted["groups"]]
        imported["passages"] = [by_passage[row["id"]] for row in trusted["passages"]]
        imported["occurrences"] = [by_occurrence[row["id"]] for row in trusted["occurrences"]]
        return imported
    finally:
        wb.close()
