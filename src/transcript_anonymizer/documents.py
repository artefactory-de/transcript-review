"""Bounded DOCX extraction and clean reconstruction.

The document adapter deliberately does not use ``python-docx`` for input.  A
DOCX is an untrusted ZIP package and the small amount of WordprocessingML we
support is easier to account for explicitly than to infer from a high-level
object model.  The writer emits a new package and therefore cannot inherit
source relationships, properties, media, or other package parts.
"""

from __future__ import annotations

import os
import posixpath
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as OutputET

from defusedxml import ElementTree as DefusedET

from .errors import AnonymizerError

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
OFFICE_DOCUMENT_REL = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
)

# These are resource limits, not performance targets.  They leave ample room
# for the observed transcripts while making archive expansion and giant XML
# parts bounded before parsing.
MAX_PARTS = 4096
MAX_PART_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1000
MAX_SEGMENTS = 100_000
MAX_TEXT_BYTES = 16 * 1024 * 1024

_ID_RE = re.compile(r"^s\d{6,}$")
_XML_PARTS = {".xml", ".rels"}
_MEDIA_PREFIX = "word/media/"
_ACTIVE_PREFIXES = (
    "word/embeddings/",
    "word/activeX/",
    "word/charts/",
    "word/ink/",
    "word/diagrams/",
)
_KNOWN_REMOVED_PARTS = {
    "word/styles.xml",
    "word/numbering.xml",
    "word/settings.xml",
    "word/webSettings.xml",
    "word/fontTable.xml",
    "word/theme/theme1.xml",
    "docProps/core.xml",
    "docProps/app.xml",
    "docProps/custom.xml",
    "docProps/thumbnail.jpeg",
    "docProps/thumbnail.png",
    "word/commentsExtended.xml",
    "word/people.xml",
    "word/commentsIds.xml",
}
_STRUCTURAL_IGNORED = {
    "bookmarkStart",
    "bookmarkEnd",
    "commentRangeStart",
    "commentRangeEnd",
    "commentReference",
    "proofErr",
    "lastRenderedPageBreak",
    "permStart",
    "permEnd",
    "moveFromRangeStart",
    "moveFromRangeEnd",
    "moveToRangeStart",
    "moveToRangeEnd",
}
_BLOCKING_TAGS = {
    "ins",
    "del",
    "moveFrom",
    "moveTo",
    "fldSimple",
    "fldChar",
    "instrText",
    "delText",
    "sdt",
    "customXml",
    "txbxContent",
    "smartTag",
    "altChunk",
    "object",
    "oleObject",
    "control",
    "subDoc",
    "AlternateContent",
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _namespace(tag: str) -> str | None:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") and "}" in tag else None


def _is_word(element, name: str) -> bool:
    return _local(element.tag) == name and _namespace(element.tag) in {W_NS, W14_NS}


def _safe_error(message: str) -> AnonymizerError:
    # Keep this helper as a visible reminder that document text and filenames
    # must never be interpolated into user-facing failures.
    return AnonymizerError(message)


def _validate_member_name(name: str) -> None:
    if not name or "\x00" in name or "\\" in name or "//" in name:
        raise _safe_error("DOCX contains an unsafe package path")
    if name.startswith("/"):
        raise _safe_error("DOCX contains an unsafe package path")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts) or ":" in path.parts[0]:
        raise _safe_error("DOCX contains an unsafe package path")


def _read_package(path: Path) -> tuple[dict[str, bytes], list[str]]:
    try:
        archive = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise _safe_error("DOCX is not a readable ZIP package") from exc

    parts: dict[str, bytes] = {}
    names: list[str] = []
    total = 0
    try:
        infos = archive.infolist()
        if len(infos) > MAX_PARTS:
            raise _safe_error("DOCX package contains too many parts")
        for info in infos:
            name = info.filename
            _validate_member_name(name)
            if name in parts or name in names:
                raise _safe_error("DOCX package contains duplicate parts")
            names.append(name)
            if info.flag_bits & 0x1:
                raise _safe_error("Encrypted DOCX packages are not supported")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise _safe_error("DOCX contains a symbolic-link package entry")
            if info.is_dir():
                parts[name] = b""
                continue
            if info.file_size < 0 or info.file_size > MAX_PART_BYTES:
                raise _safe_error("DOCX package part exceeds the supported size")
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise _safe_error("DOCX package exceeds the supported expanded size")
            if info.file_size and (
                info.compress_size == 0
                or info.file_size > max(1, info.compress_size) * MAX_COMPRESSION_RATIO
            ):
                raise _safe_error("DOCX package has an unsafe compression ratio")
            try:
                with archive.open(info, "r") as stream:
                    data = stream.read(MAX_PART_BYTES + 1)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise _safe_error("DOCX package part could not be read") from exc
            if len(data) > MAX_PART_BYTES:
                raise _safe_error("DOCX package part exceeds the supported size")
            if len(data) != info.file_size:
                raise _safe_error("DOCX package part has an invalid size")
            parts[name] = data
    finally:
        archive.close()
    return parts, names


def _parse_xml(data: bytes, *, package_part: bool = False):
    if len(data) > MAX_PART_BYTES:
        raise _safe_error("DOCX XML part exceeds the supported size")
    try:
        root = DefusedET.fromstring(data)
    except Exception as exc:  # defusedxml raises several parser-specific errors
        raise _safe_error("DOCX contains malformed or unsafe XML") from exc
    if package_part and _namespace(root.tag) not in {REL_NS, CT_NS}:
        raise _safe_error("DOCX contains an invalid package metadata part")
    return root


def _xml_text(element) -> str:
    return "".join(element.itertext())


def _meaningful_text(element) -> bool:
    return bool(_xml_text(element).strip())


def _part_from_target(source: str | None, target: str) -> str:
    if not target or "\x00" in target or target.startswith("/") or "\\" in target:
        raise _safe_error("DOCX contains an unsafe relationship target")
    base = posixpath.dirname(source) if source else ""
    resolved = posixpath.normpath(posixpath.join(base, target))
    if resolved in {"", "."} or resolved == ".." or resolved.startswith("../"):
        raise _safe_error("DOCX contains an unsafe relationship target")
    return resolved


def _relationship_part(part: str) -> str:
    parent = posixpath.dirname(part)
    filename = posixpath.basename(part)
    return posixpath.join(parent, "_rels", filename + ".rels")


def _relationships(parts: dict[str, bytes], source: str | None) -> list[dict[str, str]]:
    rels_name = "_rels/.rels" if source is None else _relationship_part(source)
    data = parts.get(rels_name)
    if data is None:
        return []
    root = _parse_xml(data, package_part=True)
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for relation in list(root):
        if _local(relation.tag) != "Relationship":
            if _meaningful_text(relation):
                raise _safe_error("DOCX contains unsupported relationship metadata")
            continue
        rid = relation.attrib.get("Id", "")
        target = relation.attrib.get("Target", "")
        kind = relation.attrib.get("Type", "")
        mode = relation.attrib.get("TargetMode", "Internal")
        if not rid or rid in seen or not kind or not target:
            raise _safe_error("DOCX contains malformed relationships")
        seen.add(rid)
        if mode.lower() != "internal":
            raise _safe_error("DOCX contains an external relationship")
        target_part = _part_from_target(source, target)
        if target_part not in parts:
            raise _safe_error("DOCX relationship target is missing")
        result.append({"id": rid, "target": target_part, "type": kind})
    return result


def _content_types(parts: dict[str, bytes]) -> tuple[dict[str, str], dict[str, str]]:
    data = parts.get("[Content_Types].xml")
    if data is None:
        raise _safe_error("DOCX is missing package content types")
    root = _parse_xml(data, package_part=True)
    defaults: dict[str, str] = {}
    overrides: dict[str, str] = {}
    for child in list(root):
        kind = _local(child.tag)
        if kind == "Default":
            extension = child.attrib.get("Extension", "").lower()
            content_type = child.attrib.get("ContentType", "")
            if not extension or not content_type or extension in defaults:
                raise _safe_error("DOCX contains malformed content types")
            defaults[extension] = content_type
        elif kind == "Override":
            name = child.attrib.get("PartName", "")
            content_type = child.attrib.get("ContentType", "")
            if not name.startswith("/") or not content_type:
                raise _safe_error("DOCX contains malformed content types")
            part = name[1:]
            _validate_member_name(part)
            if part in overrides:
                raise _safe_error("DOCX contains duplicate content types")
            overrides[part] = content_type
        elif _meaningful_text(child):
            raise _safe_error("DOCX contains unsupported content type metadata")
    for part in parts:
        if part in {"[Content_Types].xml", "_rels/.rels"} or part.endswith("/"):
            continue
        if part not in overrides and posixpath.splitext(part)[1].lstrip(".").lower() not in defaults:
            raise _safe_error("DOCX part has no declared content type")
    return defaults, overrides


def _document_part(parts: dict[str, bytes]) -> str:
    rels = _relationships(parts, None)
    docs = [rel["target"] for rel in rels if rel["type"] == OFFICE_DOCUMENT_REL]
    if len(docs) != 1 or docs[0] != "word/document.xml":
        raise _safe_error("DOCX must contain one supported main document")
    if docs[0] not in parts:
        raise _safe_error("DOCX main document is missing")
    return docs[0]


def _assert_supported_tree(root) -> None:
    """Reject structures whose text or behavior we cannot faithfully retain."""

    for element in root.iter():
        name = _local(element.tag)
        namespace = _namespace(element.tag)
        if name in _BLOCKING_TAGS:
            raise _safe_error("DOCX contains unsupported text or field content")
        if name in {"drawing", "pict"} and any(
            # A picture can be removed by policy.  A text box inside one cannot
            # be reconstructed without risking a silent omission.
            _local(desc.tag) in {"t", "delText", "instrText", "txbxContent"}
            and _meaningful_text(desc)
            for desc in element.iter()
        ):
            raise _safe_error("DOCX contains unsupported text in a drawing")
        if namespace not in {None, W_NS, W14_NS} and _meaningful_text(element):
            # Math and vendor extensions may carry visible words; do not guess.
            raise _safe_error("DOCX contains unsupported text content")


def _paragraph_text(paragraph) -> str:
    output: list[str] = []

    def visit(element) -> None:
        name = _local(element.tag)
        if name in {"rPr", "pPr"}:
            return
        if name == "t" and _namespace(element.tag) in {W_NS, W14_NS}:
            output.append(element.text or "")
            return
        if name == "tab" and _namespace(element.tag) in {W_NS, W14_NS}:
            output.append("\t")
            return
        if name in {"br", "cr"} and _namespace(element.tag) in {W_NS, W14_NS}:
            output.append("\n")
            return
        if name in _STRUCTURAL_IGNORED:
            return
        for child in list(element):
            visit(child)

    visit(paragraph)
    value = "".join(output)
    if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise _safe_error("DOCX text segment exceeds the supported size")
    return value


def _segment(text: str, locator: dict) -> dict:
    return {"id": "", "text": text, "locator": locator}


def _extract_blocks(root, part: str, *, kind: str, table_counter: list[int], segment_list: list[dict]) -> None:
    for child in list(root):
        name = _local(child.tag)
        if name == "p":
            segment_list.append(_segment(_paragraph_text(child), {"part": part, "kind": kind}))
        elif name == "tbl":
            table_no = table_counter[0]
            table_counter[0] += 1
            rows = [row for row in list(child) if _local(row.tag) == "tr"]
            for row_no, row in enumerate(rows):
                cells = [cell for cell in list(row) if _local(cell.tag) == "tc"]
                for cell_no, cell in enumerate(cells):
                    if any(_local(desc.tag) == "tbl" for desc in cell.iter() if desc is not cell):
                        raise _safe_error("DOCX contains nested tables")
                    paragraphs = [p for p in list(cell) if _local(p.tag) == "p"]
                    if not paragraphs:
                        if _meaningful_text(cell):
                            raise _safe_error("DOCX table cell has unsupported text structure")
                        paragraphs = [None]
                    for paragraph_no, paragraph in enumerate(paragraphs):
                        text = "" if paragraph is None else _paragraph_text(paragraph)
                        segment_list.append(
                            _segment(
                                text,
                                {
                                    "part": part,
                                    "kind": "table-cell",
                                    "table": table_no,
                                    "row": row_no,
                                    "cell": cell_no,
                                    "paragraph": paragraph_no,
                                },
                            )
                        )
        elif name in {"sectPr", "headerReference", "footerReference"}:
            continue
        elif _meaningful_text(child):
            raise _safe_error("DOCX contains unsupported text outside paragraphs")


def _assign_segment_ids(segments: list[dict]) -> None:
    if len(segments) > MAX_SEGMENTS:
        raise _safe_error("DOCX contains too many text segments")
    for number, segment in enumerate(segments, 1):
        segment["id"] = f"s{number:06d}"


def _inventory(parts: dict[str, bytes], names: list[str], main_part: str, extracted_parts: set[str]) -> list[dict]:
    inventory: list[dict] = []
    for name in names:
        if name == main_part or name in extracted_parts:
            inventory.append({"part": name, "action": "extracted", "reason": "supported transcript text"})
        elif name.startswith(_MEDIA_PREFIX):
            inventory.append({"part": name, "action": "removed", "reason": "source media is not copied"})
        elif name in _KNOWN_REMOVED_PARTS or name.startswith("docProps/"):
            inventory.append({"part": name, "action": "removed", "reason": "source metadata or decoration"})
        elif name.endswith(".rels") or name in {"[Content_Types].xml", "_rels/.rels"}:
            inventory.append({"part": name, "action": "removed", "reason": "source package relationship metadata"})
        elif name in {"word/comments.xml", "word/footnotes.xml", "word/endnotes.xml"}:
            inventory.append({"part": name, "action": "removed", "reason": "empty unsupported annotation part"})
        elif name.endswith("/"):
            inventory.append({"part": name, "action": "removed", "reason": "package directory entry"})
        elif any(name.startswith(prefix) for prefix in _ACTIVE_PREFIXES):
            raise _safe_error("DOCX contains unsupported active content")
        else:
            raise _safe_error("DOCX contains an unsupported package part")
    return inventory


def read_docx(path: Path) -> dict:
    """Read supported transcript text from an untrusted DOCX package."""

    source = Path(path)
    parts, names = _read_package(source)
    _content_types(parts)
    main_part = _document_part(parts)
    main_root = _parse_xml(parts[main_part])
    _assert_supported_tree(main_root)
    if _local(main_root.tag) != "document":
        raise _safe_error("DOCX main document has an invalid root")
    body = next((child for child in list(main_root) if _local(child.tag) == "body"), None)
    if body is None:
        raise _safe_error("DOCX main document has no body")

    # Parse all relationship parts, including those belonging to media and
    # headers, so a missing target or external link cannot hide in the package.
    all_relationships: dict[str, list[dict[str, str]]] = {}
    for name in names:
        if name.endswith(".rels"):
            source_part = None if name == "_rels/.rels" else name[: -len(".rels")]
            if source_part and "/_rels/" in source_part:
                source_part = source_part.replace("/_rels/", "/")
            all_relationships[name] = _relationships(parts, source_part)

    document_relationships = all_relationships.get(_relationship_part(main_part), [])
    segments: list[dict] = []
    table_counter = [0]
    _extract_blocks(body, main_part, kind="paragraph", table_counter=table_counter, segment_list=segments)
    extracted_parts = {main_part}

    # Headers and footers are text-bearing supported surfaces.  They are
    # represented in source order after the body and become body paragraphs in
    # the clean reconstruction, avoiding silent loss of titles or labels.
    referenced_headers = {
        rel["target"]
        for rel in document_relationships
        if rel["type"].endswith("/header") or rel["type"].endswith("/footer")
    }
    for part in sorted(referenced_headers):
        if part not in parts:
            raise _safe_error("DOCX header or footer part is missing")
        root = _parse_xml(parts[part])
        _assert_supported_tree(root)
        _extract_blocks(root, part, kind="header-or-footer", table_counter=table_counter, segment_list=segments)
        extracted_parts.add(part)

    # Annotation parts are supported only when empty.  Non-empty comments or
    # notes are meaningful text with no safe placement in the output.
    for part in ("word/comments.xml", "word/footnotes.xml", "word/endnotes.xml"):
        if part in parts:
            root = _parse_xml(parts[part])
            if _meaningful_text(root):
                raise _safe_error("DOCX contains unsupported annotation text")

    for name, data in parts.items():
        if name.endswith("/") or name in {"[Content_Types].xml", "_rels/.rels"}:
            continue
        if name.endswith(".xml") and name not in extracted_parts:
            root = _parse_xml(data)
            if name.startswith("word/media/"):
                continue
            if name in _KNOWN_REMOVED_PARTS or name.startswith("docProps/"):
                continue
            if name in {"word/comments.xml", "word/footnotes.xml", "word/endnotes.xml"}:
                continue
            if any(name.startswith(prefix) for prefix in _ACTIVE_PREFIXES) or _meaningful_text(root):
                raise _safe_error("DOCX contains unsupported text or active content")

    inventory = _inventory(parts, names, main_part, extracted_parts)
    _assign_segment_ids(segments)
    return {"segments": segments, "inventory": inventory}


def _validate_segments(segments: list[dict]) -> list[dict]:
    if not isinstance(segments, list) or len(segments) > MAX_SEGMENTS:
        raise _safe_error("DOCX segment records are invalid")
    result: list[dict] = []
    ids: set[str] = set()
    total_bytes = 0
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("id"), str):
            raise _safe_error("DOCX segment records are invalid")
        if segment["id"] in ids:
            raise _safe_error("DOCX segment IDs are not unique")
        ids.add(segment["id"])
        text = segment.get("text")
        if not isinstance(text, str):
            raise _safe_error("DOCX segment records are invalid")
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_TEXT_BYTES:
            raise _safe_error("DOCX text segment exceeds the supported size")
        total_bytes += len(encoded)
        if total_bytes > MAX_TOTAL_BYTES:
            raise _safe_error("DOCX text exceeds the supported size")
        locator = segment.get("locator")
        if locator is not None and not isinstance(locator, (dict, str)):
            raise _safe_error("DOCX segment locators are invalid")
        result.append({"id": segment["id"], "text": text, "locator": locator})
    return result


def _text_run(parent, text: str) -> None:
    run = OutputET.SubElement(parent, f"{{{W_NS}}}r")
    chunks = re.split(r"([\t\n])", text)
    for chunk in chunks:
        if chunk == "\t":
            OutputET.SubElement(run, f"{{{W_NS}}}tab")
        elif chunk == "\n":
            OutputET.SubElement(run, f"{{{W_NS}}}br")
        elif chunk:
            text_element = OutputET.SubElement(run, f"{{{W_NS}}}t")
            if chunk[:1].isspace() or chunk[-1:].isspace():
                text_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            text_element.text = chunk
    if not chunks or (len(chunks) == 1 and not chunks[0]):
        OutputET.SubElement(run, f"{{{W_NS}}}t").text = ""


def _paragraph(parent, text: str):
    paragraph = OutputET.SubElement(parent, f"{{{W_NS}}}p")
    _text_run(paragraph, text)
    return paragraph


def _document_tree(segments: list[dict]):
    root = OutputET.Element(f"{{{W_NS}}}document", {"{http://www.w3.org/XML/1998/namespace}space": "preserve"})
    body = OutputET.SubElement(root, f"{{{W_NS}}}body")
    current_key = None
    current_tables: dict[int, dict[int, dict[int, list[str]]]] = {}

    def flush_table() -> None:
        nonlocal current_key, current_tables
        if current_key is None:
            return
        table = OutputET.SubElement(body, f"{{{W_NS}}}tbl")
        rows = current_tables[current_key]
        for row_no in range(max(rows, default=-1) + 1):
            row = OutputET.SubElement(table, f"{{{W_NS}}}tr")
            cells = rows.get(row_no, {})
            for cell_no in range(max(cells, default=-1) + 1):
                cell = OutputET.SubElement(row, f"{{{W_NS}}}tc")
                values = cells.get(cell_no, [""])
                for value in values:
                    _paragraph(cell, value)
        current_key = None
        current_tables = {}

    for segment in segments:
        locator = segment.get("locator") or {}
        if not isinstance(locator, dict):
            locator = {}
        is_cell = locator.get("kind") == "table-cell" and isinstance(locator.get("table"), int)
        if is_cell:
            key = locator["table"]
            if current_key != key:
                flush_table()
                current_key = key
                current_tables[key] = {}
            rows = current_tables[key]
            row_no = locator.get("row")
            cell_no = locator.get("cell")
            if not isinstance(row_no, int) or row_no < 0 or not isinstance(cell_no, int) or cell_no < 0:
                raise _safe_error("DOCX table locators are invalid")
            rows.setdefault(row_no, {}).setdefault(cell_no, []).append(segment["text"])
        else:
            flush_table()
            _paragraph(body, segment["text"])
    flush_table()
    sect = OutputET.SubElement(body, f"{{{W_NS}}}sectPr")
    OutputET.SubElement(sect, f"{{{W_NS}}}pgSz", {f"{{{W_NS}}}w": "12240", f"{{{W_NS}}}h": "15840"})
    OutputET.SubElement(
        sect,
        f"{{{W_NS}}}pgMar",
        {f"{{{W_NS}}}top": "1440", f"{{{W_NS}}}right": "1440", f"{{{W_NS}}}bottom": "1440", f"{{{W_NS}}}left": "1440"},
    )
    return root


def _package_bytes(segments: list[dict]) -> bytes:
    root = _document_tree(segments)
    OutputET.register_namespace("w", W_NS)
    document = OutputET.tostring(root, encoding="utf-8", xml_declaration=True)
    content_types = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Types xmlns="{CT_NS}">'
        f'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        f'<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        f"</Types>"
    ).encode()
    root_rels = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{REL_NS}">'
        f'<Relationship Id="rId1" Type="{OFFICE_DOCUMENT_REL}" Target="word/document.xml"/>'
        f"</Relationships>"
    ).encode()
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def write_docx(segments: list[dict], path: Path) -> None:
    """Write a clean reconstructed DOCX atomically."""

    clean_segments = _validate_segments(segments)
    destination = Path(path)
    if not destination.parent.exists() or not destination.parent.is_dir():
        raise _safe_error("DOCX output directory does not exist")
    data = _package_bytes(clean_segments)
    temporary_name: str | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        validate_output(destination)
    except (OSError, AnonymizerError) as exc:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        if isinstance(exc, AnonymizerError):
            raise
        raise _safe_error("DOCX output could not be written") from exc


def validate_output(path: Path) -> None:
    """Validate that *path* is exactly a package produced by ``write_docx``."""

    parts, names = _read_package(Path(path))
    expected = {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}
    if set(names) != expected or any(name.endswith("/") for name in names):
        raise _safe_error("DOCX output contains unapproved package parts")
    defaults, overrides = _content_types(parts)
    if overrides.get("word/document.xml") != "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml":
        raise _safe_error("DOCX output has invalid content types")
    if set(defaults) != {"rels", "xml"} or set(overrides) != {"word/document.xml"}:
        raise _safe_error("DOCX output has invalid content types")
    relationships = _relationships(parts, None)
    if len(relationships) != 1 or relationships[0]["target"] != "word/document.xml" or relationships[0]["type"] != OFFICE_DOCUMENT_REL:
        raise _safe_error("DOCX output has invalid relationships")
    root = _parse_xml(parts["word/document.xml"])
    if _local(root.tag) != "document":
        raise _safe_error("DOCX output has an invalid document root")
    _assert_supported_tree(root)
    body = next((child for child in list(root) if _local(child.tag) == "body"), None)
    if body is None or not any(_local(child.tag) == "sectPr" for child in list(body)):
        raise _safe_error("DOCX output has an invalid document body")
