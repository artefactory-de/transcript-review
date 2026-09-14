from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest

from transcript_anonymizer.documents import (
    _validate_member_name,
    read_docx,
    validate_output,
    write_docx,
)
from transcript_anonymizer.errors import AnonymizerError

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
OFFICE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"


def _package(parts: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _source_bytes(document: str, *, extra: dict[str, bytes] | None = None) -> bytes:
    parts = {
        "[Content_Types].xml": f'''<?xml version="1.0"?>
          <Types xmlns="{CT}">
            <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
            <Default Extension="xml" ContentType="application/xml"/>
            <Default Extension="png" ContentType="image/png"/>
            <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
          </Types>'''.encode(),
        "_rels/.rels": f'''<Relationships xmlns="{R}">
          <Relationship Id="rId1" Type="{OFFICE}" Target="word/document.xml"/>
        </Relationships>'''.encode(),
        "word/document.xml": document.encode(),
        "word/_rels/document.xml.rels": f'''<Relationships xmlns="{R}">
          <Relationship Id="rIdMedia" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
          <Relationship Id="rIdComments" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="comments.xml"/>
        </Relationships>'''.encode(),
        "word/media/image1.png": b"hidden source media",
        "word/media/unreferenced.png": b"more hidden source media",
        "word/comments.xml": f'<comments xmlns="{W}"/>'.encode(),
        "docProps/core.xml": b"<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\"><dc:title xmlns:dc=\"http://purl.org/dc/elements/1.1/\">source title</dc:title></cp:coreProperties>",
    }
    if extra:
        parts.update(extra)
    return _package(parts)


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def test_read_preserves_run_text_order_and_tables(tmp_path: Path) -> None:
    document = f'''<w:document xmlns:w="{W}">
      <w:body>
        <w:p><w:r><w:t>Speaker: </w:t></w:r><w:r><w:t>Anna</w:t></w:r></w:p>
        <w:p><w:r><w:t>00:01</w:t><w:tab/><w:t>starts</w:t><w:br/><w:t>now</w:t></w:r></w:p>
        <w:tbl>
          <w:tr><w:tc><w:p><w:r><w:t>left</w:t></w:r></w:p></w:tc>
              <w:tc><w:p><w:r><w:t>right</w:t></w:r></w:p></w:tc></w:tr>
        </w:tbl>
        <w:p/>
        <w:sectPr/>
      </w:body>
    </w:document>'''
    source = tmp_path / "source.docx"
    _write(source, _source_bytes(document))

    result = read_docx(source)

    assert [segment["text"] for segment in result["segments"]] == [
        "Speaker: Anna",
        "00:01\tstarts\nnow",
        "left",
        "right",
        "",
    ]
    assert [segment["id"] for segment in result["segments"]] == [
        "s000001",
        "s000002",
        "s000003",
        "s000004",
        "s000005",
    ]
    assert any(row["part"] == "word/media/unreferenced.png" and row["action"] == "removed" for row in result["inventory"])
    assert any(row["part"] == "word/comments.xml" and row["action"] == "removed" for row in result["inventory"])


def test_write_is_clean_and_round_trips_tables(tmp_path: Path) -> None:
    source = tmp_path / "source.docx"
    _write(
        source,
        _source_bytes(
            f'''<w:document xmlns:w="{W}"><w:body>
              <w:p><w:r><w:t>one</w:t></w:r></w:p>
              <w:tbl><w:tr><w:tc><w:p><w:r><w:t>cell 1</w:t></w:r></w:p></w:tc>
                <w:tc><w:p><w:r><w:t>cell 2</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
              <w:p><w:r><w:t>three</w:t></w:r></w:p><w:sectPr/></w:body></w:document>''',
        ),
    )
    extracted = read_docx(source)
    output = tmp_path / "candidate.docx"
    write_docx(extracted["segments"], output)

    validate_output(output)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}
        xml = archive.read("word/document.xml")
        assert b"cell 1" in xml and b"cell 2" in xml
        assert b"image1.png" not in xml
    assert [part["text"] for part in read_docx(output)["segments"]] == ["one", "cell 1", "cell 2", "three"]


@pytest.mark.parametrize(
    "member",
    ["../outside.xml", "/absolute.xml", "word\\unsafe.xml"],
)
def test_rejects_unsafe_zip_paths(tmp_path: Path, member: str) -> None:
    # ZipFile normalizes backslashes while creating an archive on Windows, so
    # exercise the same validator directly for this platform-specific case.
    if os.name == "nt" and "\\" in member:
        with pytest.raises(AnonymizerError, match="unsafe package path"):
            _validate_member_name(member)
        return
    path = tmp_path / "unsafe.docx"
    _write(path, _package({member: b"x"}))
    with pytest.raises(AnonymizerError, match="unsafe package path"):
        read_docx(path)


@pytest.mark.parametrize(
    "fragment",
    [
        '<w:p><w:ins><w:r><w:t>tracked</w:t></w:r></w:ins></w:p>',
        '<w:p><w:fldSimple w:instr="PAGE"><w:r><w:t>field</w:t></w:r></w:fldSimple></w:p>',
        '<w:p><w:sdt><w:sdtContent><w:p><w:r><w:t>controlled</w:t></w:r></w:p></w:sdtContent></w:sdt></w:p>',
    ],
)
def test_rejects_unsupported_text_structures(tmp_path: Path, fragment: str) -> None:
    source = tmp_path / "unsupported.docx"
    _write(source, _source_bytes(f'<w:document xmlns:w="{W}"><w:body>{fragment}<w:sectPr/></w:body></w:document>'))
    with pytest.raises(AnonymizerError, match="unsupported"):
        read_docx(source)


def test_rejects_nonempty_comments_and_external_relationships(tmp_path: Path) -> None:
    comments_source = tmp_path / "comments.docx"
    _write(
        comments_source,
        _source_bytes(
            f'<w:document xmlns:w="{W}"><w:body><w:p/><w:sectPr/></w:body></w:document>',
            extra={"word/comments.xml": f'<w:comments xmlns:w="{W}"><w:comment><w:p><w:r><w:t>note</w:t></w:r></w:p></w:comment></w:comments>'.encode()},
        ),
    )
    with pytest.raises(AnonymizerError, match="annotation"):
        read_docx(comments_source)

    external_source = tmp_path / "external.docx"
    data = _source_bytes(f'<w:document xmlns:w="{W}"><w:body><w:p/><w:sectPr/></w:body></w:document>')
    buffer = io.BytesIO(data)
    with zipfile.ZipFile(buffer) as old:
        parts = {name: old.read(name) for name in old.namelist()}
    parts["word/_rels/document.xml.rels"] = f'''<Relationships xmlns="{R}">
      <Relationship Id="rIdExternal" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.invalid" TargetMode="External"/>
    </Relationships>'''.encode()
    _write(external_source, _package(parts))
    with pytest.raises(AnonymizerError, match="external relationship"):
        read_docx(external_source)
