"""Fetch and verify the pinned local GLiNER2 checkpoint at build time.

This script intentionally uses only the Python standard library. It streams each
asset to an ignored output directory, verifies the immutable manifest, and never
imports an ML package or executes code from the downloaded checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

CHUNK_SIZE = 1024 * 1024
ALLOWED_HOST = "huggingface.co"


def read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not read model asset manifest") from exc
    if isinstance(manifest, dict) and set(manifest) == {"canonical_resource"}:
        target = (path.parent / str(manifest["canonical_resource"])).resolve()
        if target == path.resolve():
            raise RuntimeError("Model asset manifest pointer loops")
        return read_manifest(target)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported model asset manifest")
    repository = manifest.get("repository")
    revision = manifest.get("revision")
    files = manifest.get("files")
    if not isinstance(repository, str) or not repository or not isinstance(revision, str) or len(revision) != 40:
        raise RuntimeError("Manifest repository/revision is invalid")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Manifest has no model files")
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise TypeError("Manifest contains an invalid file entry")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or len(item.get("sha256", "")) != 64:
            raise RuntimeError("Manifest contains an unsafe or unchecksummed path")
        if not isinstance(item.get("size"), int) or item["size"] < 0:
            raise RuntimeError("Manifest contains an invalid file size")
    return manifest


def _target_path(root: Path, relative: str) -> Path:
    target = root / Path(relative)
    # Resolve only for containment validation; do not create or delete anything
    # outside the requested output directory.
    if root.resolve() not in target.resolve().parents:
        raise RuntimeError("Manifest path escapes model output directory")
    return target


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while block := stream.read(CHUNK_SIZE):
                size += len(block)
                digest.update(block)
    except OSError as exc:
        raise RuntimeError("Could not read a model asset for verification") from exc
    return size == expected_size and digest.hexdigest() == expected_sha256


def fetch_file(url: str, destination: Path, expected_size: int, expected_sha256: str) -> None:
    if destination.exists() and verify_file(destination, expected_size, expected_sha256):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        request = Request(url, headers={"User-Agent": "transcript-anonymizer-build/1"})
        with urlopen(request, timeout=120) as response, partial.open("wb") as stream:
            digest = hashlib.sha256()
            size = 0
            while block := response.read(CHUNK_SIZE):
                size += len(block)
                digest.update(block)
                stream.write(block)
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise RuntimeError("Downloaded model asset failed manifest verification")
        os.replace(partial, destination)
    except Exception:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise


def fetch(manifest: dict, output: Path) -> None:
    repository = manifest["repository"]
    revision = manifest["revision"]
    for item in manifest["files"]:
        relative = item["path"]
        destination = _target_path(output, relative)
        encoded = quote(relative, safe="/")
        url = f"https://{ALLOWED_HOST}/{repository}/resolve/{revision}/{encoded}?download=true"
        print(f"verifying {relative}", file=sys.stderr)
        fetch_file(url, destination, item["size"], item["sha256"])
    for item in manifest["files"]:
        if not verify_file(_target_path(output, item["path"]), item["size"], item["sha256"]):
            raise RuntimeError("Model directory failed final manifest verification")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("src/transcript_anonymizer/model-assets.json"))
    parser.add_argument("--output", type=Path, default=Path("models/gliner2-privacy-filter-PII-multi"))
    args = parser.parse_args()
    try:
        fetch(read_manifest(args.manifest), args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"model fetch failed: {exc}", file=sys.stderr)
        return 2
    print(f"verified model assets in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
