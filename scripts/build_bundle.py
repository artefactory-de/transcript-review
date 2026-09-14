"""Build the Windows PyInstaller bundles.

The script is intentionally a build orchestrator, not an application launcher.
It never imports the detector, loads torch, downloads model files, or publishes
artifacts.  A production build must run on Windows.  ``--allow-host-trial`` is
available for a local PyInstaller smoke build on another host and marks its
manifest as ineligible for Windows release acceptance.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path


class BuildError(RuntimeError):
    """An actionable local build failure."""


MODEL_MANIFEST_RELATIVE = Path("src/transcript_anonymizer/model-assets.json")
MODEL_MANIFEST_RESOURCE = "transcript_anonymizer/model-assets.json"
# Avoid importing optional ML backends during analysis. Collect package data
# and let normal analysis follow the runtime imports.
COLLECT_MODULES: tuple[str, ...] = ()
COLLECT_DATA_MODULES = ("gliner2", "transformers", "tokenizers", "safetensors")
HIDDEN_IMPORTS = ("gliner2.auto",)
_SYNTHETIC_SAMPLE = (
    "00:00:01 Anna Beispiel: Kontakt anna@example.invalid.",
    "00:00:05 Max Muster: Ab 100000 Euro erfolgt eine zweite Freigabe.",
)
ENTRYPOINT_MODULES = {
    "cli": ("transcript_anonymizer.cli", False),
    "desktop": ("transcript_anonymizer.desktop", True),
}

# These names are deliberately exact.  In particular, do not discover or copy
# DLLs from site-packages: NumPy and other wheels may carry private, patched
# copies whose names and import tables are not interchangeable with the
# Microsoft redistributable.
RUNTIME_REQUIRED_DLLS = (
    "MSVCP140.dll",
    "VCRUNTIME140.dll",
    "VCRUNTIME140_1.dll",
)
RUNTIME_OPTIONAL_DLLS = frozenset(
    {
        "concrt140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll",
        "vcomp140.dll",
    }
)
RUNTIME_PROVENANCE_FILES = frozenset(
    {
        "license",
        "license.txt",
        "notice",
        "notice.txt",
        "third_party_notices.txt",
        "runtime-provenance.json",
        "runtime-manifest.json",
    }
)
RUNTIME_MAX_FILE_BYTES = 64 * 1024 * 1024
_RUNTIME_PE_HEADER_LIMIT = 1024 * 1024


def _running_under_wine() -> bool:
    """Return whether this Windows Python process is hosted by Wine.

    Wine exposes ``wine_get_version`` from ntdll.  Keeping this probe local to
    the manifest path lets the same script run on Linux, native Windows, and a
    Wine PE-build trial without treating the latter as native acceptance.
    """
    if platform.system() != "Windows":
        return False
    try:
        return bool(ctypes.windll.ntdll.wine_get_version)
    except (AttributeError, OSError):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_runtime_pe(path: Path) -> None:
    """Require a bounded, native x64 PE DLL without loading it.

    This is intentionally only a file-format check.  A PE can be built or
    signed in many environments, so native Windows acceptance still requires
    running the resulting bundle on the supported Windows host.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise BuildError("A selected runtime DLL could not be inspected.") from exc
    if size <= 0 or size > RUNTIME_MAX_FILE_BYTES:
        raise BuildError("A selected runtime DLL has an unsafe file size.")
    try:
        with path.open("rb") as stream:
            dos = stream.read(64)
            if len(dos) < 64 or dos[:2] != b"MZ":
                raise BuildError("A selected runtime DLL is not a PE file.")
            pe_offset = struct.unpack_from("<I", dos, 0x3C)[0]
            if pe_offset > _RUNTIME_PE_HEADER_LIMIT:
                raise BuildError("A selected runtime DLL has an unsafe PE header offset.")
            stream.seek(pe_offset)
            header = stream.read(26)
    except OSError as exc:
        raise BuildError("A selected runtime DLL could not be read.") from exc
    if len(header) < 26 or header[:4] != b"PE\0\0":
        raise BuildError("A selected runtime DLL is not a PE file.")
    machine = struct.unpack_from("<H", header, 4)[0]
    optional_magic = struct.unpack_from("<H", header, 24)[0]
    if machine != 0x8664 or optional_magic != 0x20B:
        raise BuildError("A selected runtime DLL is not a native x64 PE file.")


def _runtime_inventory(runtime_dir: Path) -> list[dict[str, object]]:
    """Validate and hash the explicit app-local VC runtime input directory."""
    runtime_dir = Path(runtime_dir)
    if runtime_dir.is_symlink() or not runtime_dir.is_dir():
        raise BuildError("The runtime directory does not exist or is not a directory.")
    try:
        children = sorted(runtime_dir.iterdir(), key=lambda item: item.name.casefold())
    except OSError as exc:
        raise BuildError("The runtime directory could not be read.") from exc
    allowed_dlls = {name.casefold() for name in RUNTIME_REQUIRED_DLLS} | {
        name.casefold() for name in RUNTIME_OPTIONAL_DLLS
    }
    allowed_files = allowed_dlls | RUNTIME_PROVENANCE_FILES
    by_name: dict[str, Path] = {}
    for child in children:
        key = child.name.casefold()
        if key in by_name:
            raise BuildError("The runtime directory contains duplicate case-insensitive names.")
        if child.is_symlink():
            raise BuildError("Runtime files must not be symbolic links.")
        if child.is_dir():
            raise BuildError("The runtime directory must contain only allowlisted files.")
        if not child.is_file() or key not in allowed_files:
            raise BuildError("The runtime directory contains a file outside its allowlist.")
        by_name[key] = child

    missing = [name for name in RUNTIME_REQUIRED_DLLS if name.casefold() not in by_name]
    if missing:
        raise BuildError("The runtime directory is missing required Microsoft runtime DLLs.")
    records: list[dict[str, object]] = []
    for key, child in sorted(by_name.items()):
        if key in allowed_dlls:
            _validate_runtime_pe(child)
            kind = "dll"
        else:
            try:
                if child.stat().st_size > RUNTIME_MAX_FILE_BYTES:
                    raise BuildError("A selected runtime provenance file has an unsafe file size.")
            except OSError as exc:
                raise BuildError("A selected runtime provenance file could not be inspected.") from exc
            kind = "provenance"
        records.append(
            {
                "path": child.name,
                "bytes": child.stat().st_size,
                "sha256": _sha256(child),
                "kind": kind,
            }
        )
    return records


def _runtime_pyinstaller_args(
    runtime_dir: Path, records: list[dict[str, object]] | None = None
) -> list[str]:
    """Return explicit PyInstaller flags for validated runtime files."""
    runtime_dir = Path(runtime_dir)
    current_records = _runtime_inventory(runtime_dir)
    if records is not None and records != current_records:
        raise BuildError("The runtime input changed after it was inventoried.")
    records = current_records
    args: list[str] = []
    for record in records:
        source = runtime_dir / str(record["path"])
        if record["kind"] == "dll":
            args.extend(["--add-binary", f"{source}{os.pathsep}."])
        else:
            args.extend(["--add-data", f"{source}{os.pathsep}."])
    return args


def _stage_runtime_assets(
    runtime_dir: Path,
    artifact_dir: Path,
    records: list[dict[str, object]] | None = None,
    *,
    replace_existing: bool = False,
) -> list[dict[str, object]]:
    """Copy validated runtime files into an existing onedir artifact root.

    The helper is also usable when an artifact was frozen before this runtime
    input was introduced. Existing files are never silently replaced unless
    the explicit ``replace_existing`` option is set.
    """
    artifact_dir = Path(artifact_dir)
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise BuildError("The runtime staging destination is not a directory.")
    runtime_dir = Path(runtime_dir).resolve()
    artifact_dir = artifact_dir.resolve()
    if runtime_dir == artifact_dir:
        raise BuildError("The runtime staging source and destination must differ.")
    current_records = _runtime_inventory(runtime_dir)
    if records is not None and records != current_records:
        raise BuildError("The runtime input changed after it was inventoried.")
    records = current_records
    for record in records:
        source = runtime_dir / str(record["path"])
        destination = artifact_dir / str(record["path"])
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise BuildError("The artifact contains an unsafe runtime destination.")
        matches = destination.exists() and _sha256(destination) == record["sha256"]
        if destination.exists() and not matches and not replace_existing:
            raise BuildError("The artifact already contains a different runtime file.")
        if not destination.exists() or not matches:
            try:
                shutil.copy2(source, destination)
            except OSError as exc:
                raise BuildError("Could not stage the selected runtime files.") from exc
    return records


def _verify_runtime_artifact(
    artifact: Path, runtime_files: list[dict[str, object]]
) -> None:
    """Verify that onedir staging retained each selected runtime file.

    A onefile executable is opaque without running the target platform's
    extractor, so its pre-build validated inputs are the available check.
    """
    if artifact.is_file():
        return
    if not artifact.is_dir():
        raise BuildError("The built artifact is not a file or directory.")
    # PyInstaller 6 places ``--add-binary SOURCE;.`` payloads in the onedir
    # bundle's ``_internal`` directory. Older layouts put them beside the EXE;
    # retain that fallback for already-frozen artifacts.
    payload_root = artifact / "_internal" if (artifact / "_internal").is_dir() else artifact
    files = {path.name.casefold(): path for path in payload_root.iterdir() if path.is_file()}
    for record in runtime_files:
        name = str(record["path"])
        path = files.get(name.casefold())
        if path is None or path.is_symlink() or _sha256(path) != record["sha256"]:
            raise BuildError("The built artifact is missing a selected runtime file.")


def _model_inventory(model_dir: Path) -> list[dict[str, object]]:
    if not model_dir.is_dir():
        raise BuildError("The model directory does not exist or is not a directory.")
    records: list[dict[str, object]] = []
    root = model_dir.resolve()
    for path in sorted(model_dir.rglob("*")):
        if path.is_symlink():
            raise BuildError("Model assets must not contain symbolic links.")
        if not path.is_file():
            continue
        resolved = path.resolve()
        if root not in resolved.parents:
            raise BuildError("Model asset resolution escaped the selected directory.")
        records.append(
            {
                "path": path.relative_to(model_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not records:
        raise BuildError("The model directory contains no files.")
    return records


def _read_model_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError("The canonical model-assets.json package resource is invalid.") from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise BuildError("The canonical model-assets.json package resource is unsupported.")
    files = payload.get("files")
    if not isinstance(files, list):
        raise BuildError("The canonical model-assets.json package resource has no file list.")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise BuildError("The canonical model-assets.json package resource has an invalid file.")
        relative = Path(item["path"])
        digest = item.get("sha256")
        size = item.get("size")
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.as_posix()
            or relative.as_posix() in seen
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise BuildError("The canonical model-assets.json package resource has an invalid file.")
        seen.add(relative.as_posix())
    return payload


def _model_manifest(project_root: Path) -> tuple[Path, str]:
    path = project_root / MODEL_MANIFEST_RELATIVE
    if not path.is_file():
        raise BuildError("The canonical model-assets.json package resource is missing.")
    _read_model_manifest(path)
    return path, _sha256(path)


def _write_entrypoint(path: Path, entrypoint: str = "cli") -> None:
    # Running the package module through a small generated script keeps the
    # package-relative imports in cli.py intact for PyInstaller analysis.
    try:
        module, _ = ENTRYPOINT_MODULES[entrypoint]
    except KeyError as exc:
        raise BuildError(f"Unknown application entrypoint: {entrypoint}") from exc
    path.write_text(
        f"from {module} import main\nraise SystemExit(main())\n",
        encoding="utf-8",
    )


def _write_synthetic_sample(path: Path) -> None:
    """Create the same safe demonstration DOCX as the CLI sample command."""
    from docx import Document

    document = Document()
    for text in _SYNTHETIC_SAMPLE:
        document.add_paragraph(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)


def _validate_model_inventory(records: list[dict], manifest_path: Path) -> None:
    expected = _read_model_manifest(manifest_path)["files"]
    actual_by_path = {item["path"]: item for item in records}
    if set(actual_by_path) != {item["path"] for item in expected}:
        raise BuildError("Model directory does not match the required asset manifest.")
    for item in expected:
        actual = actual_by_path[item["path"]]
        if actual["bytes"] != item["size"] or actual["sha256"] != item["sha256"]:
            raise BuildError("Model asset checksum mismatch; fetch the pinned snapshot again.")


def _run_pyinstaller(
    *,
    project_root: Path,
    entrypoint_path: Path,
    output_dir: Path,
    mode: str,
    model_dir: Path | None,
    model_manifest: Path,
    entrypoint: str = "cli",
    runtime_dir: Path | None = None,
    runtime_files: list[dict[str, object]] | None = None,
) -> Path:
    try:
        _, windowed = ENTRYPOINT_MODULES[entrypoint]
    except KeyError as exc:
        raise BuildError(f"Unknown application entrypoint: {entrypoint}") from exc
    suffix = "" if entrypoint == "cli" else f"-{entrypoint}"
    name = f"transcript-anonymizer{suffix}-{mode}"
    bundle_root = output_dir if entrypoint == "cli" else output_dir / entrypoint
    dist_dir = bundle_root / mode
    work_dir = output_dir / ".pyinstaller" / entrypoint / mode
    spec_dir = output_dir / ".spec"
    dist_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        name,
        "--paths",
        str(project_root / "src"),
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        "--windowed" if windowed else "--console",
        "--onedir" if mode == "onedir" else "--onefile",
    ]
    if model_dir is not None:
        # PyInstaller uses the host separator: ';' on Windows, ':' on POSIX.
        command.extend(["--add-data", f"{model_dir}{os.pathsep}model"])
    else:
        for package in ("gliner2", "torch", "transformers", "peft", "tokenizers", "safetensors"):
            command.extend(["--exclude-module", package])
    command.extend(
        ["--add-data", f"{model_manifest}{os.pathsep}{MODEL_MANIFEST_RESOURCE.rsplit('/', 1)[0]}"]
    )
    # AutoExtractor discovers registries and model backends dynamically. These
    # collection flags are build-time analysis instructions; they do not import
    # or execute the model in this script.
    if model_dir is not None:
        for module in COLLECT_MODULES:
            command.extend(["--collect-all", module])
        for module in COLLECT_DATA_MODULES:
            command.extend(["--collect-data", module])
        for module in HIDDEN_IMPORTS:
            command.extend(["--hidden-import", module])
    if runtime_dir is not None:
        command.extend(_runtime_pyinstaller_args(runtime_dir, runtime_files))
    command.append(str(entrypoint_path))
    try:
        subprocess.run(command, cwd=project_root, check=True)
    except FileNotFoundError as exc:
        raise BuildError("PyInstaller is not installed in this Python environment.") from exc
    except subprocess.CalledProcessError as exc:
        raise BuildError(f"PyInstaller failed while building the {mode} bundle.") from exc
    if mode == "onedir":
        artifact = dist_dir / name
    else:
        suffix = ".exe" if platform.system() == "Windows" else ""
        artifact = dist_dir / f"{name}{suffix}"
    if not artifact.exists():
        raise BuildError(f"PyInstaller did not create the expected {mode} artifact.")
    return artifact


def _artifact_files(root: Path) -> list[dict[str, object]]:
    paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    records: list[dict[str, object]] = []
    for path in paths:
        records.append(
            {
                "path": path.relative_to(root.parent if root.is_file() else root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _write_delivery_zip(
    output_dir: Path,
    bundles: list[tuple[Path, Path]],
    zip_path: Path,
    support_files: tuple[tuple[Path, str], ...] = (),
) -> Path:
    """Package selected bundles, manifests, and explicit support files.

    Support files are passed with their archive names so the caller has to
    opt into each delivery document. This prevents an output directory or a
    repository tree from becoming an accidental ZIP allowlist.
    """

    if not bundles:
        raise BuildError("No bundles were built; cannot create a delivery ZIP.")
    root = output_dir.resolve()
    zip_path = zip_path.resolve()
    if zip_path.exists():
        raise BuildError("The delivery ZIP already exists; choose a new output directory.")
    members: dict[str, Path] = {}
    for artifact, manifest in bundles:
        candidates = [artifact] if artifact.is_file() else sorted(artifact.rglob("*"))
        candidates.append(manifest)
        for path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            resolved = path.resolve()
            if root not in resolved.parents:
                raise BuildError("A bundle file escaped the selected output directory.")
            relative = path.relative_to(root).as_posix()
            if relative in members and members[relative] != path:
                raise BuildError("Bundle files have colliding ZIP paths.")
            members[relative] = path
    for source, archive_name in support_files:
        source = source.resolve()
        archive_relative = Path(archive_name)
        if (
            not source.is_file()
            or source.is_symlink()
            or archive_relative.is_absolute()
            or ".." in archive_relative.parts
            or not archive_relative.as_posix()
        ):
            raise BuildError("A selected support file is invalid.")
        relative = archive_relative.as_posix()
        if relative in members and members[relative] != source:
            raise BuildError("Bundle files have colliding ZIP paths.")
        members[relative] = source
    if not members:
        raise BuildError("The selected bundles contain no files for the delivery ZIP.")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative, path in sorted(members.items()):
                archive.write(path, relative)
    except OSError as exc:
        raise BuildError("Could not create the delivery ZIP.") from exc
    return zip_path


def _write_manifest(
    *,
    artifact: Path,
    output_dir: Path,
    mode: str,
    model_files: list[dict[str, object]],
    model_manifest_sha256: str,
    rules_only: bool,
    host_trial: bool,
    entrypoint: str = "cli",
    runtime_dir: Path | None = None,
    runtime_files: list[dict[str, object]] | None = None,
) -> Path:
    if runtime_dir is not None:
        current_runtime_files = _runtime_inventory(runtime_dir)
        if runtime_files is not None and runtime_files != current_runtime_files:
            raise BuildError("The runtime input changed before the manifest was written.")
        runtime_files = current_runtime_files
    under_wine = _running_under_wine()
    manifest = {
        "schema_version": 1,
        "application": "transcript-anonymizer",
        "mode": mode,
        "entrypoint": entrypoint,
        "host": platform.platform(),
        "host_trial": host_trial or under_wine,
        "under_wine": under_wine,
        "native_windows_build": platform.system() == "Windows" and not under_wine,
        "native_windows_acceptance": False,
        "windows_artifact_candidate": (
            not host_trial and not under_wine and platform.system() == "Windows"
        ),
        "release_eligible": False,
        "acceptance_status": "unverified",
        "rules_only": rules_only,
        "model": {
            "included": bool(model_files),
            "asset_count": len(model_files),
            "asset_bytes": sum(int(item["bytes"]) for item in model_files),
            "destination": "model",
            "manifest_resource": MODEL_MANIFEST_RESOURCE,
            "manifest_sha256": model_manifest_sha256,
            "runtime_policy": "local-only; no runtime download",
        },
        "runtime": {
            "included": runtime_dir is not None,
            "required": list(RUNTIME_REQUIRED_DLLS),
            "files": runtime_files or [],
            "provenance": "explicit local Microsoft VC runtime allowlist",
        },
        "artifacts": _artifact_files(artifact),
        "manifest_scope": "This manifest does not include its own bytes.",
    }
    manifest_path = artifact.parent / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build local PyInstaller bundles; Windows release builds require Windows."
    )
    parser.add_argument(
        "--model", type=Path, help="Local model directory to bundle under sys._MEIPASS/model"
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help=(
            "Explicit flat directory containing the canonical x64 Microsoft VC runtime "
            "DLLs and optional allowlisted notices"
        ),
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Build a limited rules-only trial without model assets",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("build/windows"), help="Output directory"
    )
    parser.add_argument("--mode", choices=("both", "onedir", "onefile"), default="both")
    parser.add_argument(
        "--entrypoint",
        choices=("cli", "desktop", "both"),
        default="cli",
        help="Build the CLI, the windowed desktop entrypoint, or both",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a ZIP containing the selected bundles and manifests",
    )
    parser.add_argument(
        "--allow-host-trial",
        action="store_true",
        help="Permit a non-Windows PyInstaller trial; its manifest is not release eligible",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if sys.version_info[:2] != (3, 12):
        print("Python 3.12 is required for the locked PyInstaller build.", file=sys.stderr)
        return 2
    if platform.system() != "Windows" and not args.allow_host_trial:
        print(
            "Refusing a Windows release build on this host. Run on Windows, or use "
            "--allow-host-trial for a non-release local PyInstaller trial.",
            file=sys.stderr,
        )
        return 2
    if not args.rules_only and args.model is None:
        print(
            "A local --model directory is required unless --rules-only is explicit.",
            file=sys.stderr,
        )
        return 2
    if platform.system() == "Windows" and args.model is not None and args.runtime_dir is None:
        print(
            "A Windows model-enabled build requires --runtime-dir with the canonical "
            "Microsoft VC runtime DLLs.",
            file=sys.stderr,
        )
        return 2
    if args.entrypoint in {"desktop", "both"}:
        desktop_source = project_root / "src" / "transcript_anonymizer" / "desktop.py"
        if not desktop_source.is_file():
            print("The desktop entrypoint is unavailable in this source tree.", file=sys.stderr)
            return 2
    try:
        model_files = _model_inventory(args.model) if args.model is not None else []
        runtime_files = (
            _runtime_inventory(args.runtime_dir) if args.runtime_dir is not None else []
        )
        model_manifest, model_manifest_sha256 = _model_manifest(project_root)
        if model_files:
            _validate_model_inventory(model_files, model_manifest)
        output_dir = args.output.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        entrypoints = ("cli", "desktop") if args.entrypoint == "both" else (args.entrypoint,)
        modes = ("onedir", "onefile") if args.mode == "both" else (args.mode,)
        built: list[tuple[Path, Path]] = []
        for entrypoint_kind in entrypoints:
            entrypoint = output_dir / f".entrypoint-{entrypoint_kind}.py"
            _write_entrypoint(entrypoint, entrypoint_kind)
            try:
                for mode in modes:
                    artifact = _run_pyinstaller(
                        project_root=project_root,
                        entrypoint_path=entrypoint,
                        output_dir=output_dir,
                        mode=mode,
                        model_dir=args.model.resolve() if args.model is not None else None,
                        model_manifest=model_manifest,
                        entrypoint=entrypoint_kind,
                        runtime_dir=args.runtime_dir.resolve() if args.runtime_dir is not None else None,
                        runtime_files=runtime_files,
                    )
                    if runtime_files and mode == "onedir":
                        _verify_runtime_artifact(artifact, runtime_files)
                    manifest = _write_manifest(
                        artifact=artifact,
                        output_dir=output_dir,
                        mode=mode,
                        model_files=model_files,
                        model_manifest_sha256=model_manifest_sha256,
                        rules_only=args.rules_only,
                        host_trial=platform.system() != "Windows",
                        entrypoint=entrypoint_kind,
                        runtime_dir=args.runtime_dir,
                        runtime_files=runtime_files,
                    )
                    built.append((artifact, manifest))
                    print(f"Built {entrypoint_kind} {mode} bundle: {artifact}")
                    print(f"Manifest: {manifest}")
            finally:
                entrypoint.unlink(missing_ok=True)
        if args.zip:
            zip_path = output_dir / "transcript-anonymizer-windows.zip"
            support_files = []
            if args.entrypoint in {"desktop", "both"}:
                guide = project_root / "docs" / "desktop-guide.md"
                if guide.is_file():
                    support_files.append((guide, "docs/desktop-guide.md"))
                sample = output_dir / ".delivery-support" / "synthetic.docx"
                _write_synthetic_sample(sample)
                support_files.append((sample, "synthetic/synthetic.docx"))
            _write_delivery_zip(output_dir, built, zip_path, support_files=tuple(support_files))
            print(f"ZIP: {zip_path}")
        return 0
    except BuildError as error:
        print(f"Build error: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("Build error: local build files could not be read or written.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
