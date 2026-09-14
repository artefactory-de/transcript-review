from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).parents[1] / "scripts" / "build_bundle.py"
    spec = importlib.util.spec_from_file_location("build_bundle_runtime_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pe_x64(payload: bytes = b"runtime") -> bytes:
    data = bytearray(256)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<H", data, 128 + 4, 0x8664)
    struct.pack_into("<H", data, 128 + 24, 0x20B)
    data.extend(payload)
    return bytes(data)


def _runtime_dir(builder, root: Path) -> Path:
    runtime = root / "Microsoft VC runtime ä"
    runtime.mkdir(parents=True)
    for name in builder.RUNTIME_REQUIRED_DLLS:
        (runtime / name.lower()).write_bytes(_pe_x64(name.encode()))
    return runtime


def test_runtime_inventory_is_case_insensitive_and_hashes_allowlisted_files(builder, tmp_path):
    runtime = _runtime_dir(builder, tmp_path)
    (runtime / "ConCrT140.DLL").write_bytes(_pe_x64(b"optional"))
    (runtime / "LICENSE.txt").write_text("official redistribution notice\n", encoding="utf-8")

    records = builder._runtime_inventory(runtime)

    assert {record["path"].casefold() for record in records} == {
        "msvcp140.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "concrt140.dll",
        "license.txt",
    }
    assert all(len(record["sha256"]) == 64 for record in records)
    assert {record["kind"] for record in records} == {"dll", "provenance"}


def test_runtime_inventory_rejects_missing_unknown_duplicate_and_non_x64_files(builder, tmp_path):
    runtime = _runtime_dir(builder, tmp_path)
    (runtime / "extra.dll").write_bytes(_pe_x64())
    with pytest.raises(builder.BuildError, match="allowlist"):
        builder._runtime_inventory(runtime)

    runtime = tmp_path / "missing"
    runtime.mkdir()
    for name in builder.RUNTIME_REQUIRED_DLLS[:2]:
        (runtime / name).write_bytes(_pe_x64())
    with pytest.raises(builder.BuildError, match="missing"):
        builder._runtime_inventory(runtime)

    runtime = _runtime_dir(builder, tmp_path / "bad-pe")
    (runtime / "msvcp140.dll").write_bytes(b"MZ" + b"not a PE")
    with pytest.raises(builder.BuildError, match="PE"):
        builder._runtime_inventory(runtime)


def test_runtime_inventory_rejects_case_collisions_and_symlinks(builder, tmp_path):
    runtime = _runtime_dir(builder, tmp_path)
    (runtime / "MSVCP140.DLL").write_bytes(_pe_x64())
    with pytest.raises(builder.BuildError, match="duplicate"):
        builder._runtime_inventory(runtime)

    runtime = _runtime_dir(builder, tmp_path / "symlink")
    link = runtime / "LICENSE.txt"
    link.symlink_to(runtime / "msvcp140.dll")
    with pytest.raises(builder.BuildError, match="symbolic"):
        builder._runtime_inventory(runtime)


def test_runtime_pyinstaller_args_use_binary_root_and_data_root(builder, tmp_path):
    runtime = _runtime_dir(builder, tmp_path)
    (runtime / "runtime-provenance.json").write_text("{}\n", encoding="utf-8")

    args = builder._runtime_pyinstaller_args(runtime)
    binary_values = [args[index + 1] for index, value in enumerate(args) if value == "--add-binary"]
    data_values = [args[index + 1] for index, value in enumerate(args) if value == "--add-data"]

    assert len(binary_values) == 3
    assert all(value.endswith(f"{builder.os.pathsep}.") for value in binary_values + data_values)
    assert len(data_values) == 1
    assert "runtime-provenance.json" in data_values[0]


def test_runtime_staging_is_reusable_but_never_overwrites(builder, tmp_path):
    runtime = _runtime_dir(builder, tmp_path)
    artifact = tmp_path / "frozen" / "app"
    artifact.mkdir(parents=True)

    records = builder._stage_runtime_assets(runtime, artifact)
    assert len(records) == 3
    # Replaying the same validated input is idempotent for post-staging.
    builder._stage_runtime_assets(runtime, artifact, records)

    (artifact / "msvcp140.dll").write_bytes(b"different")
    with pytest.raises(builder.BuildError, match="different"):
        builder._stage_runtime_assets(runtime, artifact, records)
    builder._stage_runtime_assets(runtime, artifact, records, replace_existing=True)
    assert (artifact / "msvcp140.dll").read_bytes() == (runtime / "msvcp140.dll").read_bytes()

    (runtime / "msvcp140.dll").write_bytes(_pe_x64(b"changed"))
    with pytest.raises(builder.BuildError, match="changed"):
        builder._stage_runtime_assets(runtime, artifact, records, replace_existing=True)


def test_onedir_runtime_artifact_must_contain_hashed_inputs(builder, tmp_path):
    runtime = _runtime_dir(builder, tmp_path)
    records = builder._runtime_inventory(runtime)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload_root = artifact / "_internal"
    payload_root.mkdir()
    (payload_root / "msvcp140.dll").write_bytes((runtime / "msvcp140.dll").read_bytes())

    with pytest.raises(builder.BuildError, match="missing"):
        builder._verify_runtime_artifact(artifact, records)

    for record in records[1:]:
        (payload_root / str(record["path"])).write_bytes(
            (runtime / str(record["path"])).read_bytes()
        )
    builder._verify_runtime_artifact(artifact, records)


def test_manifest_records_runtime_inventory_without_copying_numpy_files(builder, tmp_path, monkeypatch):
    runtime = _runtime_dir(builder, tmp_path)
    artifact = tmp_path / "app.exe"
    artifact.write_bytes(b"candidate")
    monkeypatch.setattr(builder.platform, "system", lambda: "Windows")

    records = builder._runtime_inventory(runtime)
    manifest_path = builder._write_manifest(
        artifact=artifact,
        output_dir=tmp_path,
        mode="onefile",
        model_files=[],
        model_manifest_sha256="a" * 64,
        rules_only=False,
        host_trial=False,
        runtime_dir=runtime,
        runtime_files=records,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime"]["included"] is True
    assert "source_dir" not in manifest["runtime"]
    assert manifest["runtime"]["required"] == list(builder.RUNTIME_REQUIRED_DLLS)
    assert [item["path"] for item in manifest["runtime"]["files"]] == [
        item["path"] for item in records
    ]


def test_windows_model_build_requires_explicit_runtime_directory(builder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(builder.platform, "system", lambda: "Windows")

    assert builder.main(["--model", str(tmp_path / "model")]) == 2
    assert "--runtime-dir" in capsys.readouterr().err
