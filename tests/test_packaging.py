from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).parents[1] / "scripts" / "build_bundle.py"
    spec = importlib.util.spec_from_file_location("build_bundle_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_windows_release_refuses_before_creating_output(builder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(builder.platform, "system", lambda: "Linux")
    output = tmp_path / "build"

    assert builder.main(["--rules-only", "--output", str(output)]) == 2
    assert not output.exists()
    assert "Refusing a Windows release build" in capsys.readouterr().err


def test_build_requires_reference_python(builder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(builder.platform, "system", lambda: "Windows")
    monkeypatch.setattr(builder.sys, "version_info", (3, 13, 0))

    assert builder.main(["--rules-only", "--output", str(tmp_path / "build")]) == 2
    assert "Python 3.12 is required" in capsys.readouterr().err


def test_model_inventory_is_bounded_to_regular_local_files(builder, tmp_path):
    model = tmp_path / "model"
    (model / "nested").mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "nested" / "weights.bin").write_bytes(b"weights")

    records = builder._model_inventory(model)

    assert [record["path"] for record in records] == ["config.json", "nested/weights.bin"]
    assert all(len(record["sha256"]) == 64 for record in records)


def test_model_manifest_is_json_resource(builder, tmp_path):
    manifest = tmp_path / "src" / "transcript_anonymizer" / "model-assets.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"schema_version": 1, "files": [{"path": "weights.bin", "size": 1, "sha256": "'
        + "a" * 64
        + '"}]}\n',
        encoding="utf-8",
    )

    path, digest = builder._model_manifest(tmp_path)

    assert path == manifest
    assert digest == builder._sha256(manifest)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "files": []},
        {"schema_version": 1},
        {"schema_version": 1, "files": [{"path": "../weights.bin", "size": 1, "sha256": "a" * 64}]},
        {"schema_version": 1, "files": [{"path": "weights.bin", "size": 1, "sha256": "not-a-digest"}]},
        {"schema_version": 1, "files": [{"path": "weights.bin", "size": True, "sha256": "a" * 64}]},
    ],
)
def test_model_manifest_rejects_malformed_file_metadata(builder, tmp_path, payload):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(builder.BuildError, match="model-assets.json|canonical"):
        builder._read_model_manifest(manifest)


def test_model_inventory_must_match_required_assets(builder, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [{"path": "model.safetensors", "size": 7, "sha256": "a" * 64}],
            }
        )
    )
    records = [{"path": "model.safetensors", "bytes": 7, "sha256": "a" * 64}]
    builder._validate_model_inventory(records, manifest)
    with pytest.raises(builder.BuildError):
        builder._validate_model_inventory(
            records + [{"path": "unexpected.docx", "bytes": 1, "sha256": "b" * 64}], manifest
        )
    with pytest.raises(builder.BuildError):
        builder._validate_model_inventory([{**records[0], "sha256": "c" * 64}], manifest)


def test_pyinstaller_command_wires_resource_and_model_modules(builder, tmp_path, monkeypatch):
    output = tmp_path / "output"
    entrypoint = tmp_path / "entrypoint.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    model_manifest = tmp_path / "model-assets.json"
    model_manifest.write_text("{}\n", encoding="utf-8")
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))
        artifact = output / "onedir" / "transcript-anonymizer-onedir"
        artifact.mkdir(parents=True)

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    artifact = builder._run_pyinstaller(
        project_root=tmp_path,
        entrypoint_path=entrypoint,
        output_dir=output,
        mode="onedir",
        model_dir=model,
        model_manifest=model_manifest,
    )

    command = calls[0][0]
    assert artifact.is_dir()
    assert "--onedir" in command
    for module in builder.COLLECT_MODULES:
        position = command.index(module)
        assert command[position - 1] == "--collect-all"
    for module in builder.COLLECT_DATA_MODULES:
        position = command.index(module)
        assert command[position - 1] == "--collect-data"
    for module in builder.HIDDEN_IMPORTS:
        position = command.index(module)
        assert command[position - 1] == "--hidden-import"
    data_args = [command[index + 1] for index, value in enumerate(command) if value == "--add-data"]
    assert any(
        str(model) in value and value.endswith(f"{builder.os.pathsep}model") for value in data_args
    )
    assert any(
        str(model_manifest) in value
        and value.endswith(f"{builder.os.pathsep}transcript_anonymizer")
        for value in data_args
    )


def test_desktop_pyinstaller_command_is_windowed_and_separate(builder, tmp_path, monkeypatch):
    output = tmp_path / "output"
    entrypoint = tmp_path / "desktop-entrypoint.py"
    entrypoint.write_text("pass\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    model_manifest = tmp_path / "model-assets.json"
    model_manifest.write_text("{}\n", encoding="utf-8")
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))
        artifact = output / "desktop" / "onedir" / "transcript-anonymizer-desktop-onedir"
        artifact.mkdir(parents=True)

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    artifact = builder._run_pyinstaller(
        project_root=tmp_path,
        entrypoint_path=entrypoint,
        output_dir=output,
        mode="onedir",
        model_dir=model,
        model_manifest=model_manifest,
        entrypoint="desktop",
    )

    command = calls[0][0]
    assert artifact.is_dir()
    assert "--windowed" in command
    assert "--console" not in command
    assert "--name" in command
    assert command[command.index("--name") + 1] == "transcript-anonymizer-desktop-onedir"


def test_delivery_zip_contains_only_selected_artifacts_and_manifests(builder, tmp_path):
    output = tmp_path / "output"
    artifact = output / "desktop" / "onedir" / "app.exe"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"exe")
    (artifact.parent / "runtime.dll").write_bytes(b"dll")
    manifest = artifact.parent / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    (output / ".pyinstaller").mkdir()
    (output / ".pyinstaller" / "scratch").write_bytes(b"skip")

    zip_path = builder._write_delivery_zip(output, [(artifact.parent, manifest)], output / "delivery.zip")

    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == [
            "desktop/onedir/app.exe",
            "desktop/onedir/manifest.json",
            "desktop/onedir/runtime.dll",
        ]


def test_delivery_zip_requires_explicit_support_allowlist(builder, tmp_path):
    output = tmp_path / "output"
    artifact = output / "desktop" / "onedir"
    artifact.mkdir(parents=True)
    (artifact / "app.exe").write_bytes(b"exe")
    manifest = artifact / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    guide = tmp_path / "desktop-guide.md"
    guide.write_text("synthetic guide\n", encoding="utf-8")
    (output / "private-notes.md").write_text("must not be included\n", encoding="utf-8")

    zip_path = builder._write_delivery_zip(
        output,
        [(artifact, manifest)],
        output / "delivery.zip",
        support_files=((guide, "docs/desktop-guide.md"),),
    )

    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == [
            "desktop/onedir/app.exe",
            "desktop/onedir/manifest.json",
            "docs/desktop-guide.md",
        ]

    with pytest.raises(builder.BuildError, match="support file"):
        builder._write_delivery_zip(
            output,
            [(artifact, manifest)],
            output / "second-delivery.zip",
            support_files=((guide, "../desktop-guide.md"),),
        )


def test_synthetic_sample_matches_safe_cli_text(builder, tmp_path):
    sample = tmp_path / "synthetic.docx"
    builder._write_synthetic_sample(sample)

    from docx import Document

    document = Document(sample)
    assert [paragraph.text for paragraph in document.paragraphs] == list(
        builder._SYNTHETIC_SAMPLE
    )


def test_manifest_remains_unverified_until_windows_qa(builder, tmp_path, monkeypatch):
    artifact = tmp_path / "transcript-anonymizer-onefile.exe"
    artifact.write_bytes(b"candidate")
    monkeypatch.setattr(builder.platform, "system", lambda: "Windows")

    manifest_path = builder._write_manifest(
        artifact=artifact,
        output_dir=tmp_path / "output",
        mode="onefile",
        model_files=[{"path": "config.json", "bytes": 2, "sha256": "x" * 64}],
        model_manifest_sha256="y" * 64,
        rules_only=False,
        host_trial=False,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["windows_artifact_candidate"] is True
    assert manifest["release_eligible"] is False
    assert manifest["acceptance_status"] == "unverified"
    assert manifest["model"]["manifest_resource"] == "transcript_anonymizer/model-assets.json"
    assert manifest["model"]["asset_count"] == 1
    assert manifest["model"]["asset_bytes"] == 2
    assert "files" not in manifest["model"]


def test_wine_manifest_is_not_native_windows_acceptance(builder, tmp_path, monkeypatch):
    artifact = tmp_path / "transcript-anonymizer-desktop-onedir"
    artifact.mkdir()
    (artifact / "app.exe").write_bytes(b"candidate")
    monkeypatch.setattr(builder.platform, "system", lambda: "Windows")
    monkeypatch.setattr(builder, "_running_under_wine", lambda: True)

    manifest_path = builder._write_manifest(
        artifact=artifact,
        output_dir=tmp_path,
        mode="onedir",
        model_files=[],
        model_manifest_sha256="a" * 64,
        rules_only=False,
        host_trial=False,
        entrypoint="desktop",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["under_wine"] is True
    assert manifest["host_trial"] is True
    assert manifest["native_windows_build"] is False
    assert manifest["native_windows_acceptance"] is False
    assert manifest["windows_artifact_candidate"] is False
