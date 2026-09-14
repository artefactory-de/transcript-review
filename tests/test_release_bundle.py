import json
import zipfile

import pytest

from scripts.release_bundle import (
    MANIFEST,
    apply_update,
    checked_manifest,
    inventory,
    make_release,
    make_split_release,
    safe_name,
)


def fixture(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    for root in (old, new):
        (root / '_internal').mkdir(parents=True)
        (root / '_internal/model.bin').write_bytes(b'unchanged synthetic model')
    (old / 'app.exe').write_bytes(b'old synthetic exe')
    (old / 'removed.txt').write_text('obsolete')
    (new / 'app.exe').write_bytes(b'new synthetic exe')
    (new / 'added.txt').write_text('new resource')
    updater = tmp_path / 'updater.exe'
    updater.write_bytes(b'synthetic updater')
    manifest = make_release(old, tmp_path / 'v1', '1.0.0')
    (old / MANIFEST).write_text(json.dumps(manifest))
    make_release(new, tmp_path / 'v2', '1.1.0', next((tmp_path / 'v1').glob('*.zip')), updater)
    update = tmp_path / 'update'
    with zipfile.ZipFile(next((tmp_path / 'v2').glob('*update.zip'))) as archive:
        archive.extractall(update)
    return old, new, update


def test_update_reconstructs_full_build_and_preserves_old_and_user_files(tmp_path):
    old, new, update = fixture(tmp_path)
    (old / 'private-review.txt').write_text('user data is never copied or removed')
    assert not (update / 'payload/_internal/model.bin').exists()
    target = apply_update(old, update, tmp_path / 'installed')
    assert inventory(target, '1.1.0') == inventory(new, '1.1.0')
    assert not (target / 'removed.txt').exists()
    assert (old / 'removed.txt').exists() and (old / 'private-review.txt').exists()
    assert not (target / 'private-review.txt').exists()


@pytest.mark.parametrize('damage', ['base', 'payload', 'manifest', 'destination'])
def test_update_refuses_mismatch_without_editing_old(tmp_path, damage):
    old, _, update = fixture(tmp_path)
    target = tmp_path / 'installed'
    if damage == 'base':
        (old / 'app.exe').write_bytes(b'changed')
    elif damage == 'payload':
        (update / 'payload/app.exe').write_bytes(b'changed')
    elif damage == 'manifest':
        spec = json.loads((update / 'update.json').read_text())
        spec['base'] = '0' * 64
        (update / 'update.json').write_text(json.dumps(spec))
    else:
        target.mkdir()
    before = (old / 'app.exe').read_bytes()
    with pytest.raises(ValueError):
        apply_update(old, update, target)
    assert (old / 'app.exe').read_bytes() == before
    assert not list(tmp_path.glob('.transcript-update-*'))


@pytest.mark.parametrize('path', ['.', '../escape', '/root', 'C:/escape', 'a\\b', 'a/../b', 'a//b', 'NUL.txt', 'a:stream', 'a.', 'a/COM1', 'x\nfile'])
def test_rejects_unsafe_windows_paths(path):
    with pytest.raises(ValueError):
        safe_name(path)


def test_copy_failure_leaves_old_unchanged_and_cleans_staging(tmp_path, monkeypatch):
    from scripts import release_bundle
    old, _, update = fixture(tmp_path)
    def fail(*args):
        raise OSError('synthetic disk failure')
    monkeypatch.setattr(release_bundle.shutil, 'copyfile', fail)
    with pytest.raises(OSError):
        apply_update(old, update, tmp_path / 'installed')
    assert (old / 'app.exe').read_bytes() == b'old synthetic exe'
    assert not (tmp_path / 'installed').exists()
    assert not list(tmp_path.glob('.transcript-update-*'))


@pytest.mark.parametrize('names', [
    ['app.exe', 'APP.exe'], ['a', 'A/file'], ['RELEASE-MANIFEST.JSON'],
])
def test_manifest_rejects_case_and_directory_collisions(names):
    data = {'format': 1, 'version': '1.0.0', 'files': {
        name: {'bytes': 1, 'sha256': '0' * 64} for name in names}}
    with pytest.raises(ValueError):
        checked_manifest(data)


@pytest.mark.parametrize('field,value', [('version', None), ('sha256', 123)])
def test_manifest_rejects_wrong_value_types(field, value):
    data = {'format': 1, 'version': '1.0.0', 'files': {'app.exe': {'bytes': 1, 'sha256': '0' * 64}}}
    if field == 'version':
        data[field] = value
    else:
        data['files']['app.exe'][field] = value
    with pytest.raises(ValueError):
        checked_manifest(data)


def test_split_release_separates_code_libraries_and_model_parts(tmp_path):
    root = tmp_path / 'app'
    (root / '_internal' / 'model').mkdir(parents=True)
    (root / 'app.exe').write_bytes(b'exe')
    (root / '_internal' / 'library.dll').write_bytes(b'library')
    (root / '_internal' / 'model' / 'config.json').write_bytes(b'{}')
    (root / '_internal' / 'model' / 'model.safetensors').write_bytes(b'0123456789abcdefghij')
    make_split_release(root, tmp_path / 'release', '1.0.0', part_limit=10)
    code = next((tmp_path / 'release').glob('*-code.zip'))
    libraries = next((tmp_path / 'release').glob('*-libraries.zip'))
    model_zips = sorted((tmp_path / 'release').glob('*-model-assets-*.zip'))
    assert len(model_zips) == 2
    with zipfile.ZipFile(code) as archive:
        assert set(archive.namelist()) >= {'app.exe', MANIFEST, '_internal/model-parts.json', 'INSTALL.txt'}
        assert '_internal/library.dll' not in archive.namelist()
    with zipfile.ZipFile(libraries) as archive:
        assert '_internal/library.dll' in archive.namelist()
        assert '_internal/model/model.safetensors' not in archive.namelist()
    assert all(path.stat().st_size < 900 * 1024**2 for path in (tmp_path / 'release').glob('*.zip'))
