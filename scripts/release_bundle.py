"""Create full and base-bound update ZIPs from one verified application tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

MANIFEST = 'release-manifest.json'
MODEL_PARTS = '_internal/model-parts.json'
MODEL_WEIGHTS = '_internal/model/model.safetensors'
SPLIT_LIMIT = 768 * 1024**2
VERSION = re.compile(r'v?\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?\Z')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Unsafe package path')
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or str(path) != value or any(p in {'.', '..'} or p.endswith((' ', '.')) for p in path.parts):
        raise ValueError('Unsafe package path')
    reserved = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)), *(f'lpt{i}' for i in range(1, 10))}
    if any(p.split('.')[0].casefold() in reserved or any(c in p for c in '<>"|?*') or any(ord(c) < 32 for c in p) for p in path.parts):
        raise ValueError('Unsafe Windows path')
    return value


def checked_manifest(data):
    if not isinstance(data, dict) or data.get('format') != 1 or not isinstance(data.get('version'), str) or not VERSION.fullmatch(data['version']):
        raise ValueError('Unsupported release manifest')
    files = data.get('files')
    if not isinstance(files, dict) or not files or len(files) > 20000:
        raise ValueError('Invalid file inventory')
    seen = set()
    for name, record in files.items():
        safe_name(name)
        if name.casefold() == MANIFEST or name.casefold() in seen:
            raise ValueError('Duplicate/reserved package path')
        seen.add(name.casefold())
        if not isinstance(record, dict) or type(record.get('bytes')) is not int or not 0 <= record['bytes'] <= 4 * 1024**3 or not isinstance(record.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', record['sha256']):
            raise ValueError('Invalid file checksum')
    for name in seen:
        if any(str(parent) in seen for parent in PurePosixPath(name).parents):
            raise ValueError('Package file conflicts with a directory')
    if sum(r['bytes'] for r in files.values()) > 8 * 1024**3:
        raise ValueError('Package exceeds size limit')
    return data


def fingerprint(manifest):
    checked_manifest(manifest)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def inventory(root, version):
    files = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlinks are forbidden in packages')
        if path.is_file() and path.relative_to(root).as_posix() != MANIFEST:
            name = path.relative_to(root).as_posix()
            files[name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    return checked_manifest({'format': 1, 'version': version, 'files': files})


def archive_manifest(archive):
    names = archive.namelist()
    if len(names) != len({n.casefold() for n in names}):
        raise ValueError('Duplicate ZIP members')
    if archive.getinfo(MANIFEST).file_size > 8 * 1024**2:
        raise ValueError('Manifest is too large')
    manifest = checked_manifest(json.loads(archive.read(MANIFEST)))
    if set(names) != {*manifest['files'], MANIFEST}:
        raise ValueError('ZIP inventory mismatch')
    for name, record in manifest['files'].items():
        if archive.getinfo(name).file_size != record['bytes']:
            raise ValueError('ZIP size mismatch')
        with archive.open(name) as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != record['sha256']:
                raise ValueError('ZIP checksum mismatch')
    return manifest


def make_release(root, out, version, base=None, updater=None):
    root, out = Path(root), Path(out)
    manifest = inventory(root, version)
    out.mkdir(parents=True, exist_ok=False)
    full = out / f'Transcript-Review-{version}-windows-x64-full.zip'
    raw = json.dumps(manifest, indent=2, sort_keys=True).encode()
    with zipfile.ZipFile(full, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST, raw)
        for name in manifest['files']:
            archive.write(root / name, name)
    with zipfile.ZipFile(full) as archive:
        assert archive_manifest(archive) == manifest
    if base:
        if updater is None:
            raise ValueError('Update executable is required')
        with zipfile.ZipFile(base) as archive:
            previous = archive_manifest(archive)
        changed = [n for n, r in manifest['files'].items() if previous['files'].get(n) != r]
        spec = {'format': 1, 'base': fingerprint(previous), 'target': manifest, 'changed': changed}
        update = out / f'Transcript-Review-{version}-from-{previous["version"]}-update.zip'
        with zipfile.ZipFile(update, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(updater, 'Apply-Update.exe')
            archive.writestr('update.json', json.dumps(spec, indent=2, sort_keys=True))
            for name in changed:
                archive.write(root / name, 'payload/' + name)
            archive.writestr('UPDATE.txt', 'Extract this ZIP to its own folder. Close the application, then run Apply-Update.exe. Select the previous application folder. A verified new installation is created alongside it. The old installation and review folders remain untouched. Do not disable security controls if Windows blocks this unsigned candidate.\n')
    for path in out.glob('*.zip'):
        if path.stat().st_size >= 2 * 1024**3:
            raise ValueError('Release asset exceeds GitHub size limit')
    (out / MANIFEST).write_bytes(raw)
    (out / 'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file()), encoding='utf-8')
    return manifest


def _write_zip(path, files, extra=()):
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in files:
            archive.write(source, name)
        for name, data in extra:
            archive.writestr(name, data)


def make_split_release(root, out, version, *, part_limit=SPLIT_LIMIT):
    """Package executable code, libraries, and model weights separately.

    Every ZIP stays below the download limit. Extract all packages into the
    same fresh folder; the application verifies and rebuilds split weights on
    its first model load.
    """
    root, out = Path(root), Path(out)
    if part_limit <= 0:
        raise ValueError('Invalid split release part limit')
    manifest = inventory(root, version)
    weights = root / MODEL_WEIGHTS
    if not weights.is_file():
        raise ValueError('Expected bundled model weights are missing')
    # A complete release may include the monolithic ZIP as well as these
    # delivery-sized packages.  Existing files still fail closed because ZIPs
    # are opened in exclusive-create mode below.
    out.mkdir(parents=True, exist_ok=True)
    if not out.is_dir():
        raise ValueError('Split release destination is not a directory')
    part_dir = out / '.model-parts'
    part_dir.mkdir()
    parts, index = [], 1
    with weights.open('rb') as source:
        while chunk := source.read(part_limit):
            name = f'model.safetensors.part{index:03d}'
            path = part_dir / name
            path.write_bytes(chunk)
            parts.append({'name': name, 'bytes': len(chunk), 'sha256': digest(path)})
            index += 1
    plan = {'format': 1, 'target': 'model.safetensors', 'bytes': weights.stat().st_size,
            'sha256': digest(weights), 'parts': parts}
    raw_manifest = json.dumps(manifest, indent=2, sort_keys=True).encode()
    instructions = (
        'Extract Code, Libraries, and every Model-Assets ZIP into one new folder. '
        'Keep the folder writable. Start transcript-anonymizer-desktop-onedir.exe. '
        'The first model use verifies and rebuilds the local model; this needs temporary free disk space.\n'
    )
    code = [(name, root / name) for name in manifest['files'] if not name.startswith('_internal/')]
    libraries = [(name, root / name) for name in manifest['files'] if name.startswith('_internal/') and name != MODEL_WEIGHTS]
    split_packages = [out / f'Transcript-Review-{version}-windows-x64-code.zip']
    _write_zip(split_packages[0], code, [
        (MANIFEST, raw_manifest), (MODEL_PARTS, json.dumps(plan, indent=2, sort_keys=True)), ('INSTALL.txt', instructions)])
    split_packages.append(out / f'Transcript-Review-{version}-windows-x64-libraries.zip')
    _write_zip(split_packages[-1], libraries)
    for part in parts:
        package = out / f'Transcript-Review-{version}-windows-x64-model-assets-{part["name"][-3:]}.zip'
        _write_zip(package, [(f'_internal/model-parts/{part["name"]}', part_dir / part['name'])])
        split_packages.append(package)
    shutil.rmtree(part_dir)
    for path in split_packages:
        if path.stat().st_size >= 900 * 1024**2:
            raise ValueError('Split release asset exceeds the 900 MiB delivery limit')
    (out / MANIFEST).write_bytes(raw_manifest)
    (out / 'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p.name != 'SHA256SUMS.txt'), encoding='utf-8')
    return manifest


def apply_update(installed, update_dir, destination):
    """Verify first, build alongside, then atomically rename. Never edit installed."""
    installed, update_dir, destination = map(Path, (installed, update_dir, destination))
    if installed.is_symlink() or update_dir.is_symlink() or destination.exists() or destination.is_symlink():
        raise ValueError('Choose real source folders and a new destination')
    if installed.resolve() == destination.resolve() or installed.resolve() in destination.resolve().parents or update_dir.resolve() in destination.resolve().parents:
        raise ValueError('Destination must be outside source folders')
    old = checked_manifest(json.loads((installed / MANIFEST).read_text(encoding='utf-8')))
    spec = json.loads((update_dir / 'update.json').read_text(encoding='utf-8'))
    if spec.get('format') != 1 or spec.get('base') != fingerprint(old):
        raise ValueError('This update does not support the selected installation')
    target = checked_manifest(spec['target'])
    changed = {n for n, r in target['files'].items() if old['files'].get(n) != r}
    if not isinstance(spec.get('changed'), list) or set(spec['changed']) != changed or len(spec['changed']) != len(changed):
        raise ValueError('Update inventory mismatch')
    for name, record in old['files'].items():
        source = installed / name
        if not source.is_file() or source.resolve() != installed.resolve() / name or source.stat().st_size != record['bytes'] or digest(source) != record['sha256']:
            raise ValueError('Existing installation is modified or incomplete')
    sources = {}
    for name, record in target['files'].items():
        parent = update_dir / 'payload' if name in changed else installed
        source = parent / name
        if not source.is_file() or source.resolve() != parent.resolve() / name or source.stat().st_size != record['bytes'] or digest(source) != record['sha256']:
            raise ValueError('Update payload is modified or incomplete')
        sources[name] = source
    staging = Path(tempfile.mkdtemp(prefix='.transcript-update-', dir=destination.parent))
    try:
        for name, source in sources.items():
            output = staging / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
        if inventory(staging, target['version']) != target:
            raise ValueError('Updated installation failed verification')
        (staging / MANIFEST).write_text(json.dumps(target, indent=2, sort_keys=True), encoding='utf-8')
        # Windows rename refuses an existing destination; check again on all hosts.
        if destination.exists():
            raise ValueError('Destination appeared during update')
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--updater', type=Path)
    args = parser.parse_args()
    make_release(args.root, args.out, args.version, args.base, args.updater)


if __name__ == '__main__':
    main()
