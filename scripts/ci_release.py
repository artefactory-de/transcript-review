"""Native Windows release orchestration. Builds only; never publishes."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from release_bundle import VERSION, make_split_release


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'win32' or not VERSION.fullmatch(args.version):
        parser.error('A native Windows builder and valid version are required')
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(work).free < 8 * 1024**3:
        raise RuntimeError('At least 8 GiB free staging space is required')
    run(sys.executable, 'scripts/fetch_model.py', '--output', work / 'model')
    run(sys.executable, 'scripts/fetch_runtime.py', '--output', work / 'runtime', '--scratch', work / 'runtime-download')
    run(sys.executable, 'scripts/build_bundle.py', '--model', work / 'model', '--runtime-dir', work / 'runtime',
        '--mode', 'onedir', '--entrypoint', 'desktop', '--output', work / 'build')
    folder = work / 'build/desktop/onedir/transcript-anonymizer-desktop-onedir'
    exe = folder / 'transcript-anonymizer-desktop-onedir.exe'
    assert exe.is_file(), 'Expected desktop bundle not found'
    # Windowed EXEs do not provide reliable stdout; self-test produces a file.
    smoke = work / 'frozen-smoke'
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    run(exe, '--self-test', smoke, cwd=work, env=environment, timeout=600)
    report = json.loads((smoke / 'result.json').read_text())
    assert report['ok'] and report['model_enabled'] and report['model_findings'] > 0
    run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onefile', '--windowed',
        '--name', 'Apply-Update', '--distpath', work / 'updater', '--workpath', work / 'updater-work',
        '--specpath', work / 'updater-spec', 'scripts/update_launcher.py')
    shutil.copyfile(folder.parent / 'manifest.json', folder / 'build-manifest.json')
    shutil.copyfile('docs/desktop-guide.md', folder / 'desktop-guide.md')
    shutil.copyfile('docs/releases.md', folder / 'releases.md')
    shutil.copyfile('THIRD_PARTY_NOTICES.md', folder / 'THIRD_PARTY_NOTICES.md')
    if Path('LICENSE').is_file():
        shutil.copyfile('LICENSE', folder / 'LICENSE')
    # Record a dependency licence inventory without leaking runner directories.
    from importlib.metadata import distributions
    records = []
    for dist in distributions():
        name = re.sub(r'[^A-Za-z0-9_.-]', '_', dist.metadata['Name'])
        records.append({'name': name, 'version': dist.version,
                        'license': dist.metadata.get('License-Expression') or dist.metadata.get('License', '')})
        for item in dist.files or []:
            if any(part.lower() in {'licenses', 'license', 'copying', 'notice'} or part.lower().startswith(('license.', 'copying.', 'notice.')) for part in item.parts):
                source = Path(dist.locate_file(item))
                if source.is_file():
                    # Distribution records can contain parent or absolute paths.
                    # Preserve only safe component names inside our licence tree.
                    parts = [re.sub(r'[^A-Za-z0-9_.-]', '_', part) for part in item.parts if part not in {'.', '..'}]
                    target = folder / 'licenses' / name
                    for part in parts:
                        target = target / (part or '_')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
    (folder / 'dependency-licenses.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    if args.base:
        raise RuntimeError('Split releases do not yet support update packages')
    make_split_release(folder, work / 'release', args.version)
    (work / 'release/verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    from release_bundle import digest
    release = work / 'release'
    (release / 'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(release.iterdir()) if p.is_file() and p.name != 'SHA256SUMS.txt'), encoding='utf-8')


if __name__ == '__main__':
    main()
